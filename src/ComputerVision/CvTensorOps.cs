using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision;

/// <summary>
/// Tape-visible tensor operations shared by the computer-vision detection and OCR models.
/// </summary>
/// <remarks>
/// <para>
/// These models were written with hand-rolled scalar loops: each helper read elements out one at a
/// time and wrote a freshly allocated tensor. Arithmetically fine, but every such loop severs the
/// autodiff tape, so any trainable layer upstream of it silently received no gradient.
/// </para>
/// <para>
/// Every operation here is composed from engine primitives that record themselves on the tape, and
/// each reproduces the EXACT semantics of the loop it replaces - including the non-standard ones.
/// <see cref="ResizeBilinearAsymmetric"/> uses the asymmetric source mapping
/// <c>src = dst * in / out</c>, not the half-pixel convention of a generic interpolate, and
/// <see cref="MaxPool2x2Ceil"/> keeps the partial edge window rather than dropping it. Swapping in a
/// convenient engine op with a different convention would have shifted every feature map.
/// </para>
/// <para>
/// Resampling and padding are expressed as index selection with precomputed integer indices. A
/// selection is an exact copy, so a remap built from it matches the scalar loop bit for bit, and its
/// backward pass is a scatter-add, which is the correct gradient of a gather.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
internal static class CvTensorOps<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Nearest-neighbour resize of an NCHW tensor using <c>src = min(dst * in / out, in - 1)</c>
    /// on each spatial axis (integer division). This is the convention the neck and head
    /// <c>ResizeToMatch</c> helpers used.
    /// </summary>
    public static Tensor<T> ResizeNearest(Tensor<T> x, int targetH, int targetW)
    {
        int srcH = x.Shape[2];
        int srcW = x.Shape[3];
        if (srcH == targetH && srcW == targetW)
        {
            return x;
        }

        // The engine's nearest rule is floor(dst * in / out), the same index; its kernel copies from per-axis tables
        // where two gathers materialized an intermediate and scattered back through two index tensors.
        return Engine.Interpolate(x, new[] { targetH, targetW }, InterpolateMode.Nearest);
    }

    /// <summary>
    /// Nearest-neighbour 2x upsample (each source pixel becomes a 2x2 block).
    /// </summary>
    public static Tensor<T> Upsample2xNearest(Tensor<T> x)
        => ResizeNearest(x, x.Shape[2] * 2, x.Shape[3] * 2);

    /// <summary>
    /// Bilinear resize of an NCHW tensor with the ASYMMETRIC source mapping
    /// <c>src = dst * in / out</c> (no half-pixel offset), clamping the upper neighbour to the last
    /// row or column. Separable: interpolate along width, then along height, which evaluates
    /// <c>wy0*(wx0*v00 + wx1*v01) + wy1*(wx0*v10 + wx1*v11)</c> in the same order as the loop it
    /// replaces.
    /// </summary>
    public static Tensor<T> ResizeBilinearAsymmetric(Tensor<T> x, int targetH, int targetW)
    {
        int srcH = x.Shape[2];
        int srcW = x.Shape[3];

        var (x0, x1, wx0, wx1) = BilinearTaps(srcW, targetW);
        var (y0, y1, wy0, wy1) = BilinearTaps(srcH, targetH);

        // Along width: [N, C, srcH, targetW].
        var left = Select(x, x0, 3);
        var right = Select(x, x1, 3);
        var alongW = Engine.TensorAdd(
            Engine.TensorMultiply(left, Broadcast(wx0, Dims(left), 3)),
            Engine.TensorMultiply(right, Broadcast(wx1, Dims(right), 3)));

        // Along height: [N, C, targetH, targetW].
        var top = Select(alongW, y0, 2);
        var bottom = Select(alongW, y1, 2);
        return Engine.TensorAdd(
            Engine.TensorMultiply(Broadcast(wy0, Dims(top), 2), top),
            Engine.TensorMultiply(Broadcast(wy1, Dims(bottom), 2), bottom));
    }

    /// <summary>
    /// 2x2, stride-2 max pooling in CEIL mode: an odd-sized input keeps its partial last window,
    /// whose maximum is taken over the in-bounds cells only.
    /// </summary>
    /// <remarks>
    /// Implemented by replicating the last row and column when the extent is odd, then pooling in
    /// floor mode. A replicated cell duplicates a value already inside the same window, so it can
    /// never change that window's maximum - which is exactly "max over the in-bounds cells".
    /// </remarks>
    public static Tensor<T> MaxPool2x2Ceil(Tensor<T> x)
    {
        int h = x.Shape[2];
        int w = x.Shape[3];
        var padded = x;
        if (h % 2 == 1)
        {
            padded = Select(padded, ClampedRange(0, h + 1, h), 2);
        }

        if (w % 2 == 1)
        {
            padded = Select(padded, ClampedRange(0, w + 1, w), 3);
        }

        return Engine.MaxPool2DWithIndices(padded, new[] { 2, 2 }, new[] { 2, 2 }, out _);
    }

    /// <summary>
    /// Max pooling with window and stride both equal to <c>(kernelH, kernelW)</c>, in FLOOR mode:
    /// trailing rows or columns that do not fill a whole window are dropped.
    /// </summary>
    public static Tensor<T> MaxPoolFloor(Tensor<T> x, int kernelH, int kernelW)
        => Engine.MaxPool2DWithIndices(x, new[] { kernelH, kernelW }, new[] { kernelH, kernelW }, out _);

    /// <summary>
    /// Stride-1 "same" max pooling with a square window of odd size <paramref name="kernelSize"/>:
    /// the output has the input's spatial size and each window's maximum is taken over the cells that
    /// fall inside the image (out-of-bounds positions are ignored, not treated as zero).
    /// </summary>
    public static Tensor<T> MaxPoolSame(Tensor<T> x, int kernelSize)
        => MaxPoolPadded(x, kernelSize, 1, kernelSize / 2);

    /// <summary>
    /// Max pooling with a square window, a stride and symmetric padding, where padded positions are
    /// ignored rather than treated as zero - PyTorch's <c>nn.MaxPool2d(kernel, stride, padding)</c>
    /// in floor mode.
    /// </summary>
    public static Tensor<T> MaxPoolPadded(Tensor<T> x, int kernelSize, int stride, int padding)
    {
        if (padding < 0 || padding >= kernelSize)
        {
            throw new ArgumentOutOfRangeException(nameof(padding), padding,
                $"Padding must lie in [0, kernelSize) = [0, {kernelSize}).");
        }

        // The engine pool skips padded positions (PyTorch's -inf padding) and records the padded backward.
        return Engine.MaxPool2D(x, kernelSize, stride, padding);
    }

    /// <summary>
    /// Normalises each channel of an NCHW tensor with the statistics of the CURRENT batch
    /// (biased variance, no affine parameters): <c>(x - mean_c) / sqrt(var_c + eps)</c>.
    /// </summary>
    public static Tensor<T> BatchStatisticsNorm(Tensor<T> x, double epsilon)
    {
        var axes = new[] { 0, 2, 3 };
        var mean = Engine.ReduceMean(x, axes, true);
        var centered = Engine.TensorSubtract(x, Engine.TensorBroadcastTo(mean, Dims(x)));
        var variance = Engine.ReduceMean(Engine.TensorMultiply(centered, centered), axes, true);
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(epsilon)));
        return Engine.TensorDivide(centered, Engine.TensorBroadcastTo(std, Dims(x)));
    }

    /// <summary>
    /// Concatenates NCHW tensors along the channel axis.
    /// </summary>
    public static Tensor<T> ConcatChannels(Tensor<T> first, Tensor<T> second)
        => Engine.TensorConcatenate(new[] { first, second }, 1);

    /// <summary>
    /// Flattens the spatial axes of an NCHW tensor into a token sequence <c>[N, H*W, C]</c>, in
    /// row-major spatial order.
    /// </summary>
    public static Tensor<T> FlattenSpatial(Tensor<T> x)
    {
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        return Engine.Reshape(Engine.TensorPermute(x, new[] { 0, 2, 3, 1 }), new[] { n, h * w, c });
    }

    /// <summary>
    /// Inverse of <see cref="FlattenSpatial"/>: <c>[N, H*W, C]</c> back to <c>[N, C, H, W]</c>.
    /// </summary>
    public static Tensor<T> UnflattenSpatial(Tensor<T> tokens, int height, int width)
    {
        int n = tokens.Shape[0], c = tokens.Shape[2];
        return Engine.TensorPermute(Engine.Reshape(tokens, new[] { n, height, width, c }), new[] { 0, 3, 1, 2 });
    }

    /// <summary>
    /// Layer normalisation over the last axis with learnable scale and shift:
    /// <c>gamma * (x - mean) / sqrt(var + eps) + beta</c>, biased variance.
    /// </summary>
    public static Tensor<T> LayerNormLastAxis(Tensor<T> x, Tensor<T> gamma, Tensor<T> beta, double epsilon)
        => Engine.LayerNorm(x, gamma, beta, epsilon, out _, out _);

    /// <summary>
    /// Multi-head scaled dot-product attention over <c>[N, L, D]</c> sequences. Heads are the
    /// contiguous <c>D / numHeads</c> slices of the model dimension, softmax runs over the keys,
    /// and the heads are concatenated back in order. No projections: the caller applies those.
    /// </summary>
    /// <param name="query">Queries <c>[N, Lq, D]</c>.</param>
    /// <param name="key">Keys <c>[N, Lk, D]</c>.</param>
    /// <param name="value">Values <c>[N, Lk, D]</c>.</param>
    /// <param name="numHeads">Number of heads; must divide <c>D</c>.</param>
    /// <param name="scale">Score scale, normally <c>1 / sqrt(D / numHeads)</c>.</param>
    /// <param name="causal">When true, query <c>i</c> attends only to keys <c>j &lt;= i</c>.</param>
    /// <param name="scoreBias">Optional additive bias on the scaled scores, broadcastable to
    /// <c>[N, H, Lq, Lk]</c> - for example <see cref="RelativePositionBias"/>.</param>
    /// <returns>The attended values <c>[N, Lq, D]</c>.</returns>
    public static Tensor<T> MultiHeadAttention(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, int numHeads, double scale, bool causal = false,
        Tensor<T>? scoreBias = null)
    {
        int n = query.Shape[0], lq = query.Shape[1], d = query.Shape[2], lk = key.Shape[1];
        int headDim = d / numHeads;

        var q = SplitHeads(query, numHeads, headDim);
        var k = SplitHeads(key, numHeads, headDim);
        var v = SplitHeads(value, numHeads, headDim);

        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(q, Engine.TensorPermute(k, new[] { 0, 1, 3, 2 })), NumOps.FromDouble(scale));

        if (scoreBias is not null)
        {
            scores = Engine.TensorAdd(scores, Engine.TensorBroadcastTo(scoreBias, Dims(scores)));
        }

        if (causal)
        {
            // Additive mask: a large negative score gives the masked key an attention weight that
            // underflows to exactly zero after the softmax, matching a loop that skips j > i.
            // Key j is blocked for query i exactly when j > i: compare a column ramp against a row ramp.
            var rowIndex = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<T>(0, lq, 1), new[] { lq, 1 }), new[] { lq, lk });
            var colIndex = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<T>(0, lk, 1), new[] { 1, lk }), new[] { lq, lk });
            var blockedMask = Engine.TensorGreaterThan(colIndex, rowIndex);
            var mask = Engine.Reshape(
                Engine.TensorWhere(blockedMask, Fill(new[] { lq, lk }, NumOps.FromDouble(-1e30)), new Tensor<T>(new[] { lq, lk })),
                new[] { 1, 1, lq, lk });

            scores = Engine.TensorAdd(scores, Engine.TensorBroadcastTo(mask, Dims(scores)));
        }

        var weights = Engine.Softmax(scores, -1);
        var attended = Engine.TensorMatMul(weights, v); // [N, H, Lq, hd]
        return Engine.Reshape(Engine.TensorPermute(attended, new[] { 0, 2, 1, 3 }), new[] { n, lq, d });
    }

    private static Tensor<T> SplitHeads(Tensor<T> x, int numHeads, int headDim)
    {
        int n = x.Shape[0], l = x.Shape[1];
        return Engine.TensorPermute(Engine.Reshape(x, new[] { n, l, numHeads, headDim }), new[] { 0, 2, 1, 3 });
    }

    /// <summary>
    /// Pads a <c>[rows, width]</c> tensor with zero rows to exactly <paramref name="count"/> rows.
    /// </summary>
    /// <remarks>
    /// A two-stage detector's per-RoI heads have one row per proposal, and how many proposals survive NMS depends on
    /// the scores. Padding them to the proposal cap gives the raw prediction a layout fixed by the input size alone.
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when <paramref name="rows"/> already has more rows.</exception>
    public static Tensor<T> PadRows(Tensor<T> rows, int count)
    {
        int present = rows.Shape[0];
        if (present > count)
            throw new InvalidOperationException($"{present} rows exceed the fixed count of {count}.");
        return present == count
            ? rows
            : Engine.TensorConstantPad(rows, new[] { 0, 0, 0, count - present }, NumOps.Zero);
    }

    /// <summary>
    /// Concatenates every head's raw output into one tensor. When all outputs share a leading
    /// (per-image) dimension N, each is flattened to <c>[N, -1]</c> and the result is
    /// <c>[N, total]</c>; when they do not - a two-stage detector's per-RoI heads beside its per-image
    /// RPN maps - everything is flattened into <c>[1, total]</c>. A single output is returned unchanged.
    /// </summary>
    public static Tensor<T> ConcatenateOutputs(IReadOnlyList<Tensor<T>> outputs)
    {
        if (outputs.Count == 0)
        {
            return new Tensor<T>(new[] { 1, 0 });
        }

        if (outputs.Count == 1)
        {
            return outputs[0];
        }

        bool sharedLeading = true;
        for (int i = 1; i < outputs.Count && sharedLeading; i++)
        {
            sharedLeading = outputs[i].Shape[0] == outputs[0].Shape[0] && outputs[i].Shape[0] > 0;
        }

        var flat = new Tensor<T>[outputs.Count];
        for (int i = 0; i < outputs.Count; i++)
        {
            int leading = sharedLeading ? outputs[i].Shape[0] : 1;
            flat[i] = Engine.Reshape(outputs[i], new[] { leading, outputs[i].Length / leading });
        }

        return Engine.TensorConcatenate(flat, 1);
    }

    /// <summary>
    /// Builds a Swin relative-position bias <c>[1, H, L, L]</c> from a learnable table
    /// <c>[R, H]</c> and an index map <c>index[i, j]</c> into its rows. The lookup is a gather, so the
    /// table receives gradients.
    /// </summary>
    public static Tensor<T> RelativePositionBias(Tensor<T> table, int[,] index)
    {
        int lq = index.GetLength(0), lk = index.GetLength(1), heads = table.Shape[1];
        var flat = new int[lq * lk];
        for (int i = 0; i < lq; i++)
        {
            for (int j = 0; j < lk; j++)
            {
                flat[(i * lk) + j] = index[i, j];
            }
        }

        var gathered = Select(table, flat, 0);                                     // [L*L, H]
        return Engine.Reshape(Engine.TensorPermute(gathered, new[] { 1, 0 }), new[] { 1, heads, lq, lk });
    }

    /// <summary>
    /// Rolls a <c>[N, H, W, C]</c> map by <paramref name="shift"/> along both spatial axes with
    /// wrap-around: <c>out[i, j] = x[(i - shift) mod H, (j - shift) mod W]</c> (Swin's cyclic shift).
    /// </summary>
    public static Tensor<T> CyclicShift(Tensor<T> x, int shift)
        => Engine.TensorRoll(x, new[] { shift, shift }, new[] { 1, 2 });

    /// <summary>
    /// Partitions a <c>[N, H, W, C]</c> map into non-overlapping <paramref name="windowSize"/>
    /// squares, zero-padding the bottom and right edges up to a multiple of the window size.
    /// Returns <c>[N * nH * nW, windowSize^2, C]</c> with windows in row-major order per image and
    /// tokens in row-major order per window.
    /// </summary>
    public static (Tensor<T> Windows, int WindowsH, int WindowsW) WindowPartition(Tensor<T> x, int windowSize)
    {
        int n = x.Shape[0], h = x.Shape[1], w = x.Shape[2], c = x.Shape[3];
        int padH = (windowSize - (h % windowSize)) % windowSize;
        int padW = (windowSize - (w % windowSize)) % windowSize;

        var padded = ZeroPadBottomRight(x, padH, padW);

        int wh = (h + padH) / windowSize, ww = (w + padW) / windowSize;
        var blocks = Engine.Reshape(padded, new[] { n, wh, windowSize, ww, windowSize, c });
        var ordered = Engine.TensorPermute(blocks, new[] { 0, 1, 3, 2, 4, 5 });   // [N, wh, ww, ws, ws, C]
        return (Engine.Reshape(ordered, new[] { n * wh * ww, windowSize * windowSize, c }), wh, ww);
    }

    /// <summary>
    /// Inverse of <see cref="WindowPartition"/>: reassembles windows into a <c>[N, H, W, C]</c> map
    /// and drops the padding.
    /// </summary>
    public static Tensor<T> WindowReverse(
        Tensor<T> windows, int windowsH, int windowsW, int batch, int height, int width, int windowSize)
    {
        int c = windows.Shape[2];
        var blocks = Engine.Reshape(windows, new[] { batch, windowsH, windowsW, windowSize, windowSize, c });
        var ordered = Engine.TensorPermute(blocks, new[] { 0, 1, 3, 2, 4, 5 });   // [N, wh, ws, ww, ws, C]
        var full = Engine.Reshape(ordered, new[] { batch, windowsH * windowSize, windowsW * windowSize, c });
        if (full.Shape[1] == height && full.Shape[2] == width)
        {
            return full;
        }

        return Engine.TensorSlice(full, new[] { 0, 0, 0, 0 }, new[] { batch, height, width, c });
    }

    /// <summary>
    /// Applies a row-wise layer (a linear map, typically) independently to every position of a
    /// <c>[..., features]</c> tensor by folding the leading axes into one batch axis and unfolding the
    /// result. Replaces the copy-one-row, forward, copy-back loops, which were slow and severed the tape.
    /// </summary>
    public static Tensor<T> Tokenwise(Tensor<T> x, Func<Tensor<T>, Tensor<T>> rowwise)
    {
        int rank = x.Shape.Length;
        if (rank <= 2)
        {
            return rowwise(x);
        }

        int rows = 1;
        for (int d = 0; d < rank - 1; d++)
        {
            rows *= x.Shape[d];
        }

        var result = rowwise(Engine.Reshape(x, new[] { rows, x.Shape[rank - 1] }));
        var outShape = new int[rank];
        for (int d = 0; d < rank - 1; d++)
        {
            outShape[d] = x.Shape[d];
        }

        outShape[rank - 1] = result.Shape[1];
        return Engine.Reshape(result, outShape);
    }

    /// <summary>
    /// Swin patch merging on a <c>[N, H, W, C]</c> map: zero-pads odd sides to even, then
    /// concatenates each 2x2 quad's tokens along channels in the order (r0,c0), (r0,c1), (r1,c0),
    /// (r1,c1), giving <c>[N, (H/2)*(W/2), 4C]</c> with quads in row-major order.
    /// </summary>
    public static Tensor<T> PatchMerge2x2(Tensor<T> x)
    {
        int n = x.Shape[0], h = x.Shape[1], w = x.Shape[2], c = x.Shape[3];
        var padded = ZeroPadBottomRight(x, h & 1, w & 1);
        int newH = (h + (h & 1)) / 2, newW = (w + (w & 1)) / 2;
        var quads = Engine.Reshape(padded, new[] { n, newH, 2, newW, 2, c });
        var ordered = Engine.TensorPermute(quads, new[] { 0, 1, 3, 2, 4, 5 });    // [N, newH, newW, 2, 2, C]
        return Engine.Reshape(ordered, new[] { n, newH * newW, 4 * c });
    }

    /// <summary>
    /// Zero-pads a <c>[N, H, W, C]</c> map with <paramref name="padH"/> rows at the bottom and
    /// <paramref name="padW"/> columns at the right, by concatenating constant zero blocks (which the
    /// tape treats as constants, so the gradient passes straight through to the original cells).
    /// </summary>
    public static Tensor<T> ZeroPadBottomRight(Tensor<T> x, int padH, int padW)
    {
        int n = x.Shape[0], h = x.Shape[1], w = x.Shape[2], c = x.Shape[3];
        var padded = x;
        if (padH > 0)
        {
            padded = Engine.TensorConcatenate(new[] { padded, new Tensor<T>(new[] { n, padH, w, c }) }, 1);
        }

        if (padW > 0)
        {
            padded = Engine.TensorConcatenate(new[] { padded, new Tensor<T>(new[] { n, h + padH, padW, c }) }, 2);
        }

        return padded;
    }

    /// <summary>
    /// RoIAlign: pools each region of interest into an <c>outputSize x outputSize</c> grid by averaging
    /// <c>samplingRatio^2</c> bilinear samples per bin, over an NCHW feature map - torchvision's <c>roi_align</c>
    /// with <c>aligned = false</c>, as its <c>MultiScaleRoIAlign</c> runs it for Faster, Mask and Cascade R-CNN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The boxes are constants: as in torchvision, the gradient flows into the features, not the box coordinates.
    /// A RoI is at least one feature cell on each side, a sample outside <c>[-1, size]</c> contributes zero but still
    /// counts in its bin's average, and coordinates in <c>(-1, 0]</c> clamp to the first cell.
    /// </para>
    /// <para>
    /// The engine kernel builds each RoI's bilinear taps once and gathers them per channel. This used to be composed
    /// from GridSample, a batched product and two permutes, which the tape held as four RoI-sized intermediates (about
    /// 850 MB over five FasterRCNN training steps), and it averaged only the in-bounds samples, unlike torchvision.
    /// </para>
    /// </remarks>
    /// <param name="features">Feature map <c>[N, C, H, W]</c>.</param>
    /// <param name="boxes">Per-RoI <c>(x1, y1, x2, y2)</c> in image coordinates, length <c>4 * R</c>.</param>
    /// <param name="batchIndices">Per-RoI image index into <paramref name="features"/>, length <c>R</c>.</param>
    /// <param name="spatialScale">Image-to-feature-map scale.</param>
    /// <param name="outputSize">Pooled grid side.</param>
    /// <param name="samplingRatio">Samples per bin side; 0 or less picks <c>ceil(roi size / outputSize)</c> per RoI.</param>
    /// <returns>Pooled features <c>[R, C, outputSize, outputSize]</c>.</returns>
    public static Tensor<T> RoIAlign(
        Tensor<T> features, Tensor<T> boxes, int[] batchIndices, double spatialScale, int outputSize, int samplingRatio)
    {
        int rois = batchIndices.Length;
        if (rois == 0)
        {
            return new Tensor<T>(new[] { 0, features.Shape[1], outputSize, outputSize });
        }

        // The engine's rows are (batch index, x1, y1, x2, y2): the first 4 * R box values, row-major.
        var flatBoxes = Engine.Reshape(boxes, new[] { boxes.Length });
        var boxRows = Engine.Reshape(Engine.TensorSlice(flatBoxes, new[] { 0 }, new[] { 4 * rois }), new[] { rois, 4 });
        int maxIndex = 0;
        foreach (int index in batchIndices)
        {
            maxIndex = Math.Max(maxIndex, index);
        }

        var indexColumn = Engine.Reshape(IndicesAsValues(batchIndices, maxIndex + 1), new[] { rois, 1 });
        var boxTensor = Engine.Concat(new[] { indexColumn, boxRows }, 1);
        return Engine.RoIAlign(features, boxTensor, outputSize, outputSize, (float)spatialScale, samplingRatio, aligned: false);
    }

    /// <summary>
    /// Gathers slices of <paramref name="x"/> along <paramref name="axis"/> at the given indices.
    /// </summary>
    public static Tensor<T> Select(Tensor<T> x, int[] indices, int axis)
        => Engine.TensorGather(x, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis);

    /// <summary>Indices <c>start .. end-1</c> clamped into <c>[0, extent-1]</c>.</summary>
    private static int[] ClampedRange(int start, int end, int extent)
    {
        var result = new int[end - start];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = Math.Min(Math.Max(start + i, 0), extent - 1);
        }

        return result;
    }

    private static (int[] Lo, int[] Hi, Tensor<T> WeightLo, Tensor<T> WeightHi) BilinearTaps(int src, int dst)
    {
        var lo = new int[dst];
        var hi = new int[dst];
        for (int i = 0; i < dst; i++)
        {
            double s = (double)i / dst * src;
            int i0 = (int)Math.Floor(s);
            lo[i] = i0;
            hi[i] = Math.Min(i0 + 1, src - 1);
        }

        // frac = i / dst * src - lo, evaluated for every output position at once. Subtracting the
        // integer tap (rather than flooring again) keeps each weight pair consistent with its taps.
        var position = Engine.TensorMultiplyScalar(
            Engine.TensorDivideScalar(Engine.TensorArange<T>(0, dst, 1), NumOps.FromDouble(dst)), NumOps.FromDouble(src));
        var wHi = Engine.TensorSubtract(position, IndicesAsValues(lo, Math.Max(src, 1)));
        var wLo = Engine.ScalarMinusTensor(NumOps.One, wHi);
        return (lo, hi, wLo, wHi);
    }

    /// <summary>
    /// Expands a per-position weight vector along <paramref name="axis"/> to the full target shape.
    /// </summary>
    private static Tensor<T> Broadcast(Tensor<T> weights, int[] dims, int axis)
    {
        var viewShape = new int[dims.Length];
        for (int d = 0; d < dims.Length; d++)
        {
            viewShape[d] = d == axis ? weights.Length : 1;
        }

        return Engine.TensorBroadcastTo(Engine.Reshape(weights, viewShape), dims);
    }

    /// <summary>
    /// The integers in <paramref name="indices"/> as values of type <typeparamref name="T"/>: a gather
    /// from the ramp <c>0 .. count-1</c>, so every value is exact. Each index must lie in <c>[0, count)</c>.
    /// </summary>
    public static Tensor<T> IndicesAsValues(int[] indices, int count)
        => IndicesAsValues(new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), count);

    /// <inheritdoc cref="IndicesAsValues(int[], int)"/>
    public static Tensor<T> IndicesAsValues(Tensor<int> indices, int count)
        => Engine.TensorGather(Engine.TensorArange<T>(0, Math.Max(count, 1), 1), indices, 0);

    /// <summary>A new tensor of the given shape with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> Fill(int[] shape, T value)
    {
        var result = new Tensor<T>(shape);
        Engine.TensorFill(result, value);
        return result;
    }

    /// <summary>A new tensor of the given shape with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> Fill(int[] shape, double value) => Fill(shape, NumOps.FromDouble(value));

    /// <summary>
    /// 1 where <paramref name="x"/> &gt;= <paramref name="threshold"/>, else 0 (NaN compares false).
    /// </summary>
    public static Tensor<T> GreaterOrEqual(Tensor<T> x, T threshold)
        => Engine.TensorGreaterEqual(x, Fill(Dims(x), threshold));

    /// <summary>
    /// Copies <paramref name="input"/> into a new ONNX Runtime <c>float</c> tensor of the same shape, converting
    /// the whole buffer in one bulk call.
    /// </summary>
    public static Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float> ToOnnxFloat(Tensor<T> input)
    {
        var data = new float[input.Length];
        NumOps.ToFloatSpan(input.Data.Span, data);
        return new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(data, input._shape);
    }

    /// <summary>
    /// Copies an ONNX Runtime <c>float</c> output into a new tensor of <typeparamref name="T"/> with the same
    /// dimensions, converting the whole buffer in one bulk call.
    /// </summary>
    public static Tensor<T> FromOnnxFloat(Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> output)
    {
        var dense = output.ToDenseTensor();
        var data = new T[dense.Length];
        NumOps.FromFloatSpan(dense.Buffer.Span, data);
        return new Tensor<T>(dense.Dimensions.ToArray(), new Vector<T>(data));
    }

    /// <summary>
    /// The items ordered by descending <paramref name="key"/> (compared as <see cref="double"/>), ties in input
    /// order - what <c>OrderByDescending(i =&gt; ToDouble(key(i)))</c> gives, with the keys converted in bulk.
    /// </summary>
    public static List<TItem> OrderByDescending<TItem>(IReadOnlyList<TItem> items, Func<TItem, T> key)
        => OrderBy(items, key, descending: true);

    /// <summary>
    /// The items ordered by <paramref name="key"/> (compared as <see cref="double"/>), ties in input order.
    /// </summary>
    public static List<TItem> OrderBy<TItem>(IReadOnlyList<TItem> items, Func<TItem, T> key, bool descending = false)
    {
        if (items.Count == 0) return new List<TItem>();
        var keys = ToDoubleArray(new Tensor<T>(new[] { items.Count }, new Vector<T>(items.Select(key).ToArray())));
        var order = Enumerable.Range(0, items.Count);
        var sorted = descending ? order.OrderByDescending(i => keys[i]) : order.OrderBy(i => keys[i]);
        return sorted.Select(i => items[i]).ToList();
    }

    /// <summary>
    /// Greedy NMS over XYXY <paramref name="boxes"/> <c>[N, 4]</c> already in priority order: each box in turn is
    /// kept unless an earlier kept box overlaps it with IoU strictly above <paramref name="iouThreshold"/>, until
    /// <paramref name="maxKeep"/> are kept. Returns the kept rows in order. The pairwise IoU is one engine op.
    /// </summary>
    public static List<int> GreedyNms(Tensor<T> boxes, double iouThreshold, int maxKeep)
    {
        int n = boxes.Shape[0];
        var selected = new List<int>();
        if (n == 0) return selected;

        var iou = ToDoubleArray(Engine.BoxIou(boxes, boxes));
        var used = new bool[n];
        for (int i = 0; i < n && selected.Count < maxKeep; i++)
        {
            if (used[i]) continue;
            selected.Add(i);
            used[i] = true;
            for (int j = i + 1; j < n; j++)
            {
                if (!used[j] && iou[i * n + j] > iouThreshold) used[j] = true;
            }
        }

        return selected;
    }

    /// <summary>Stacks equal-length <c>[N]</c> columns into an <c>[N, columns.Length]</c> tensor.</summary>
    public static Tensor<T> StackColumns(params Tensor<T>[] columns)
    {
        var reshaped = new Tensor<T>[columns.Length];
        for (int k = 0; k < columns.Length; k++)
            reshaped[k] = Engine.Reshape(columns[k], new[] { columns[k].Length, 1 });
        return Engine.Concat(reshaped, 1);
    }

    /// <summary>A rank-2 <c>[rows.Count, columns]</c> tensor from per-row value arrays (no arithmetic).</summary>
    public static Tensor<T> FromRows(IReadOnlyList<T[]> rows, int columns)
        => new Tensor<T>(new[] { rows.Count, columns }, new Vector<T>(rows.SelectMany(r => r).ToArray()));

    /// <summary>
    /// The elements of <paramref name="values"/> as <see cref="double"/>, for result types whose public
    /// contract is a <c>double[]</c> (detection and segmentation scores). A bulk conversion of the whole buffer.
    /// </summary>
    public static double[] ToDoubleArray(Tensor<T> values) => Array.ConvertAll(values.ToArray(), NumOps.ToDouble);

    /// <inheritdoc cref="ToDoubleArray(Tensor{T})"/>
    public static double[] ToDoubleArray(T[] values) => Array.ConvertAll(values, NumOps.ToDouble);

    /// <summary>
    /// A tensor of <typeparamref name="T"/> from a <c>double[]</c> supplied through a public API or a literal
    /// coefficient table, converted in bulk.
    /// </summary>
    public static Tensor<T> FromDoubleArray(double[] values, int[] shape)
        => new Tensor<T>(shape, new Vector<T>(Array.ConvertAll(values, NumOps.FromDouble)));

    /// <summary>
    /// Overwrites every element of <paramref name="destination"/> (in place, so live parameter tensors keep their
    /// identity) with <paramref name="values"/>, converted in bulk.
    /// </summary>
    public static void AssignFromDoubles(Tensor<T> destination, double[] values)
    {
        if (values.Length != destination.Length)
            throw new ArgumentException($"Expected {destination.Length} values but received {values.Length}.", nameof(values));
        Array.ConvertAll(values, NumOps.FromDouble).AsSpan().CopyTo(destination.AsWritableSpan());
    }

    /// <summary>Writes every element of <paramref name="tensor"/> as a <see cref="double"/>, in storage order.</summary>
    public static void WriteDoubles(System.IO.BinaryWriter writer, Tensor<T> tensor)
    {
        foreach (double value in ToDoubleArray(tensor))
            writer.Write(value);
    }

    /// <summary>Reads <c>tensor.Length</c> doubles and writes them through into <paramref name="tensor"/>.</summary>
    public static void ReadDoublesInto(System.IO.BinaryReader reader, Tensor<T> tensor)
    {
        var wire = new double[tensor.Length];
        for (int i = 0; i < wire.Length; i++)
            wire[i] = reader.ReadDouble();
        AssignFromDoubles(tensor, wire);
    }

    private static int[] Dims(Tensor<T> t)
    {
        var dims = new int[t.Shape.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = t.Shape[i];
        }

        return dims;
    }
}
