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

        var rows = new int[targetH];
        for (int h = 0; h < targetH; h++)
        {
            rows[h] = Math.Min((int)((long)h * srcH / targetH), srcH - 1);
        }

        var cols = new int[targetW];
        for (int w = 0; w < targetW; w++)
        {
            cols[w] = Math.Min((int)((long)w * srcW / targetW), srcW - 1);
        }

        return Select(Select(x, rows, 2), cols, 3);
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
    /// <remarks>
    /// Out-of-bounds positions are filled by clamping their index to the nearest edge. The clamped
    /// cell always lies inside the same window, so it duplicates an in-bounds candidate and cannot
    /// change the maximum.
    /// </remarks>
    public static Tensor<T> MaxPoolSame(Tensor<T> x, int kernelSize)
        => MaxPoolPadded(x, kernelSize, 1, kernelSize / 2);

    /// <summary>
    /// Max pooling with a square window, a stride and symmetric padding, where padded positions are
    /// ignored rather than treated as zero - PyTorch's <c>nn.MaxPool2d(kernel, stride, padding)</c>
    /// in floor mode.
    /// </summary>
    /// <remarks>
    /// Out-of-bounds positions are filled by clamping their index to the nearest edge. While
    /// <paramref name="padding"/> is smaller than <paramref name="kernelSize"/> (which PyTorch itself
    /// requires), every window reaches at least one in-bounds cell on the side it overhangs, and the
    /// clamped cell IS that edge cell - a duplicate candidate that cannot change the maximum.
    /// </remarks>
    public static Tensor<T> MaxPoolPadded(Tensor<T> x, int kernelSize, int stride, int padding)
    {
        if (padding < 0 || padding >= kernelSize)
        {
            throw new ArgumentOutOfRangeException(nameof(padding), padding,
                $"Padding must lie in [0, kernelSize) = [0, {kernelSize}).");
        }

        int h = x.Shape[2];
        int w = x.Shape[3];
        int outH = (h + 2 * padding - kernelSize) / stride + 1;
        int outW = (w + 2 * padding - kernelSize) / stride + 1;

        // Only the cells some window reads: from -padding to the last window's far edge.
        int endH = (outH - 1) * stride - padding + kernelSize;
        int endW = (outW - 1) * stride - padding + kernelSize;
        var padded = x;
        if (padding > 0 || endH != h)
        {
            padded = Select(padded, ClampedRange(-padding, endH, h), 2);
        }

        if (padding > 0 || endW != w)
        {
            padded = Select(padded, ClampedRange(-padding, endW, w), 3);
        }

        return Engine.MaxPool2DWithIndices(padded, new[] { kernelSize, kernelSize }, new[] { stride, stride }, out _);
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
            var mask = new Tensor<T>(new[] { 1, 1, lq, lk });
            var blocked = NumOps.FromDouble(-1e30);
            for (int i = 0; i < lq; i++)
            {
                for (int j = i + 1; j < lk; j++)
                {
                    mask[(i * lk) + j] = blocked;
                }
            }

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
    /// RoIAlign: pools each region of interest into an <c>outputSize x outputSize</c> grid by
    /// averaging <c>samplingRatio^2</c> bilinear samples per bin, over an NCHW feature map.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The boxes are treated as constants - as in standard RoIAlign, the gradient flows into the
    /// FEATURES, not the box coordinates - so the whole operation is a fixed sparse linear map of the
    /// feature map. It is built as one gather of the four bilinear corners of every sample, a
    /// multiply by the precomputed corner weights (each already divided by the bin's in-bounds sample
    /// count), and a sum. Samples that fall outside the map contribute nothing and are not counted; a
    /// bin with no in-bounds samples is zero.
    /// </para>
    /// </remarks>
    /// <param name="features">Feature map <c>[N, C, H, W]</c>.</param>
    /// <param name="boxes">Per-RoI <c>(x1, y1, x2, y2)</c> in image coordinates, length <c>4 * R</c>.</param>
    /// <param name="batchIndices">Per-RoI image index into <paramref name="features"/>, length <c>R</c>.</param>
    /// <param name="spatialScale">Image-to-feature-map scale.</param>
    /// <param name="outputSize">Pooled grid side.</param>
    /// <param name="samplingRatio">Samples per bin side.</param>
    /// <returns>Pooled features <c>[R, C, outputSize, outputSize]</c>.</returns>
    public static Tensor<T> RoIAlign(
        Tensor<T> features, double[] boxes, int[] batchIndices, double spatialScale, int outputSize, int samplingRatio)
    {
        int n = features.Shape[0], c = features.Shape[1], h = features.Shape[2], w = features.Shape[3];
        int rois = batchIndices.Length;
        int bins = rois * outputSize * outputSize;
        int taps = samplingRatio * samplingRatio * 4;

        var index = new int[bins * taps];
        var weight = new T[bins * taps];
        var zero = NumOps.Zero;
        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = zero;
        }

        for (int r = 0; r < rois; r++)
        {
            int b = batchIndices[r];
            double x1 = boxes[(4 * r) + 0] * spatialScale, y1 = boxes[(4 * r) + 1] * spatialScale;
            double x2 = boxes[(4 * r) + 2] * spatialScale, y2 = boxes[(4 * r) + 3] * spatialScale;
            double binW = (x2 - x1) / outputSize, binH = (y2 - y1) / outputSize;

            for (int ph = 0; ph < outputSize; ph++)
            {
                for (int pw = 0; pw < outputSize; pw++)
                {
                    int bin = ((r * outputSize) + ph) * outputSize + pw;
                    double startY = y1 + (ph * binH), startX = x1 + (pw * binW);

                    int count = 0;
                    for (int iy = 0; iy < samplingRatio; iy++)
                    {
                        for (int ix = 0; ix < samplingRatio; ix++)
                        {
                            double y = startY + ((iy + 0.5) * binH / samplingRatio);
                            double x = startX + ((ix + 0.5) * binW / samplingRatio);
                            if (y >= 0 && y < h && x >= 0 && x < w)
                            {
                                count++;
                            }
                        }
                    }

                    if (count == 0)
                    {
                        continue;
                    }

                    int tap = bin * taps;
                    for (int iy = 0; iy < samplingRatio; iy++)
                    {
                        for (int ix = 0; ix < samplingRatio; ix++)
                        {
                            double y = startY + ((iy + 0.5) * binH / samplingRatio);
                            double x = startX + ((ix + 0.5) * binW / samplingRatio);
                            if (!(y >= 0 && y < h && x >= 0 && x < w))
                            {
                                tap += 4;
                                continue;
                            }

                            int y0 = (int)Math.Floor(y), x0 = (int)Math.Floor(x);
                            int yy1 = Math.Min(y0 + 1, h - 1), xx1 = Math.Min(x0 + 1, w - 1);
                            double wy1 = y - y0, wy0 = 1.0 - wy1, wx1 = x - x0, wx0 = 1.0 - wx1;
                            int rowBase = b * h;

                            index[tap] = ((rowBase + y0) * w) + x0;
                            weight[tap++] = NumOps.FromDouble(wy0 * wx0 / count);
                            index[tap] = ((rowBase + y0) * w) + xx1;
                            weight[tap++] = NumOps.FromDouble(wy0 * wx1 / count);
                            index[tap] = ((rowBase + yy1) * w) + x0;
                            weight[tap++] = NumOps.FromDouble(wy1 * wx0 / count);
                            index[tap] = ((rowBase + yy1) * w) + xx1;
                            weight[tap++] = NumOps.FromDouble(wy1 * wx1 / count);
                        }
                    }
                }
            }
        }

        var positions = Engine.Reshape(Engine.TensorPermute(features, new[] { 0, 2, 3, 1 }), new[] { n * h * w, c });
        var gathered = Select(positions, index, 0);                                        // [bins*taps, C]
        var weights = Engine.TensorBroadcastTo(
            new Tensor<T>(new[] { bins * taps, 1 }, new Vector<T>(weight)), new[] { bins * taps, c });
        var weighted = Engine.Reshape(Engine.TensorMultiply(gathered, weights), new[] { bins, taps, c });
        var pooled = Engine.ReduceSum(weighted, new[] { 1 }, false);                       // [bins, C]
        return Engine.TensorPermute(
            Engine.Reshape(pooled, new[] { rois, outputSize, outputSize, c }), new[] { 0, 3, 1, 2 });
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

    private static (int[] Lo, int[] Hi, T[] WeightLo, T[] WeightHi) BilinearTaps(int src, int dst)
    {
        var lo = new int[dst];
        var hi = new int[dst];
        var wLo = new T[dst];
        var wHi = new T[dst];
        for (int i = 0; i < dst; i++)
        {
            double s = (double)i / dst * src;
            int i0 = (int)Math.Floor(s);
            lo[i] = i0;
            hi[i] = Math.Min(i0 + 1, src - 1);
            double frac = s - i0;
            wHi[i] = NumOps.FromDouble(frac);
            wLo[i] = NumOps.FromDouble(1.0 - frac);
        }

        return (lo, hi, wLo, wHi);
    }

    /// <summary>
    /// Expands a per-position weight vector along <paramref name="axis"/> to the full target shape.
    /// </summary>
    private static Tensor<T> Broadcast(T[] weights, int[] dims, int axis)
    {
        var viewShape = new int[dims.Length];
        for (int d = 0; d < dims.Length; d++)
        {
            viewShape[d] = d == axis ? weights.Length : 1;
        }

        var view = new Tensor<T>(viewShape, new Vector<T>(weights));
        return Engine.TensorBroadcastTo(view, dims);
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
