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
    {
        int pad = kernelSize / 2;
        int h = x.Shape[2];
        int w = x.Shape[3];
        var padded = Select(Select(x, ClampedRange(-pad, h + pad, h), 2), ClampedRange(-pad, w + pad, w), 3);
        return Engine.MaxPool2DWithIndices(padded, new[] { kernelSize, kernelSize }, new[] { 1, 1 }, out _);
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
    /// <returns>The attended values <c>[N, Lq, D]</c>.</returns>
    public static Tensor<T> MultiHeadAttention(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, int numHeads, double scale, bool causal = false)
    {
        int n = query.Shape[0], lq = query.Shape[1], d = query.Shape[2], lk = key.Shape[1];
        int headDim = d / numHeads;

        var q = SplitHeads(query, numHeads, headDim);
        var k = SplitHeads(key, numHeads, headDim);
        var v = SplitHeads(value, numHeads, headDim);

        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(q, Engine.TensorPermute(k, new[] { 0, 1, 3, 2 })), NumOps.FromDouble(scale));

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
    /// Flattens each per-image output to <c>[N, -1]</c> and concatenates them along axis 1, giving
    /// one <c>[N, total]</c> tensor that carries every head's raw output. A single output is
    /// returned unchanged.
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

        var flat = new Tensor<T>[outputs.Count];
        for (int i = 0; i < outputs.Count; i++)
        {
            int batch = outputs[i].Shape[0];
            flat[i] = Engine.Reshape(outputs[i], new[] { batch, outputs[i].Length / batch });
        }

        return Engine.TensorConcatenate(flat, 1);
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
