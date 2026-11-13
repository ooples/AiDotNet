using AiDotNet.Audio;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Document;

/// <summary>
/// Whole-tensor helpers shared by the document models (input normalization, greedy decoding, score
/// maps), written with engine operations so callers do not loop over elements.
/// </summary>
internal static class DocumentTensorOps
{
    /// <summary>ImageNet channel means on the [0, 1] pixel scale.</summary>
    internal static readonly double[] ImageNetMean = { 0.485, 0.456, 0.406 };

    /// <summary>ImageNet channel standard deviations on the [0, 1] pixel scale.</summary>
    internal static readonly double[] ImageNetStd = { 0.229, 0.224, 0.225 };

    /// <summary>The 0.5 / 0.5 statistics of BEiT, DiT, LayoutLMv3, UDOP and Donut (maps [0, 1] to [-1, 1]).</summary>
    internal static readonly double[] HalfStatistics = { 0.5, 0.5, 0.5 };

    /// <summary>OpenAI CLIP channel means.</summary>
    internal static readonly double[] ClipMean = { 0.48145466, 0.4578275, 0.40821073 };

    /// <summary>OpenAI CLIP channel standard deviations.</summary>
    internal static readonly double[] ClipStd = { 0.26862954, 0.26130258, 0.27577711 };

    /// <summary>
    /// <c>(x - mean[c]) / std[c]</c> over a <c>[B, C, H, W]</c> image, where channels past the end of the
    /// tables use <paramref name="fallback"/> for both the mean and the standard deviation.
    /// </summary>
    internal static Tensor<T> NormalizeChannels<T>(
        IEngine engine, Tensor<T> image, double[] means, double[] stds, double fallback = 0.5)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int channels = image.Shape[1];
        var mean = Enumerable.Range(0, channels).Select(c => c < means.Length ? means[c] : fallback).ToArray();
        var inverse = Enumerable.Range(0, channels).Select(c => 1.0 / (c < stds.Length ? stds[c] : fallback)).ToArray();
        var shape = image.Shape.ToArray();
        var meanTensor = new Tensor<T>(new[] { 1, channels, 1, 1 }, new Vector<T>(mean.Select(numOps.FromDouble).ToArray()));
        var inverseTensor = new Tensor<T>(new[] { 1, channels, 1, 1 }, new Vector<T>(inverse.Select(numOps.FromDouble).ToArray()));
        return engine.TensorMultiply(
            engine.TensorSubtract(image, engine.TensorBroadcastTo(meanTensor, shape)),
            engine.TensorBroadcastTo(inverseTensor, shape));
    }

    /// <summary>
    /// The greedy token of every row of a <c>[rows, vocab]</c> score matrix: the first strict maximum of the
    /// row (later ties never replace it, NaN never wins, 0 when nothing beats -infinity).
    /// </summary>
    internal static int[] RowArgMax<T>(IEngine engine, Tensor<T> scores)
        => AudioEngineOps.RowFirstArgMax(engine, scores);

    /// <summary>
    /// The greedy token of each of the first <c>min(rows, maxLength)</c> rows of a decoder output whose first
    /// axis is the sequence (the remaining axes are the vocabulary).
    /// </summary>
    internal static int[] GreedyRowTokens<T>(IEngine engine, Tensor<T> output, int maxLength)
    {
        int rows = Math.Min(output.Shape[0], maxLength);
        if (rows <= 0 || output.Length == 0)
            return Array.Empty<int>();
        int vocab = output.Length / output.Shape[0];
        var matrix = engine.Reshape(output, new[] { output.Shape[0], vocab });
        var head = rows == output.Shape[0] ? matrix : engine.TensorSlice(matrix, new[] { 0, 0 }, new[] { rows, vocab });
        return RowArgMax(engine, head);
    }

    /// <summary>
    /// The rows <c>i &lt; min(rows, limit)</c> of an output viewed as <c>[rows, columns]</c> whose value in
    /// <paramref name="column"/> is at least <paramref name="threshold"/> (NaN never qualifies), in ascending
    /// row order, with those values.
    /// </summary>
    internal static (int[] Rows, T[] Scores) RowsAtLeast<T>(IEngine engine, Tensor<T> output, int limit, int column, T threshold)
    {
        var scores = ColumnHead(engine, output, limit, column);
        if (scores.Length == 0)
            return (Array.Empty<int>(), Array.Empty<T>());
        var hit = engine.TensorGreaterEqual(scores, Tensor<T>.CreateDefault(new[] { scores.Length }, threshold));
        int[] rows = engine.TensorNonzero(hit).ToArray();
        if (rows.Length == 0)
            return (rows, Array.Empty<T>());
        var picked = engine.TensorIndexSelect(scores, new Tensor<int>((int[])rows.Clone(), new[] { rows.Length }), 0);
        return (rows, picked.ToArray());
    }

    /// <summary>
    /// Column <paramref name="column"/> of the first <c>min(rows, limit)</c> rows of an output viewed as
    /// <c>[rows, columns]</c> (its first axis is the row axis).
    /// </summary>
    internal static Tensor<T> ColumnHead<T>(IEngine engine, Tensor<T> output, int limit, int column)
    {
        int rows = output.Rank == 0 ? 0 : Math.Min(output.Shape[0], limit);
        if (rows <= 0 || output.Length == 0)
            return new Tensor<T>(new[] { 0 });
        int columns = output.Length / output.Shape[0];
        if (column < 0 || column >= columns)
            throw new IndexOutOfRangeException($"Column {column} is outside the output's {columns} columns.");
        var matrix = engine.Reshape(output, new[] { output.Shape[0], columns });
        return engine.Reshape(engine.TensorSlice(matrix, new[] { 0, column }, new[] { rows, 1 }), new[] { rows });
    }
}
