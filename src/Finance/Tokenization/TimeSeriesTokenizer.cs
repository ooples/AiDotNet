using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Finance.Tokenization;

/// <summary>
/// Standard tokenization pipeline for time series foundation models, supporting patching,
/// quantization, and instance normalization strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Before feeding raw time series data into a foundation model,
/// we need to "tokenize" it — convert it into a format the model expects. Different models
/// use different tokenization strategies:
/// <list type="bullet">
/// <item><b>Patching</b> (PatchTST, Chronos-2, Moirai): Splits the series into fixed-size chunks</item>
/// <item><b>Quantization</b> (Chronos v1): Maps continuous values to discrete vocabulary tokens</item>
/// <item><b>Instance Normalization</b> (RevIN): Normalizes each series independently</item>
/// <item><b>Adaptive</b> (Kairos): Variable-size patches based on local information density</item>
/// </list>
/// </para>
/// <para>
/// <b>Reference:</b> Standard tokenization approaches from Chronos (ICML 2024),
/// PatchTST (ICLR 2023), and Kairos (2025).
/// </para>
/// </remarks>
public class TimeSeriesTokenizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the tokenization strategy being used.
    /// </summary>
    public TimeSeriesTokenizationStrategy Strategy { get; }

    /// <summary>
    /// Gets the patch length for patch-based tokenization.
    /// </summary>
    public int PatchLength { get; }

    /// <summary>
    /// Gets the stride between patches.
    /// </summary>
    public int Stride { get; }

    /// <summary>
    /// Gets the vocabulary size for quantization-based tokenization.
    /// </summary>
    public int VocabularySize { get; }

    /// <summary>
    /// Creates a new tokenizer with the specified strategy.
    /// </summary>
    /// <param name="strategy">The tokenization strategy to use.</param>
    /// <param name="patchLength">Patch length for patching strategies (default: 16).</param>
    /// <param name="stride">Stride between patches (default: same as patchLength for non-overlapping).</param>
    /// <param name="vocabularySize">Vocabulary size for quantization strategies (default: 4096).</param>
    public TimeSeriesTokenizer(
        TimeSeriesTokenizationStrategy strategy = TimeSeriesTokenizationStrategy.Patching,
        int patchLength = 16,
        int? stride = null,
        int vocabularySize = 4096)
    {
        Guard.Positive(patchLength);
        Guard.Positive(vocabularySize);

        Strategy = strategy;
        PatchLength = patchLength;
        Stride = stride ?? patchLength;
        Guard.Positive(Stride, nameof(Stride));
        VocabularySize = vocabularySize;
    }

    /// <summary>
    /// Tokenizes a time series into patches.
    /// </summary>
    /// <param name="series">Input time series of shape [sequence_length] (rank-1 only).</param>
    /// <returns>
    /// List of patch tensors, each of shape [patch_length].
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> Patching splits a long series into fixed-size chunks:
    /// <code>
    /// Input:  [1, 2, 3, 4, 5, 6, 7, 8]  (length 8)
    /// Patches (size=4, stride=4): [[1,2,3,4], [5,6,7,8]]  (2 patches)
    /// Patches (size=4, stride=2): [[1,2,3,4], [3,4,5,6], [5,6,7,8]]  (3 overlapping patches)
    /// </code>
    /// </remarks>
    public List<Tensor<T>> CreatePatches(Tensor<T> series)
    {
        Guard.NotNull(series);

        if (series.Rank > 1)
            throw new ArgumentException(
                $"CreatePatches expects a rank-1 (univariate) time series but received rank {series.Rank}. " +
                "Flatten or slice the input to a single sequence before patching.",
                nameof(series));

        int seqLen = series.Length;
        var patches = new List<Tensor<T>>();

        var flat = AiDotNetEngine.Current.Reshape(series, new[] { seqLen });
        for (int start = 0; start + PatchLength <= seqLen; start += Stride)
        {
            patches.Add(AiDotNetEngine.Current.TensorSlice(flat, new[] { start }, new[] { PatchLength }));
        }

        return patches;
    }

    /// <summary>
    /// Reconstructs a time series from patches (inverse of CreatePatches for non-overlapping case).
    /// </summary>
    /// <param name="patches">List of patch tensors each of shape [patch_length].</param>
    /// <returns>Reconstructed time series tensor.</returns>
    public Tensor<T> ReconstructFromPatches(List<Tensor<T>> patches)
    {
        Guard.NotNull(patches);

        if (patches.Count == 0)
            return new Tensor<T>(new[] { 0 });

        // Validate each patch has the expected length
        for (int i = 0; i < patches.Count; i++)
        {
            if (patches[i] == null)
                throw new ArgumentException($"Patch at index {i} is null.", nameof(patches));
            if (patches[i].Length != PatchLength)
                throw new ArgumentException(
                    $"Patch at index {i} has length {patches[i].Length} but expected {PatchLength}.",
                    nameof(patches));
        }

        int totalLen = 0;
        if (Stride == PatchLength)
        {
            totalLen = patches.Count * PatchLength;
        }
        else
        {
            totalLen = (patches.Count - 1) * Stride + PatchLength;
        }

        // Overlap-add every patch at p * Stride, then average positions covered more than once.
        var engine = AiDotNetEngine.Current;
        int stride = Stride;
        int patchLength = PatchLength;
        var positions = Enumerable.Range(0, patches.Count * patchLength)
            .Select(k => ((k / patchLength) * stride) + (k % patchLength))
            .ToArray();
        var target = new Tensor<int>(positions, new[] { positions.Length });
        var values = engine.TensorConcatenate(
            patches.Select(patch => engine.Reshape(patch, new[] { patchLength })).ToArray(), 0);
        var ones = new Tensor<T>(new[] { positions.Length });
        engine.TensorFill(ones, NumOps.One);

        var sums = engine.ScatterAdd(new Tensor<T>(new[] { totalLen }), target, values, 0);
        var counts = engine.ScatterAdd(new Tensor<T>(new[] { totalLen }), target, ones, 0);
        return engine.TensorDivide(sums, engine.TensorClampMin(counts, NumOps.One));
    }

    /// <summary>
    /// Applies instance normalization (RevIN) to a time series.
    /// </summary>
    /// <param name="series">Input time series of shape [sequence_length].</param>
    /// <returns>Tuple of (normalized series, mean, std) for later denormalization.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> RevIN (Reversible Instance Normalization) normalizes each time series
    /// independently to zero mean and unit variance. This helps models handle non-stationary data
    /// (where the mean/variance changes over time). After prediction, you can reverse the
    /// normalization using the returned mean and std.
    /// </remarks>
    public (Tensor<T> Normalized, T Mean, T Std) ApplyInstanceNorm(Tensor<T> series)
    {
        Guard.NotNull(series);

        int len = series.Length;
        if (len == 0)
            return (new Tensor<T>(series._shape), NumOps.Zero, NumOps.One);

        // RevIN: population mean and variance over the instance, std = sqrt(var + 1e-5).
        var engine = AiDotNetEngine.Current;
        T mean = engine.TensorMean(series);
        var centred = engine.TensorSubtractScalar(series, mean);
        T variance = NumOps.Divide(engine.TensorSumOfSquares(centred), NumOps.FromDouble(len));
        T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));

        return (engine.TensorDivideScalar(centred, std), mean, std);
    }

    /// <summary>
    /// Reverses instance normalization on a prediction.
    /// </summary>
    /// <param name="prediction">Normalized prediction tensor.</param>
    /// <param name="mean">Original series mean from ApplyInstanceNorm.</param>
    /// <param name="std">Original series std from ApplyInstanceNorm.</param>
    /// <returns>Denormalized prediction tensor in the original scale.</returns>
    public Tensor<T> ReverseInstanceNorm(Tensor<T> prediction, T mean, T std)
    {
        Guard.NotNull(prediction);

        var engine = AiDotNetEngine.Current;
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(prediction, std), mean);
    }

    /// <summary>
    /// Quantizes continuous values into discrete tokens using uniform binning.
    /// </summary>
    /// <param name="series">Input time series of shape [sequence_length].</param>
    /// <returns>Tuple of (token indices, bin edges) for later dequantization.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Quantization converts continuous values (like 23.456) into
    /// discrete vocabulary tokens (like token #1234). This is how Chronos v1 works — it
    /// treats time series forecasting as a language modeling problem.
    /// </remarks>
    public (int[] Tokens, Tensor<T> BinEdges) Quantize(Tensor<T> series)
    {
        Guard.NotNull(series);

        int len = series.Length;
        if (len == 0)
            return (Array.Empty<int>(), new Tensor<T>(new[] { NumOps.Zero, NumOps.One }, new[] { 2 }));

        var engine = AiDotNetEngine.Current;
        T min = engine.TensorMinValue(series);
        T range = NumOps.Subtract(engine.TensorMaxValue(series), min);
        if (NumOps.LessThan(range, NumOps.FromDouble(1e-10))) range = NumOps.One;

        // Edges min + range * i / V for i = 0..V.
        T vocabulary = NumOps.FromDouble(VocabularySize);
        var binEdges = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<T>(0, VocabularySize + 1), NumOps.Divide(range, vocabulary)), min);

        // Map value to bin index [0, VocabularySize-1] using consistent binEdges spacing
        var scaled = engine.TensorMultiplyScalar(
            engine.TensorDivideScalar(engine.TensorSubtractScalar(series, min), range), vocabulary);
        var bins = engine.TensorClamp(engine.TensorFloor(scaled), NumOps.Zero, NumOps.FromDouble(VocabularySize - 1));
        var tokens = bins.Cast<int>().ToArray();

        return (tokens, binEdges);
    }

    /// <summary>
    /// Dequantizes token indices back to continuous values using bin centers.
    /// </summary>
    /// <param name="tokens">Token indices from Quantize.</param>
    /// <param name="binEdges">Bin edges from Quantize.</param>
    /// <returns>Continuous tensor with dequantized values.</returns>
    public Tensor<T> Dequantize(int[] tokens, Tensor<T> binEdges)
    {
        Guard.NotNull(tokens);
        Guard.NotNull(binEdges);

        if (binEdges.Length < 2)
            throw new ArgumentException("binEdges must have at least 2 elements (one bin requires a lower and upper edge).", nameof(binEdges));

        // Bin centres (edge_b + edge_(b+1)) / 2, gathered at each token clamped to [0, bins - 1].
        var engine = AiDotNetEngine.Current;
        int binCount = binEdges.Length - 1;
        var edges = engine.Reshape(binEdges, new[] { binEdges.Length });
        var centers = engine.TensorMultiplyScalar(
            engine.TensorAdd(
                engine.TensorSlice(edges, new[] { 0 }, new[] { binCount }),
                engine.TensorSlice(edges, new[] { 1 }, new[] { binCount })),
            NumOps.FromDouble(0.5));
        var bins = tokens.Select(token => Math.Max(0, Math.Min(token, binCount - 1))).ToArray();
        return engine.TensorTake(centers, new Tensor<int>(bins, new[] { bins.Length }));
    }
}
