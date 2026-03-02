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
        VocabularySize = vocabularySize;
    }

    /// <summary>
    /// Tokenizes a time series into patches.
    /// </summary>
    /// <param name="series">Input time series of shape [sequence_length] or [batch, sequence_length].</param>
    /// <returns>
    /// Patched tensor of shape [num_patches, patch_length] or [batch, num_patches, patch_length].
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

        int seqLen = series.Rank == 1 ? series.Length : series.Shape[series.Rank - 1];
        var patches = new List<Tensor<T>>();

        for (int start = 0; start + PatchLength <= seqLen; start += Stride)
        {
            var patch = new Tensor<T>(new[] { PatchLength });
            for (int j = 0; j < PatchLength; j++)
            {
                int idx = start + j;
                if (idx < series.Length)
                    patch.Data.Span[j] = series[idx];
            }
            patches.Add(patch);
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

        int totalLen = 0;
        if (Stride == PatchLength)
        {
            totalLen = patches.Count * PatchLength;
        }
        else
        {
            totalLen = (patches.Count - 1) * Stride + PatchLength;
        }

        var result = new Tensor<T>(new[] { totalLen });
        var counts = new int[totalLen];

        for (int p = 0; p < patches.Count; p++)
        {
            int start = p * Stride;
            for (int j = 0; j < PatchLength && (start + j) < totalLen; j++)
            {
                int idx = start + j;
                result.Data.Span[idx] = NumOps.Add(result[idx], patches[p][j]);
                counts[idx]++;
            }
        }

        // Average overlapping regions
        for (int i = 0; i < totalLen; i++)
        {
            if (counts[i] > 1)
                result.Data.Span[i] = NumOps.Divide(result[i], NumOps.FromDouble(counts[i]));
        }

        return result;
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
        T mean = NumOps.Zero;
        for (int i = 0; i < len; i++)
            mean = NumOps.Add(mean, series[i]);
        mean = NumOps.Divide(mean, NumOps.FromDouble(len));

        T variance = NumOps.Zero;
        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(series[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(len));
        T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));

        var normalized = new Tensor<T>(series.Shape);
        for (int i = 0; i < len; i++)
            normalized.Data.Span[i] = NumOps.Divide(NumOps.Subtract(series[i], mean), std);

        return (normalized, mean, std);
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

        var result = new Tensor<T>(prediction.Shape);
        for (int i = 0; i < prediction.Length; i++)
            result.Data.Span[i] = NumOps.Add(NumOps.Multiply(prediction[i], std), mean);

        return result;
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
    public (int[] Tokens, double[] BinEdges) Quantize(Tensor<T> series)
    {
        Guard.NotNull(series);

        int len = series.Length;
        double min = double.MaxValue, max = double.MinValue;

        for (int i = 0; i < len; i++)
        {
            double val = NumOps.ToDouble(series[i]);
            if (val < min) min = val;
            if (val > max) max = val;
        }

        double range = max - min;
        if (range < 1e-10) range = 1.0;

        var binEdges = new double[VocabularySize + 1];
        for (int i = 0; i <= VocabularySize; i++)
            binEdges[i] = min + (range * i / VocabularySize);

        var tokens = new int[len];
        for (int i = 0; i < len; i++)
        {
            double val = NumOps.ToDouble(series[i]);
            int bin = (int)((val - min) / range * (VocabularySize - 1));
            tokens[i] = Math.Max(0, Math.Min(VocabularySize - 1, bin));
        }

        return (tokens, binEdges);
    }

    /// <summary>
    /// Dequantizes token indices back to continuous values using bin centers.
    /// </summary>
    /// <param name="tokens">Token indices from Quantize.</param>
    /// <param name="binEdges">Bin edges from Quantize.</param>
    /// <returns>Continuous tensor with dequantized values.</returns>
    public Tensor<T> Dequantize(int[] tokens, double[] binEdges)
    {
        Guard.NotNull(tokens);
        Guard.NotNull(binEdges);

        var result = new Tensor<T>(new[] { tokens.Length });
        for (int i = 0; i < tokens.Length; i++)
        {
            int bin = Math.Max(0, Math.Min(tokens[i], binEdges.Length - 2));
            double center = (binEdges[bin] + binEdges[bin + 1]) / 2.0;
            result.Data.Span[i] = NumOps.FromDouble(center);
        }

        return result;
    }
}
