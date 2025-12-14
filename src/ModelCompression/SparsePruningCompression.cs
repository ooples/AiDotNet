using System;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Implements sparse pruning compression by zeroing out small-magnitude weights.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Sparse pruning removes weights below a certain threshold, setting them to zero.
/// This creates sparsity in the model which can be exploited for efficient storage
/// using sparse matrix formats (only non-zero values and their indices are stored).
/// </para>
/// <para><b>For Beginners:</b> Sparse pruning is like cleaning out your closet.
///
/// The idea:
/// - Many neural network weights are very small (close to zero)
/// - These tiny weights contribute little to the output
/// - We can set them to exactly zero without much accuracy loss
/// - Only store the non-zero weights and their positions
///
/// How it works:
/// 1. Calculate a threshold (e.g., smallest 90% of weights by magnitude)
/// 2. Set all weights below threshold to zero
/// 3. Store only non-zero values with their indices
///
/// Benefits:
/// - Can achieve 90%+ sparsity (90% zeros) with minimal accuracy loss
/// - Sparse storage is very efficient (only store ~10% of weights)
/// - Works well combined with quantization or clustering
///
/// Example:
/// - Original: [0.001, 0.5, -0.002, 0.8, 0.003, -0.7]
/// - After 50% pruning: [0, 0.5, 0, 0.8, 0, -0.7]
/// - Sparse storage: values=[0.5, 0.8, -0.7], indices=[1, 3, 5]
/// </para>
/// </remarks>
public class SparsePruningCompression<T> : ModelCompressionBase<T>
{
    private readonly double _sparsityTarget;
    private readonly double _minMagnitudeThreshold;
    private readonly bool _useGlobalThreshold;

    /// <summary>
    /// Initializes a new instance of the SparsePruningCompression class.
    /// </summary>
    /// <param name="sparsityTarget">Target sparsity level (0.0 to 1.0, default: 0.9 = 90% zeros).</param>
    /// <param name="minMagnitudeThreshold">Minimum magnitude threshold (default: 0 = use sparsity target).</param>
    /// <param name="useGlobalThreshold">Whether to use a global threshold or per-layer (default: true).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> These parameters control pruning aggressiveness:
    ///
    /// - sparsityTarget: What fraction of weights to set to zero
    ///   * 0.5 = 50% zeros (mild pruning)
    ///   * 0.9 = 90% zeros (aggressive pruning, common choice)
    ///   * 0.95 = 95% zeros (very aggressive)
    ///
    /// - minMagnitudeThreshold: Absolute threshold (overrides sparsity target if > 0)
    ///   * 0.01 means all weights with |w| &lt; 0.01 become zero
    ///   * Useful when you know a good threshold for your model
    ///
    /// - useGlobalThreshold: Apply same threshold to all weights
    ///   * true = find one threshold for entire model
    ///   * false = find separate threshold for each layer (preserves layer balance)
    /// </para>
    /// </remarks>
    public SparsePruningCompression(
        double sparsityTarget = 0.9,
        double minMagnitudeThreshold = 0,
        bool useGlobalThreshold = true)
    {
        if (sparsityTarget < 0 || sparsityTarget > 1)
        {
            throw new ArgumentException("Sparsity target must be between 0 and 1.", nameof(sparsityTarget));
        }

        if (minMagnitudeThreshold < 0)
        {
            throw new ArgumentException("Minimum magnitude threshold cannot be negative.", nameof(minMagnitudeThreshold));
        }

        _sparsityTarget = sparsityTarget;
        _minMagnitudeThreshold = minMagnitudeThreshold;
        _useGlobalThreshold = useGlobalThreshold;
    }

    /// <summary>
    /// Compresses weights by pruning small-magnitude values and storing in sparse format.
    /// </summary>
    /// <param name="weights">The original model weights.</param>
    /// <returns>Compressed sparse representation and metadata.</returns>
    public override (Vector<T> compressedWeights, object metadata) Compress(Vector<T> weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        if (weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be empty.", nameof(weights));
        }

        // Calculate threshold
        double threshold = CalculateThreshold(weights);

        // Apply pruning and collect non-zero values with indices
        var nonZeroValues = new List<T>();
        var nonZeroIndices = new List<int>();

        for (int i = 0; i < weights.Length; i++)
        {
            double magnitude = Math.Abs(NumOps.ToDouble(weights[i]));
            if (magnitude >= threshold)
            {
                nonZeroValues.Add(weights[i]);
                nonZeroIndices.Add(i);
            }
        }

        // Create metadata
        var metadata = new SparsePruningMetadata<T>(
            nonZeroIndices: nonZeroIndices.ToArray(),
            originalLength: weights.Length,
            threshold: threshold,
            actualSparsity: 1.0 - ((double)nonZeroValues.Count / weights.Length));

        // Return non-zero values as compressed representation
        return (new Vector<T>(nonZeroValues.ToArray()), metadata);
    }

    /// <summary>
    /// Decompresses sparse weights back to dense format.
    /// </summary>
    /// <param name="compressedWeights">The non-zero weight values.</param>
    /// <param name="metadata">The metadata containing indices and original length.</param>
    /// <returns>The reconstructed dense weights (with zeros filled in).</returns>
    public override Vector<T> Decompress(Vector<T> compressedWeights, object metadata)
    {
        if (compressedWeights == null)
        {
            throw new ArgumentNullException(nameof(compressedWeights));
        }

        var sparseMetadata = metadata as SparsePruningMetadata<T>;
        if (sparseMetadata == null)
        {
            throw new ArgumentException("Invalid metadata type for sparse pruning.", nameof(metadata));
        }

        // Reconstruct dense array
        var denseArray = new T[sparseMetadata.OriginalLength];

        // Initialize with zeros
        for (int i = 0; i < denseArray.Length; i++)
        {
            denseArray[i] = NumOps.Zero;
        }

        // Fill in non-zero values
        for (int i = 0; i < sparseMetadata.NonZeroIndices.Length && i < compressedWeights.Length; i++)
        {
            int idx = sparseMetadata.NonZeroIndices[i];
            if (idx >= 0 && idx < denseArray.Length)
            {
                denseArray[idx] = compressedWeights[i];
            }
        }

        return new Vector<T>(denseArray);
    }

    /// <summary>
    /// Gets the compressed size including sparse values and indices.
    /// </summary>
    public override long GetCompressedSize(Vector<T> compressedWeights, object metadata)
    {
        var sparseMetadata = metadata as SparsePruningMetadata<T>;
        if (sparseMetadata == null)
        {
            throw new ArgumentException("Invalid metadata type.", nameof(metadata));
        }

        // Size of non-zero values
        long valuesSize = compressedWeights.Length * GetElementSize();

        // Size of indices (32-bit integers)
        long indicesSize = sparseMetadata.NonZeroIndices.Length * sizeof(int);

        // Metadata overhead
        long metadataSize = sizeof(int) + sizeof(double) * 2; // originalLength, threshold, actualSparsity

        return valuesSize + indicesSize + metadataSize;
    }

    /// <summary>
    /// Calculates the pruning threshold based on configuration.
    /// </summary>
    private double CalculateThreshold(Vector<T> weights)
    {
        // If explicit threshold is set, use it
        if (_minMagnitudeThreshold > 0)
        {
            return _minMagnitudeThreshold;
        }

        // Calculate threshold based on sparsity target
        // Sort magnitudes and find the value at the sparsity percentile
        var magnitudes = new double[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            magnitudes[i] = Math.Abs(NumOps.ToDouble(weights[i]));
        }

        Array.Sort(magnitudes);

        // Find the threshold at the sparsity target percentile
        int thresholdIndex = (int)(weights.Length * _sparsityTarget);
        thresholdIndex = Math.Max(0, Math.Min(thresholdIndex, weights.Length - 1));

        return magnitudes[thresholdIndex];
    }
}

/// <summary>
/// Metadata for sparse pruning compression.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This metadata stores:
/// - Indices of non-zero values (so we know where to put them during decompression)
/// - Original vector length (to reconstruct the right size)
/// - Threshold used (for reference)
/// - Actual sparsity achieved (for statistics)
/// </para>
/// </remarks>
public class SparsePruningMetadata<T> : ICompressionMetadata<T>
{
    /// <summary>
    /// Initializes a new instance of the SparsePruningMetadata class.
    /// </summary>
    public SparsePruningMetadata(
        int[] nonZeroIndices,
        int originalLength,
        double threshold,
        double actualSparsity)
    {
        if (nonZeroIndices == null)
        {
            throw new ArgumentNullException(nameof(nonZeroIndices));
        }

        if (originalLength < 0)
        {
            throw new ArgumentException("Original length cannot be negative.", nameof(originalLength));
        }

        NonZeroIndices = nonZeroIndices;
        OriginalLength = originalLength;
        Threshold = threshold;
        ActualSparsity = actualSparsity;
    }

    /// <summary>
    /// Gets the compression type.
    /// </summary>
    public CompressionType Type => CompressionType.HybridClusteringPruning; // Using existing enum value for sparse pruning

    /// <summary>
    /// Gets the indices of non-zero values.
    /// </summary>
    public int[] NonZeroIndices { get; private set; }

    /// <summary>
    /// Gets the original length of the weights array.
    /// </summary>
    public int OriginalLength { get; private set; }

    /// <summary>
    /// Gets the threshold used for pruning.
    /// </summary>
    public double Threshold { get; private set; }

    /// <summary>
    /// Gets the actual sparsity achieved (fraction of zeros).
    /// </summary>
    public double ActualSparsity { get; private set; }

    /// <summary>
    /// Gets the size in bytes of this metadata structure.
    /// </summary>
    public long GetMetadataSize()
    {
        // Size of indices + original length + threshold + actual sparsity
        return (NonZeroIndices.Length * sizeof(int)) + sizeof(int) + sizeof(double) * 2;
    }
}
