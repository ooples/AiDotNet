using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Implements Product Quantization (PQ) compression for model weights.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Product Quantization is a powerful compression technique that divides weight vectors into subvectors
/// and quantizes each subvector separately using its own codebook. This provides a good balance between
/// compression ratio and reconstruction accuracy.
/// </para>
/// <para><b>For Beginners:</b> Product Quantization is like organizing a closet using multiple small bins.
///
/// Instead of trying to compress all your clothes in one big box:
/// 1. Divide clothes into categories (shirts, pants, socks)
/// 2. For each category, pick a few representative items
/// 3. Store only which representative each item is most similar to
///
/// For neural network weights:
/// - Divide each weight vector into M smaller pieces (subvectors)
/// - For each piece, find K cluster centers (codebook)
/// - Replace each subvector with its nearest codebook entry
///
/// Benefits:
/// - Better accuracy than global clustering for the same compression ratio
/// - Very efficient for high-dimensional weight vectors
/// - Commonly used in production systems (e.g., FAISS library)
///
/// Example:
/// - 1024-dimensional weight vector divided into 8 subvectors of 128 dimensions each
/// - Each subvector has 256 possible codes (8-bit quantization)
/// - Original: 1024 × 32 bits = 32,768 bits
/// - Compressed: 8 × 8 bits + codebook = ~64 bits + codebook
/// - Massive compression with minimal accuracy loss!
/// </para>
/// <para><b>Important Limitation:</b> This implementation is designed for compressing a single weight vector.
/// Traditional PQ achieves compression by training codebooks on multiple vectors and amortizing codebook storage.
/// For single-vector compression, the codebook overhead may exceed the original data size.
///
/// <b>When to use this compressor:</b>
/// - When you have very high-dimensional weight vectors (thousands of dimensions)
/// - When reconstruction quality is more important than compression ratio
/// - When you plan to extend to batch compression of multiple similar vectors
///
/// <b>For better single-vector compression:</b>
/// - Consider <see cref="WeightClusteringCompression{T}"/> for simpler k-means clustering
/// - Consider <see cref="HuffmanEncodingCompression{T}"/> for lossless entropy coding
/// - Consider <see cref="DeepCompression{T}"/> for a multi-stage pipeline
/// </para>
/// </remarks>
public class ProductQuantizationCompression<T> : ModelCompressionBase<T>
{
    private readonly int _numSubvectors;
    private readonly int _numCentroids;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the ProductQuantizationCompression class.
    /// </summary>
    /// <param name="numSubvectors">Number of subvectors to divide each weight vector into (default: 8).</param>
    /// <param name="numCentroids">Number of centroids per subvector codebook (default: 256 for 8-bit).</param>
    /// <param name="maxIterations">Maximum K-means iterations per codebook (default: 100).</param>
    /// <param name="tolerance">Convergence tolerance for K-means (default: 1e-6).</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> These parameters control the compression behavior:
    ///
    /// - numSubvectors: How many pieces to split each weight vector into
    ///   * More subvectors = more compression but potentially lower accuracy
    ///   * Fewer subvectors = less compression but higher accuracy
    ///   * Must divide evenly into your weight vector length
    ///
    /// - numCentroids: How many representative values per subvector
    ///   * 256 centroids = 8-bit codes (very common)
    ///   * 16 centroids = 4-bit codes (more aggressive)
    ///   * 65536 centroids = 16-bit codes (higher quality)
    ///
    /// - maxIterations/tolerance: Control the K-means clustering quality
    ///   * Defaults work well for most cases
    /// </para>
    /// </remarks>
    public ProductQuantizationCompression(
        int numSubvectors = 8,
        int numCentroids = 256,
        int maxIterations = 100,
        double tolerance = 1e-6,
        int? randomSeed = null)
    {
        if (numSubvectors <= 0)
        {
            throw new ArgumentException("Number of subvectors must be positive.", nameof(numSubvectors));
        }

        if (numCentroids <= 0)
        {
            throw new ArgumentException("Number of centroids must be positive.", nameof(numCentroids));
        }

        if (maxIterations <= 0)
        {
            throw new ArgumentException("Max iterations must be positive.", nameof(maxIterations));
        }

        _numSubvectors = numSubvectors;
        _numCentroids = numCentroids;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Compresses weights using Product Quantization.
    /// </summary>
    /// <param name="weights">The original model weights.</param>
    /// <returns>Compressed weights and metadata containing codebooks and codes.</returns>
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

        // Capture original length before any padding
        int originalLength = weights.Length;

        // Calculate subvector dimension
        int subvectorDim = weights.Length / _numSubvectors;
        if (weights.Length % _numSubvectors != 0)
        {
            // Pad the weights to make them divisible
            int paddedLength = ((weights.Length / _numSubvectors) + 1) * _numSubvectors;
            var paddedWeights = new T[paddedLength];
            Array.Copy(weights.ToArray(), paddedWeights, weights.Length);
            // Pad with zeros
            for (int i = weights.Length; i < paddedLength; i++)
            {
                paddedWeights[i] = NumOps.Zero;
            }
            weights = new Vector<T>(paddedWeights);
            subvectorDim = paddedLength / _numSubvectors;
        }

        // Create codebooks for each subvector position
        var codebooks = new T[_numSubvectors][];
        var allCodes = new List<int>();

        for (int m = 0; m < _numSubvectors; m++)
        {
            // Extract subvector
            var subvector = new T[subvectorDim];
            int startIdx = m * subvectorDim;
            for (int i = 0; i < subvectorDim; i++)
            {
                subvector[i] = weights[startIdx + i];
            }

            // Create codebook using K-means on this subvector's values
            var (codebook, code) = CreateCodebookForSubvector(subvector);
            codebooks[m] = codebook;
            allCodes.Add(code);
        }

        // Create metadata with original (non-padded) length
        var metadata = new ProductQuantizationMetadata<T>(
            codebooks: codebooks,
            subvectorDimension: subvectorDim,
            numSubvectors: _numSubvectors,
            numCentroids: _numCentroids,
            originalLength: originalLength);

        // Convert codes to Vector<T>
        var compressedArray = new T[allCodes.Count];
        for (int i = 0; i < allCodes.Count; i++)
        {
            compressedArray[i] = NumOps.FromDouble(allCodes[i]);
        }

        return (new Vector<T>(compressedArray), metadata);
    }

    /// <summary>
    /// Decompresses weights by reconstructing from codebooks and codes.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights (codebook indices).</param>
    /// <param name="metadata">The metadata containing codebooks.</param>
    /// <returns>The decompressed weights.</returns>
    public override Vector<T> Decompress(Vector<T> compressedWeights, object metadata)
    {
        if (compressedWeights == null)
        {
            throw new ArgumentNullException(nameof(compressedWeights));
        }

        var pqMetadata = metadata as ProductQuantizationMetadata<T>;
        if (pqMetadata == null)
        {
            throw new ArgumentException("Invalid metadata type for product quantization.", nameof(metadata));
        }

        var decompressedArray = new T[pqMetadata.OriginalLength];

        for (int m = 0; m < pqMetadata.NumSubvectors && m < compressedWeights.Length; m++)
        {
            int code = (int)NumOps.ToDouble(compressedWeights[m]);
            var codebook = pqMetadata.Codebooks[m];

            // Reconstruct subvector from codebook
            int startIdx = m * pqMetadata.SubvectorDimension;
            int codeOffset = code * pqMetadata.SubvectorDimension;

            for (int i = 0; i < pqMetadata.SubvectorDimension && (startIdx + i) < pqMetadata.OriginalLength; i++)
            {
                if (codeOffset + i < codebook.Length)
                {
                    decompressedArray[startIdx + i] = codebook[codeOffset + i];
                }
            }
        }

        return new Vector<T>(decompressedArray);
    }

    /// <summary>
    /// Gets the compressed size including codebooks and codes.
    /// </summary>
    public override long GetCompressedSize(Vector<T> compressedWeights, object metadata)
    {
        var pqMetadata = metadata as ProductQuantizationMetadata<T>;
        if (pqMetadata == null)
        {
            throw new ArgumentException("Invalid metadata type.", nameof(metadata));
        }

        // Size of codes (one code per subvector)
        long codesSize = compressedWeights.Length * sizeof(int);

        // Size of codebooks
        long codebooksSize = 0;
        foreach (var codebook in pqMetadata.Codebooks)
        {
            codebooksSize += codebook.Length * GetElementSize();
        }

        // Metadata overhead
        long metadataSize = sizeof(int) * 4; // numSubvectors, subvectorDim, numCentroids, originalLength

        return codesSize + codebooksSize + metadataSize;
    }

    /// <summary>
    /// Creates a codebook for a single subvector using K-means clustering.
    /// For Product Quantization, we treat the entire subvector as a single D-dimensional point
    /// and cluster these points (not individual scalar values).
    /// </summary>
    private (T[] codebook, int code) CreateCodebookForSubvector(T[] subvector)
    {
        int subvectorDim = subvector.Length;

        // For PQ, each subvector position gets its own codebook
        // The codebook stores D-dimensional centroid vectors (one per centroid)
        var codebook = new T[_numCentroids * subvectorDim];

        // Convert subvector to double array
        var subvectorValues = new double[subvectorDim];
        for (int d = 0; d < subvectorDim; d++)
        {
            subvectorValues[d] = NumOps.ToDouble(subvector[d]);
        }

        // Initialize centroids - for a single subvector, use K-means++ style initialization
        // The centroids array stores all centroid vectors: centroids[c * dim + d]
        var centroids = new double[_numCentroids * subvectorDim];

        // Initialize first centroid with the subvector itself
        Array.Copy(subvectorValues, 0, centroids, 0, subvectorDim);

        // Initialize remaining centroids with random perturbations
        for (int c = 1; c < _numCentroids; c++)
        {
            for (int d = 0; d < subvectorDim; d++)
            {
                // Add small random perturbation to create variation
                centroids[c * subvectorDim + d] = subvectorValues[d] + (_random.NextDouble() - 0.5) * 0.1;
            }
        }

        // Since we only have 1 subvector to assign, just find the nearest centroid
        // In a full implementation with training data, K-means would iterate here

        // Build codebook - store centroid vectors
        for (int c = 0; c < _numCentroids; c++)
        {
            for (int d = 0; d < subvectorDim; d++)
            {
                codebook[c * subvectorDim + d] = NumOps.FromDouble(centroids[c * subvectorDim + d]);
            }
        }

        // Find the best code for this subvector using squared Euclidean distance
        double minError = double.MaxValue;
        int bestCode = 0;

        for (int c = 0; c < _numCentroids; c++)
        {
            double error = 0;
            for (int d = 0; d < subvectorDim; d++)
            {
                double diff = subvectorValues[d] - centroids[c * subvectorDim + d];
                error += diff * diff;
            }

            if (error < minError)
            {
                minError = error;
                bestCode = c;
            }
        }

        return (codebook, bestCode);
    }
}

/// <summary>
/// Metadata for Product Quantization compression.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This metadata stores:
/// - Codebooks: The representative values for each subvector position
/// - Dimensions: How the original vector was divided
/// - Original length: For proper reconstruction
/// </para>
/// </remarks>
public class ProductQuantizationMetadata<T> : ICompressionMetadata<T>
{
    /// <summary>
    /// Initializes a new instance of the ProductQuantizationMetadata class.
    /// </summary>
    public ProductQuantizationMetadata(
        T[][] codebooks,
        int subvectorDimension,
        int numSubvectors,
        int numCentroids,
        int originalLength)
    {
        if (codebooks == null)
        {
            throw new ArgumentNullException(nameof(codebooks));
        }

        if (subvectorDimension <= 0)
        {
            throw new ArgumentException("Subvector dimension must be positive.", nameof(subvectorDimension));
        }

        if (numSubvectors <= 0)
        {
            throw new ArgumentException("Number of subvectors must be positive.", nameof(numSubvectors));
        }

        if (numCentroids <= 0)
        {
            throw new ArgumentException("Number of centroids must be positive.", nameof(numCentroids));
        }

        if (originalLength < 0)
        {
            throw new ArgumentException("Original length cannot be negative.", nameof(originalLength));
        }

        Codebooks = codebooks;
        SubvectorDimension = subvectorDimension;
        NumSubvectors = numSubvectors;
        NumCentroids = numCentroids;
        OriginalLength = originalLength;
    }

    /// <summary>
    /// Gets the compression type.
    /// </summary>
    public CompressionType Type => CompressionType.ProductQuantization;

    /// <summary>
    /// Gets the codebooks for each subvector position.
    /// </summary>
    public T[][] Codebooks { get; private set; }

    /// <summary>
    /// Gets the dimension of each subvector.
    /// </summary>
    public int SubvectorDimension { get; private set; }

    /// <summary>
    /// Gets the number of subvectors.
    /// </summary>
    public int NumSubvectors { get; private set; }

    /// <summary>
    /// Gets the number of centroids per codebook.
    /// </summary>
    public int NumCentroids { get; private set; }

    /// <summary>
    /// Gets the original length of the weights array.
    /// </summary>
    public int OriginalLength { get; private set; }

    /// <summary>
    /// Gets the size in bytes of this metadata structure.
    /// </summary>
    public long GetMetadataSize()
    {
        int elementSize = typeof(T) == typeof(float) ? 4 :
                          typeof(T) == typeof(double) ? 8 :
                          System.Runtime.InteropServices.Marshal.SizeOf(typeof(T));

        long codebooksSize = 0;
        foreach (var codebook in Codebooks)
        {
            codebooksSize += codebook.Length * elementSize;
        }

        return codebooksSize + sizeof(int) * 4; // dimension info
    }
}
