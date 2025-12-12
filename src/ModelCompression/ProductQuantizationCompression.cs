using System;
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

        // Create metadata
        var metadata = new ProductQuantizationMetadata<T>(
            codebooks: codebooks,
            subvectorDimension: subvectorDim,
            numSubvectors: _numSubvectors,
            numCentroids: _numCentroids,
            originalLength: weights.Length);

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
    /// </summary>
    private (T[] codebook, int code) CreateCodebookForSubvector(T[] subvector)
    {
        int effectiveCentroids = Math.Min(_numCentroids, subvector.Length);

        // Initialize codebook with K-means++
        var codebook = new T[effectiveCentroids * subvector.Length];
        var centroids = new double[effectiveCentroids];

        // Simple K-means for 1D values in the subvector
        // Flatten subvector values and cluster them
        var values = new double[subvector.Length];
        for (int i = 0; i < subvector.Length; i++)
        {
            values[i] = NumOps.ToDouble(subvector[i]);
        }

        // Initialize centroids randomly
        for (int i = 0; i < effectiveCentroids; i++)
        {
            centroids[i] = values[_random.Next(values.Length)];
        }

        // Run K-means
        var assignments = new int[values.Length];
        double previousInertia = double.MaxValue;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Assign each value to nearest centroid
            for (int i = 0; i < values.Length; i++)
            {
                double minDist = double.MaxValue;
                int nearestCentroid = 0;

                for (int c = 0; c < effectiveCentroids; c++)
                {
                    double dist = Math.Abs(values[i] - centroids[c]);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCentroid = c;
                    }
                }

                assignments[i] = nearestCentroid;
            }

            // Update centroids
            var sums = new double[effectiveCentroids];
            var counts = new int[effectiveCentroids];

            for (int i = 0; i < values.Length; i++)
            {
                sums[assignments[i]] += values[i];
                counts[assignments[i]]++;
            }

            for (int c = 0; c < effectiveCentroids; c++)
            {
                if (counts[c] > 0)
                {
                    centroids[c] = sums[c] / counts[c];
                }
            }

            // Check convergence
            double inertia = 0;
            for (int i = 0; i < values.Length; i++)
            {
                double diff = values[i] - centroids[assignments[i]];
                inertia += diff * diff;
            }

            if (Math.Abs(previousInertia - inertia) < _tolerance)
            {
                break;
            }
            previousInertia = inertia;
        }

        // Build codebook - store centroid values for each subvector dimension
        for (int c = 0; c < effectiveCentroids; c++)
        {
            for (int d = 0; d < subvector.Length; d++)
            {
                codebook[c * subvector.Length + d] = NumOps.FromDouble(centroids[c]);
            }
        }

        // Find the best code for this subvector
        double minError = double.MaxValue;
        int bestCode = 0;

        for (int c = 0; c < effectiveCentroids; c++)
        {
            double error = 0;
            for (int d = 0; d < subvector.Length; d++)
            {
                double diff = values[d] - centroids[c];
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
