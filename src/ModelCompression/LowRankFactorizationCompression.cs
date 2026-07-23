using System;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Implements Low-Rank Factorization compression using SVD-like decomposition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Low-Rank Factorization approximates weight matrices by decomposing them into products of
/// smaller matrices. This is based on the observation that many neural network weight matrices
/// are approximately low-rank, meaning they can be represented with fewer parameters.
/// </para>
/// <para><b>For Beginners:</b> Low-Rank Factorization is like summarizing a book.
///
/// The concept:
/// - A weight matrix might be 1000×1000 = 1,000,000 parameters
/// - But the actual "information content" might be much smaller
/// - We can approximate it as: W ≈ A × B where A is 1000×50 and B is 50×1000
/// - Now we only store: 50,000 + 50,000 = 100,000 parameters (10x compression!)
///
/// How it works:
/// 1. Treat the weight vector as a matrix (reshape it)
/// 2. Perform approximate factorization (similar to SVD)
/// 3. Keep only the top-k singular values/vectors
/// 4. Store the factored matrices instead of the original
///
/// Benefits:
/// - Compression ratio is controlled by the rank k
/// - Works especially well for fully-connected layers
/// - Maintains smoothness in the weight space
///
/// Trade-offs:
/// - Need to choose the rank k (compression vs accuracy trade-off)
/// - Works best when weights have inherent low-rank structure
/// </para>
/// </remarks>
public class LowRankFactorizationCompression<T> : ModelCompressionBase<T>
{
    private readonly int _targetRank;
    private readonly double _energyThreshold;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    /// <summary>
    /// Initializes a new instance of the LowRankFactorizationCompression class.
    /// </summary>
    /// <param name="targetRank">Target rank for the factorization (default: 0 = auto based on energy).</param>
    /// <param name="energyThreshold">Minimum energy to preserve (default: 0.95 = 95%).</param>
    /// <param name="maxIterations">Maximum iterations for power method (default: 100).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-6).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> These parameters control the factorization:
    ///
    /// - targetRank: How many dimensions to keep
    ///   * Lower rank = more compression but potentially less accuracy
    ///   * If 0, automatically determined by energyThreshold
    ///
    /// - energyThreshold: What fraction of "information" to preserve
    ///   * 0.95 = keep 95% of the variance (recommended)
    ///   * 0.99 = keep 99% (higher quality, less compression)
    ///   * 0.90 = keep 90% (more compression, lower quality)
    ///
    /// - maxIterations/tolerance: Control the numerical algorithm
    ///   * Defaults work well for most cases
    /// </para>
    /// </remarks>
    public LowRankFactorizationCompression(
        int targetRank = 0,
        double energyThreshold = 0.95,
        int maxIterations = 100,
        double tolerance = 1e-6)
    {
        if (targetRank < 0)
        {
            throw new ArgumentException("Target rank cannot be negative.", nameof(targetRank));
        }

        if (energyThreshold <= 0 || energyThreshold > 1)
        {
            throw new ArgumentException("Energy threshold must be between 0 and 1.", nameof(energyThreshold));
        }

        if (maxIterations <= 0)
        {
            throw new ArgumentException("Max iterations must be positive.", nameof(maxIterations));
        }

        _targetRank = targetRank;
        _energyThreshold = energyThreshold;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <summary>
    /// Compresses weights using low-rank factorization.
    /// </summary>
    /// <param name="weights">The original model weights.</param>
    /// <returns>Factored representation and metadata.</returns>
    public override (Vector<T> compressedWeights, ICompressionMetadata<T> metadata) Compress(Vector<T> weights)
    {
        if (weights == null) throw new ArgumentNullException(nameof(weights));

        if (weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be empty.", nameof(weights));
        }

        // Reshape weights into approximate square matrix for SVD
        int rows, cols;
        GetMatrixDimensions(weights.Length, out rows, out cols);

        // Create matrix from weights
        var matrix = new double[rows, cols];
        int idx = 0;
        for (int i = 0; i < rows && idx < weights.Length; i++)
        {
            for (int j = 0; j < cols && idx < weights.Length; j++)
            {
                matrix[i, j] = NumOps.ToDouble(weights[idx++]);
            }
        }

        // Perform approximate SVD using power iteration
        var (U, singularValues, V, effectiveRank) = ApproximateSVD(matrix, rows, cols);

        // Determine rank to keep
        int rank = _targetRank > 0 ? Math.Min(_targetRank, effectiveRank) : effectiveRank;

        // Create compressed representation: [U columns, singular values, V columns]
        var compressedList = new List<T>();

        // Store U matrix (truncated to rank)
        for (int i = 0; i < rows; i++)
        {
            for (int r = 0; r < rank; r++)
            {
                compressedList.Add(NumOps.FromDouble(U[i, r]));
            }
        }

        // Store singular values
        for (int r = 0; r < rank; r++)
        {
            compressedList.Add(NumOps.FromDouble(singularValues[r]));
        }

        // Store V matrix (truncated to rank)
        for (int r = 0; r < rank; r++)
        {
            for (int j = 0; j < cols; j++)
            {
                compressedList.Add(NumOps.FromDouble(V[r, j]));
            }
        }

        // Create metadata
        var metadata = new LowRankFactorizationMetadata<T>(
            rows: rows,
            cols: cols,
            rank: rank,
            originalLength: weights.Length);

        return (new Vector<T>(compressedList.ToArray()), metadata);
    }

    /// <summary>
    /// Decompresses by reconstructing from U, S, V factors.
    /// </summary>
    public override Vector<T> Decompress(Vector<T> compressedWeights, ICompressionMetadata<T> metadata)
    {
        if (compressedWeights == null) throw new ArgumentNullException(nameof(compressedWeights));
        if (metadata == null) throw new ArgumentNullException(nameof(metadata));

        if (metadata is not LowRankFactorizationMetadata<T> lrMetadata)
        {
            throw new ArgumentException(
                $"Expected {nameof(LowRankFactorizationMetadata<T>)} but received {metadata.GetType().Name}.",
                nameof(metadata));
        }

        int rows = lrMetadata.Rows;
        int cols = lrMetadata.Cols;
        int rank = lrMetadata.Rank;

        // Validate compressed weights length matches expected layout
        int expectedLength = (rows * rank) + rank + (rank * cols);
        if (compressedWeights.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Compressed weights length ({compressedWeights.Length}) does not match expected layout " +
                $"({expectedLength} = U[{rows}×{rank}] + S[{rank}] + V[{rank}×{cols}]).",
                nameof(compressedWeights));
        }

        // Extract U, S, V from compressed weights
        int idx = 0;

        // Extract U
        var U = new double[rows, rank];
        for (int i = 0; i < rows; i++)
        {
            for (int r = 0; r < rank && idx < compressedWeights.Length; r++)
            {
                U[i, r] = NumOps.ToDouble(compressedWeights[idx++]);
            }
        }

        // Extract singular values
        var S = new double[rank];
        for (int r = 0; r < rank && idx < compressedWeights.Length; r++)
        {
            S[r] = NumOps.ToDouble(compressedWeights[idx++]);
        }

        // Extract V
        var V = new double[rank, cols];
        for (int r = 0; r < rank; r++)
        {
            for (int j = 0; j < cols && idx < compressedWeights.Length; j++)
            {
                V[r, j] = NumOps.ToDouble(compressedWeights[idx++]);
            }
        }

        // Reconstruct: W = U * diag(S) * V
        var reconstructed = new T[lrMetadata.OriginalLength];
        idx = 0;
        for (int i = 0; i < rows && idx < lrMetadata.OriginalLength; i++)
        {
            for (int j = 0; j < cols && idx < lrMetadata.OriginalLength; j++)
            {
                double value = 0;
                for (int r = 0; r < rank; r++)
                {
                    value += U[i, r] * S[r] * V[r, j];
                }
                reconstructed[idx++] = NumOps.FromDouble(value);
            }
        }

        return new Vector<T>(reconstructed);
    }

    /// <summary>
    /// Gets the compressed size.
    /// </summary>
    public override long GetCompressedSize(Vector<T> compressedWeights, ICompressionMetadata<T> metadata)
    {
        if (compressedWeights == null) throw new ArgumentNullException(nameof(compressedWeights));
        if (metadata == null) throw new ArgumentNullException(nameof(metadata));

        if (metadata is not LowRankFactorizationMetadata<T> lrMetadata)
        {
            throw new ArgumentException(
                $"Expected {nameof(LowRankFactorizationMetadata<T>)} but received {metadata.GetType().Name}.",
                nameof(metadata));
        }

        // Size = U (rows × rank) + S (rank) + V (rank × cols)
        long dataSize = compressedWeights.Length * GetElementSize();

        return dataSize + lrMetadata.GetMetadataSize();
    }

    /// <summary>
    /// Gets appropriate matrix dimensions for the weight vector.
    /// </summary>
    private void GetMatrixDimensions(int length, out int rows, out int cols)
    {
        // Find factors close to square root
        int sqrt = (int)Math.Sqrt(length);

        // Find the largest factor <= sqrt
        rows = sqrt;
        while (rows > 1 && length % rows != 0)
        {
            rows--;
        }

        if (rows <= 1)
        {
            // If no good factor found, use sqrt approximation
            rows = sqrt > 0 ? sqrt : 1;
        }

        cols = (length + rows - 1) / rows; // Ceiling division
    }

    /// <summary>
    /// Performs approximate SVD using power iteration.
    /// </summary>
    private (double[,] U, double[] S, double[,] V, int effectiveRank) ApproximateSVD(
        double[,] matrix, int rows, int cols)
    {
        int maxRank = Math.Min(rows, cols);
        int rank = _targetRank > 0 ? Math.Min(_targetRank, maxRank) : maxRank;

        var U = new double[rows, rank];
        var S = new double[rank];
        var V = new double[rank, cols];

        // Working copy of matrix for deflation
        var A = (double[,])matrix.Clone();

        double totalEnergy = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                totalEnergy += A[i, j] * A[i, j];
            }
        }

        double capturedEnergy = 0;
        int effectiveRank = 0;

        var random = RandomHelper.CreateSeededRandom(42);

        for (int r = 0; r < rank; r++)
        {
            // Power iteration to find dominant singular vector
            var v = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                v[j] = random.NextDouble() - 0.5;
            }
            Normalize(v);

            double sigma = 0;
            double prevSigma = 0;
            var u = new double[rows];

            for (int iter = 0; iter < _maxIterations; iter++)
            {
                // u = A * v
                Array.Clear(u, 0, u.Length);
                for (int i = 0; i < rows; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < cols; j++)
                    {
                        sum += A[i, j] * v[j];
                    }
                    u[i] = sum;
                }

                sigma = Normalize(u);

                if (sigma < 1e-10)
                {
                    break;
                }

                // v = A^T * u
                for (int j = 0; j < cols; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < rows; i++)
                    {
                        sum += A[i, j] * u[i];
                    }
                    v[j] = sum;
                }

                Normalize(v);

                // Check convergence
                if (Math.Abs(sigma - prevSigma) < _tolerance * sigma)
                {
                    break;
                }
                prevSigma = sigma;
            }

            // Skip this component if sigma is effectively zero
            // (no more meaningful singular values remain)
            if (sigma < 1e-10)
            {
                break;
            }

            // Persist final iterate for this component
            for (int i = 0; i < rows; i++)
            {
                U[i, r] = u[i];
            }
            S[r] = sigma;
            for (int j = 0; j < cols; j++)
            {
                V[r, j] = v[j];
            }

            // Deflate: A = A - sigma * u * v^T
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    A[i, j] -= sigma * U[i, r] * V[r, j];
                }
            }

            capturedEnergy += sigma * sigma;
            effectiveRank = r + 1;

            // Check energy threshold
            if (totalEnergy > 0 && capturedEnergy / totalEnergy >= _energyThreshold)
            {
                break;
            }
        }

        return (U, S, V, effectiveRank);
    }

    /// <summary>
    /// Normalizes a vector and returns its norm.
    /// </summary>
    private double Normalize(double[] v)
    {
        double norm = 0;
        for (int i = 0; i < v.Length; i++)
        {
            norm += v[i] * v[i];
        }
        norm = Math.Sqrt(norm);

        if (norm > 1e-10)
        {
            for (int i = 0; i < v.Length; i++)
            {
                v[i] /= norm;
            }
        }

        return norm;
    }
}

/// <summary>
/// Metadata for Low-Rank Factorization compression.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This metadata stores:
/// - Matrix dimensions (how the vector was reshaped)
/// - Rank used (how many singular values kept)
/// - Original length (for reconstruction)
/// </para>
/// </remarks>
public class LowRankFactorizationMetadata<T> : ICompressionMetadata<T>
{
    /// <summary>
    /// Initializes a new instance of the LowRankFactorizationMetadata class.
    /// </summary>
    public LowRankFactorizationMetadata(int rows, int cols, int rank, int originalLength)
    {
        if (rows <= 0)
        {
            throw new ArgumentException("Rows must be positive.", nameof(rows));
        }

        if (cols <= 0)
        {
            throw new ArgumentException("Cols must be positive.", nameof(cols));
        }

        if (rank <= 0)
        {
            throw new ArgumentException("Rank must be positive.", nameof(rank));
        }

        if (originalLength < 0)
        {
            throw new ArgumentException("Original length cannot be negative.", nameof(originalLength));
        }

        Rows = rows;
        Cols = cols;
        Rank = rank;
        OriginalLength = originalLength;
    }

    /// <summary>
    /// Gets the compression type.
    /// </summary>
    public CompressionType Type => CompressionType.LowRankFactorization;

    /// <summary>
    /// Gets the number of rows in the reshaped matrix.
    /// </summary>
    public int Rows { get; private set; }

    /// <summary>
    /// Gets the number of columns in the reshaped matrix.
    /// </summary>
    public int Cols { get; private set; }

    /// <summary>
    /// Gets the rank of the factorization.
    /// </summary>
    public int Rank { get; private set; }

    /// <summary>
    /// Gets the original length of the weights array.
    /// </summary>
    public int OriginalLength { get; private set; }

    /// <summary>
    /// Gets the size in bytes of this metadata structure.
    /// </summary>
    public long GetMetadataSize()
    {
        return sizeof(int) * 4; // rows, cols, rank, originalLength
    }
}
