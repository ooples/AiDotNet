using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// PowerSGD: low-rank gradient compression using randomized SVD approximation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Sending a full gradient vector (millions of values) is expensive.
/// PowerSGD compresses it by finding a low-rank approximation — like summarizing a book with
/// just the key plot points. A rank-4 approximation of a 1M-parameter gradient only needs to
/// send ~8K values (500x compression!).</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Reshape the gradient into a matrix G (rows x cols).</description></item>
/// <item><description>Compute P = G * Q (project gradient onto random subspace Q).</description></item>
/// <item><description>Orthogonalize P using QR decomposition approximation.</description></item>
/// <item><description>Compute Q = G^T * P (recover subspace from projection).</description></item>
/// <item><description>Send P and Q instead of G (much smaller if rank &lt;&lt; min(rows, cols)).</description></item>
/// <item><description>Reconstruct: G_approx = P * Q^T.</description></item>
/// </list>
///
/// <para><b>Warm-start:</b> Reusing P/Q from previous rounds accelerates convergence because
/// the gradient subspace changes slowly between rounds.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class PowerSGDCompressor<T> : FederatedLearningComponentBase<T>
{
    private readonly AdvancedCompressionOptions _options;
    private readonly Dictionary<int, double[,]> _clientQFactors = new();

    /// <summary>
    /// Initializes a new instance of <see cref="PowerSGDCompressor{T}"/>.
    /// </summary>
    /// <param name="options">Advanced compression configuration.</param>
    public PowerSGDCompressor(AdvancedCompressionOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Compresses a gradient tensor using low-rank PowerSGD approximation.
    /// </summary>
    /// <param name="gradient">The gradient tensor to compress.</param>
    /// <param name="clientId">Client ID for warm-start factor reuse.</param>
    /// <returns>Compressed representation as (P factor, Q factor, original shape info).</returns>
    public (double[,] P, double[,] Q, int Rows, int Cols) Compress(Tensor<T> gradient, int clientId)
    {
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));

        int totalSize = gradient.Shape[0];
        int rank = Math.Min(_options.PowerSGDRank, totalSize);

        // Reshape 1D gradient into a 2D matrix (approximately square)
        int rows = (int)Math.Ceiling(Math.Sqrt(totalSize));
        int cols = (int)Math.Ceiling((double)totalSize / rows);

        // Build gradient matrix G
        var G = new double[rows, cols];
        for (int i = 0; i < totalSize; i++)
        {
            int r = i / cols;
            int c = i % cols;
            G[r, c] = NumOps.ToDouble(gradient[i]);
        }

        // Initialize or reuse Q factor
        double[,] Q;
        if (_options.PowerSGDWarmStart && _clientQFactors.ContainsKey(clientId))
        {
            Q = _clientQFactors[clientId];
            // Resize Q if needed
            if (Q.GetLength(0) != cols || Q.GetLength(1) != rank)
            {
                Q = InitializeRandomMatrix(cols, rank);
            }
        }
        else
        {
            Q = InitializeRandomMatrix(cols, rank);
        }

        // Step 1: P = G * Q (rows x rank)
        var P = MatMul(G, Q, rows, cols, rank);

        // Step 2: Orthogonalize P via modified Gram-Schmidt
        Orthogonalize(P, rows, rank);

        // Step 3: Q = G^T * P (cols x rank)
        var Gt = Transpose(G, rows, cols);
        Q = MatMul(Gt, P, cols, rows, rank);

        // Store Q for warm-start
        if (_options.PowerSGDWarmStart)
        {
            _clientQFactors[clientId] = Q;
        }

        return (P, Q, rows, cols);
    }

    /// <summary>
    /// Decompresses a PowerSGD representation back to a gradient tensor.
    /// </summary>
    /// <param name="P">P factor matrix.</param>
    /// <param name="Q">Q factor matrix.</param>
    /// <param name="rows">Number of rows in the gradient matrix.</param>
    /// <param name="cols">Number of columns in the gradient matrix.</param>
    /// <param name="originalSize">Original gradient vector size.</param>
    /// <returns>Reconstructed gradient tensor.</returns>
    public Tensor<T> Decompress(double[,] P, double[,] Q, int rows, int cols, int originalSize)
    {
        int rank = P.GetLength(1);

        // Reconstruct: G_approx = P * Q^T
        var Qt = Transpose(Q, cols, rank);
        var G = MatMulTranspose(P, Qt, rows, rank, cols);

        var result = new Tensor<T>(new[] { originalSize });
        for (int i = 0; i < originalSize; i++)
        {
            int r = i / cols;
            int c = i % cols;
            if (r < rows && c < cols)
            {
                result[i] = NumOps.FromDouble(G[r, c]);
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the compression ratio achieved (compressed size / original size).
    /// </summary>
    public double GetCompressionRatio(int originalSize)
    {
        int rows = (int)Math.Ceiling(Math.Sqrt(originalSize));
        int cols = (int)Math.Ceiling((double)originalSize / rows);
        int rank = Math.Min(_options.PowerSGDRank, originalSize);

        double compressedSize = (rows * rank) + (cols * rank);
        return compressedSize / originalSize;
    }

    private double[,] InitializeRandomMatrix(int rows, int cols)
    {
        var rng = RandomHelper.CreateSecureRandom();
        var matrix = new double[rows, cols];
        double scale = 1.0 / Math.Sqrt(cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Box-Muller for Gaussian initialization
                double u1 = 1.0 - rng.NextDouble();
                double u2 = rng.NextDouble();
                matrix[i, j] = scale * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            }
        }

        return matrix;
    }

    private static double[,] MatMul(double[,] A, double[,] B, int m, int k, int n)
    {
        var C = new double[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    sum += A[i, p] * B[p, j];
                }

                C[i, j] = sum;
            }
        }

        return C;
    }

    private static double[,] MatMulTranspose(double[,] A, double[,] Bt, int m, int k, int n)
    {
        // A (m x k) * B^T where Bt is stored as (k x n) transposed
        var C = new double[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    sum += A[i, p] * Bt[p, j];
                }

                C[i, j] = sum;
            }
        }

        return C;
    }

    private static double[,] Transpose(double[,] A, int rows, int cols)
    {
        var At = new double[cols, rows];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                At[j, i] = A[i, j];
            }
        }

        return At;
    }

    private static void Orthogonalize(double[,] P, int rows, int cols)
    {
        // Modified Gram-Schmidt orthogonalization
        for (int j = 0; j < cols; j++)
        {
            // Subtract projections of previous columns
            for (int k = 0; k < j; k++)
            {
                double dot = 0;
                for (int i = 0; i < rows; i++)
                {
                    dot += P[i, j] * P[i, k];
                }

                for (int i = 0; i < rows; i++)
                {
                    P[i, j] -= dot * P[i, k];
                }
            }

            // Normalize column
            double norm = 0;
            for (int i = 0; i < rows; i++)
            {
                norm += P[i, j] * P[i, j];
            }

            norm = Math.Sqrt(norm);
            if (norm > 1e-12)
            {
                for (int i = 0; i < rows; i++)
                {
                    P[i, j] /= norm;
                }
            }
        }
    }
}
