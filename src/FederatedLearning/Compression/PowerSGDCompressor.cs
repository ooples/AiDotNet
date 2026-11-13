using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class PowerSGDCompressor<T> : FederatedLearningComponentBase<T>
{
    private readonly AdvancedCompressionOptions _options;
    private readonly Dictionary<int, Tensor<T>> _clientQFactors = new();

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
    public (Matrix<T> P, Matrix<T> Q, int Rows, int Cols) Compress(Tensor<T> gradient, int clientId)
    {
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));

        int totalSize = gradient.Shape[0];

        // Reshape 1D gradient into a 2D matrix (approximately square), zero-padding the tail.
        int rows = (int)Math.Ceiling(Math.Sqrt(totalSize));
        int cols = (int)Math.Ceiling((double)totalSize / rows);
        int rank = EffectiveRank(rows, cols);
        var padded = new Tensor<T>(new[] { rows * cols });
        gradient.AsSpan().Slice(0, totalSize).CopyTo(padded.AsWritableSpan());
        var g = padded.Reshape(rows, cols);

        // Initialize or reuse Q factor
        Tensor<T> q;
        if (_options.PowerSGDWarmStart && _clientQFactors.TryGetValue(clientId, out var stored)
            && stored.Shape[0] == cols && stored.Shape[1] == rank)
        {
            q = stored;
        }
        else
        {
            q = InitializeRandomMatrix(cols, rank);
        }

        // Vogels et al. 2019, Algorithm 1: P = G Q, P <- orthogonalize(P), Q = G^T P.
        var p = Engine.TensorMatMul(g, q);
        p = AiDotNet.Helpers.LinalgDispatch<T>.QR(p, "reduced").Q;
        q = Engine.TensorMatMul(Engine.TensorTranspose(g), p);

        // Store Q for warm-start
        if (_options.PowerSGDWarmStart)
        {
            _clientQFactors[clientId] = q;
        }

        return (p.ToMatrix(), q.ToMatrix(), rows, cols);
    }

    /// <summary>
    /// Decompresses PowerSGD factors back to a gradient tensor.
    /// </summary>
    /// <param name="P">The P factor (rows x rank).</param>
    /// <param name="Q">The Q factor (cols x rank).</param>
    /// <param name="rows">Number of rows in the reshaped gradient matrix.</param>
    /// <param name="cols">Number of columns in the reshaped gradient matrix.</param>
    /// <param name="originalSize">Original 1D gradient size.</param>
    /// <returns>Reconstructed gradient tensor.</returns>
    public Tensor<T> Decompress(Matrix<T> P, Matrix<T> Q, int rows, int cols, int originalSize)
    {
        if (P is null) throw new ArgumentNullException(nameof(P));
        if (Q is null) throw new ArgumentNullException(nameof(Q));

        // Reconstruct: G_approx = P * Q^T
        var g = Engine.TensorMatMul(Tensor<T>.FromMatrix(P), Engine.TensorTranspose(Tensor<T>.FromMatrix(Q)));
        var flat = g.Reshape(rows * cols);
        int kept = Math.Min(originalSize, rows * cols);

        var result = new Tensor<T>(new[] { originalSize });
        flat.AsSpan().Slice(0, kept).CopyTo(result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// Gets the compression ratio achieved (compressed size / original size).
    /// </summary>
    public double GetCompressionRatio(int originalSize)
    {
        int rows = (int)Math.Ceiling(Math.Sqrt(originalSize));
        int cols = (int)Math.Ceiling((double)originalSize / rows);
        int rank = EffectiveRank(rows, cols);

        double compressedSize = (rows * rank) + (cols * rank);
        return compressedSize / originalSize;
    }

    /// <summary>
    /// The approximation rank min(r, rows, cols) for a rows x cols gradient matrix: a rank above min(rows, cols) has
    /// no extra columns to fill (the orthogonalized P has at most min(rows, cols) columns), as in the reference
    /// PowerSGD hook's <c>min(n, m, matrix_approximation_rank)</c>.
    /// </summary>
    private int EffectiveRank(int rows, int cols) => Math.Min(_options.PowerSGDRank, Math.Min(rows, cols));

    private static Tensor<T> InitializeRandomMatrix(int rows, int cols)
    {
        // Gaussian initialization N(0, 1/cols).
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(cols));
        return Engine.TensorRandomNormal(new[] { rows, cols }, NumOps.Zero, scale);
    }
}
