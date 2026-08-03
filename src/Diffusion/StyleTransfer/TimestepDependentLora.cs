using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// Timestep-dependent low-rank adaptation: the mechanism of T-LoRA, in which the effective rank of
/// the adapter update SHRINKS as the diffusion timestep rises.
/// </summary>
/// <remarks>
/// <para>
/// Soboleva, Alanov, Kuznetsov and Sobolev, "T-LoRA: Single Image Diffusion Model Customization
/// Without Overfitting" (arXiv:2507.05964). The "T" is Timestep-Dependent, and the finding it rests
/// on is empirical: "higher diffusion timesteps are more prone to overfitting than lower ones,
/// necessitating a timestep-sensitive fine-tuning strategy."
/// </para>
/// <para>
/// So the adapter is given less freedom exactly where overfitting happens. At high timesteps — heavy
/// noise, where a single training image most easily dictates the output — only the leading directions
/// of the update survive. At low timesteps, where the model is refining detail, the full rank is
/// available.
/// </para>
/// <para>
/// <b>The paper's rank schedule (Section 3.2):</b>
/// </para>
/// <code>
///   r(t)   = floor((r - r_min) * (T - t) / T) + r_min      with r_min = 50% of r
///   M_t    = diag(1 for i &lt; r(t), else 0)
/// </code>
/// <para>
/// Note the FLOOR at <c>r_min</c>, not at one. The paper never strips the adapter down to a single
/// direction: even the most overfitting-prone timesteps keep half the rank. An earlier revision of
/// this class used <c>ceil(r * (1 - t/T))</c> clamped to a minimum of one, which both used the wrong
/// interpolation and over-constrained the high-timestep end far past what the paper does.
/// </para>
/// <para>
/// <b>Ortho-LoRA initialization (Section 3.1).</b> The second contribution, "a weight parametrization
/// technique that ensures independence between adapter components". Independence is what makes the
/// schedule meaningful — masking the tail of a set of CORRELATED directions removes no capacity,
/// because the survivors still span what was masked. The paper obtains it from the SVD of a random
/// matrix <c>R ~ N(0, 1/r)</c>, taking the trailing (least-dominant) triplet:
/// </para>
/// <code>
///   R = U S V^T,   A_init = V_r^T,   B_init = U_r,   S_init = S_r
/// </code>
/// <para>
/// <b>Why there is a third trainable matrix.</b> Standard LoRA gets "adapter is the identity at
/// initialization" for free by setting B = 0. This paper deliberately does NOT: it wants a non-zero
/// singular-value matrix S that training can move, so it recovers the identity property by
/// SUBTRACTING the frozen initial product instead —
/// </para>
/// <code>
///   W~ = W - B_init S_init M_t A_init + B S M_t A
/// </code>
/// <para>
/// which is exactly zero while <c>B = B_init, S = S_init, A = A_init</c> and diverges from zero only
/// as training moves them. Trainable: A, B and S. Frozen: W and the three <c>_init</c> factors.
/// </para>
/// <para>
/// <b>For Beginners:</b> Teaching a picture generator a new object from a SINGLE photo usually makes
/// it memorize that photo and lose its variety. It turns out the memorizing happens mostly during
/// the noisy early stage of generation. So this hands the model fewer knobs to turn during that
/// stage and all of them later, when it is only refining details.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class TimestepDependentLora<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly int _rank;
    private readonly int _minRank;
    private readonly int _totalTimesteps;

    // The frozen initialization triplet. Subtracting its masked product is what makes the adapter
    // the identity at init WITHOUT forcing B or S to start at zero (see the class remarks).
    private readonly Matrix<T> _downInit;
    private readonly Matrix<T> _upInit;
    private readonly Vector<T> _singularInit;

    /// <summary>Down-projection A, shape [rank, inputDim]. Trainable; rows start orthonormal.</summary>
    public Matrix<T> DownProjection { get; }

    /// <summary>Up-projection B, shape [outputDim, rank]. Trainable; columns start orthonormal.</summary>
    public Matrix<T> UpProjection { get; }

    /// <summary>
    /// The diagonal singular-value factor S, length <see cref="Rank"/>. Trainable, and NON-zero at
    /// initialization — the distinguishing feature of this paper's parametrization.
    /// </summary>
    public Vector<T> SingularValues { get; }

    /// <summary>Gets the adapter's full rank R, available at timestep zero.</summary>
    public int Rank => _rank;

    /// <summary>
    /// Gets the floor of the rank schedule, r_min — 50% of <see cref="Rank"/> per the paper. The
    /// effective rank never drops below this, even beyond the horizon.
    /// </summary>
    public int MinRank => _minRank;

    /// <summary>Gets the diffusion horizon T used to normalize the timestep.</summary>
    public int TotalTimesteps => _totalTimesteps;

    /// <summary>
    /// Initializes the adapter using the paper's Ortho-LoRA scheme.
    /// </summary>
    /// <param name="rank">Full rank R. Must not exceed <c>min(inputDim, outputDim)</c>.</param>
    /// <param name="inputDim">Width the adapter reads.</param>
    /// <param name="outputDim">Width the adapter writes.</param>
    /// <param name="totalTimesteps">Diffusion horizon T.</param>
    /// <param name="random">RNG for initialization; supply a seeded one for reproducibility.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when any dimension is not positive, or when <paramref name="rank"/> exceeds the ambient
    /// width — a rank larger than the space cannot have independent directions, and the paper's SVD
    /// initialization has no trailing triplet to draw them from.
    /// </exception>
    public TimestepDependentLora(int rank, int inputDim, int outputDim, int totalTimesteps, Random random)
    {
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank), rank, "Rank must be positive.");
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim), inputDim, "Input width must be positive.");
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim), outputDim, "Output width must be positive.");
        if (totalTimesteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalTimesteps), totalTimesteps,
                "The diffusion horizon must be positive; it is the denominator of the rank schedule.");
        if (random is null) throw new ArgumentNullException(nameof(random));

        int ambient = Math.Min(inputDim, outputDim);
        if (rank > ambient)
            throw new ArgumentOutOfRangeException(nameof(rank), rank,
                $"Rank cannot exceed min(inputDim, outputDim) = {ambient}. A rank larger than the ambient " +
                "width cannot have independent directions, so the orthogonal initialization that makes the " +
                "rank schedule meaningful would not exist.");

        _rank = rank;
        _totalTimesteps = totalTimesteps;

        // r_min = 50% of r. Kept at >= 1 so a rank-1 adapter stays connected rather than vanishing.
        _minRank = Math.Max(1, rank / 2);

        (DownProjection, UpProjection, SingularValues) = OrthoLoraInit(rank, inputDim, outputDim, random);

        // Frozen copies of the initialization, for the subtraction term.
        _downInit = DownProjection.Clone();
        _upInit = UpProjection.Clone();
        _singularInit = SingularValues.Clone();
    }

    /// <summary>
    /// The effective rank at timestep <paramref name="timestep"/>: full at t = 0, falling linearly to
    /// <see cref="MinRank"/> at the horizon.
    /// </summary>
    /// <remarks>
    /// <c>floor((r - r_min) * (T - t) / T) + r_min</c>, exactly as published. Never returns zero: a
    /// zero-rank adapter is not "maximally constrained", it is DISCONNECTED, and the paper's floor of
    /// r_min is far above that anyway.
    /// </remarks>
    public int EffectiveRank(int timestep)
    {
        int t = Math.Max(0, Math.Min(timestep, _totalTimesteps));
        double span = (double)(_rank - _minRank) * (_totalTimesteps - t) / _totalTimesteps;
        int effective = (int)Math.Floor(span) + _minRank;
        return Math.Max(_minRank, Math.Min(_rank, effective));
    }

    /// <summary>
    /// Applies the adapter at a given timestep: <c>(B S M_t A - B_init S_init M_t A_init) x</c>.
    /// </summary>
    /// <remarks>
    /// The subtraction is not an optimization detail — it IS the paper's reparametrization, and it is
    /// what allows S to be non-zero at initialization while the adapter still starts as the identity.
    /// </remarks>
    public Vector<T> Apply(Vector<T> input, int timestep)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        int effective = EffectiveRank(timestep);
        var output = new Vector<T>(UpProjection.Rows);

        // Trained branch minus frozen-initial branch, both masked by the same M_t.
        AccumulateBranch(input, effective, DownProjection, UpProjection, SingularValues, output, subtract: false);
        AccumulateBranch(input, effective, _downInit, _upInit, _singularInit, output, subtract: true);
        return output;
    }

    /// <summary>
    /// Adds (or subtracts) <c>B * diag(s) * M_t * A * x</c> into <paramref name="output"/>.
    /// </summary>
    private static void AccumulateBranch(
        Vector<T> input, int effective,
        Matrix<T> down, Matrix<T> up, Vector<T> singular,
        Vector<T> output, bool subtract)
    {
        int inputDim = down.Columns;
        int outputDim = up.Rows;

        // Down-project into the surviving directions, scaling by the singular values as we go.
        var latent = new double[effective];
        for (int r = 0; r < effective; r++)
        {
            double sum = 0.0;
            for (int i = 0; i < inputDim && i < input.Length; i++)
            {
                sum += Ops.ToDouble(down[r, i]) * Ops.ToDouble(input[i]);
            }
            latent[r] = sum * Ops.ToDouble(singular[r]);
        }

        for (int o = 0; o < outputDim; o++)
        {
            double sum = 0.0;
            for (int r = 0; r < effective; r++) sum += Ops.ToDouble(up[o, r]) * latent[r];
            double current = Ops.ToDouble(output[o]);
            output[o] = Ops.FromDouble(subtract ? current - sum : current + sum);
        }
    }

    /// <summary>
    /// The paper's Ortho-LoRA initialization: SVD of a random <c>R ~ N(0, 1/r)</c>, keeping the
    /// TRAILING rank-many singular directions.
    /// </summary>
    /// <remarks>
    /// The trailing triplet is chosen deliberately: those are the least-dominant directions of the
    /// random draw, so the adapter starts small in magnitude while still being exactly orthogonal.
    /// Taking the leading triplet instead would start the adaptation off at the largest directions of
    /// a random matrix, which is not what the paper specifies.
    /// </remarks>
    private static (Matrix<T> down, Matrix<T> up, Vector<T> singular) OrthoLoraInit(
        int rank, int inputDim, int outputDim, Random random)
    {
        // R ~ N(0, 1/r): standard deviation 1/sqrt(r).
        double stdDev = 1.0 / Math.Sqrt(rank);
        var r = new Matrix<T>(outputDim, inputDim);
        for (int row = 0; row < outputDim; row++)
        {
            for (int col = 0; col < inputDim; col++)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = random.NextDouble();
                double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                r[row, col] = Ops.FromDouble(gaussian * stdDev);
            }
        }

        var svd = new SvdDecomposition<T>(r);

        // Trailing `rank` triplet. S is ordered descending, so the trailing block is the tail.
        int available = Math.Min(svd.S.Length, Math.Min(svd.U.Columns, svd.Vt.Rows));
        int start = Math.Max(0, available - rank);

        var down = new Matrix<T>(rank, inputDim);   // A_init = V_r^T -> trailing ROWS of Vt
        var up = new Matrix<T>(outputDim, rank);    // B_init = U_r   -> trailing COLUMNS of U
        var singular = new Vector<T>(rank);         // S_init = S_r

        for (int k = 0; k < rank; k++)
        {
            int source = start + k;
            for (int c = 0; c < inputDim; c++) down[k, c] = svd.Vt[source, c];
            for (int o = 0; o < outputDim; o++) up[o, k] = svd.U[o, source];
            singular[k] = svd.S[source];
        }

        return (down, up, singular);
    }
}
