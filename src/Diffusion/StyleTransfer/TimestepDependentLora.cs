using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Attributes;
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
public sealed partial class TimestepDependentLora<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    // Same accessor LayerBase exposes to layers. This class is not a layer, so it resolves it itself
    // rather than reimplementing the matrix products it needs in managed scalar code.
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _rank;
    private readonly int _minRank;
    private readonly int _totalTimesteps;

    // The frozen initialization triplet. Subtracting its masked product is what makes the adapter
    // the identity at init WITHOUT forcing B or S to start at zero (see the class remarks).
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Matrix<T> _downInit;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Matrix<T> _upInit;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _singularInit;

    // Cached [inputDim, outputDim] delta for _cachedTimestep. Invalidated by InvalidateCache().
    [Scratch]
    private Tensor<T>? _cachedDelta;
    private int _cachedTimestep = -1;

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
    /// The adapter's effective weight delta at a timestep, TRANSPOSED to <c>[inputDim, outputDim]</c>
    /// so it can be right-multiplied against a <c>[tokens, inputDim]</c> activation matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>dW(t) = B S M_t A - B_init S_init M_t A_init</c>, which depends only on the timestep and the
    /// current parameters — NOT on the activations. Forming it once per (timestep, parameter version)
    /// and applying it as a single matrix product is the whole point: the first version of this class
    /// evaluated the rank-masked product per TOKEN, which made a decorated UNet so slow that the
    /// diffusion test suite went from 14 seconds to over 10 minutes. Scalar per-element loops on a hot
    /// path are also against this repository's standing guidance.
    /// </para>
    /// <para>
    /// Cached because a denoising loop revisits the same timestep across blocks and the parameters do
    /// not move within a forward pass. <see cref="InvalidateCache"/> must be called after any write to
    /// A, B or S, or the cache would serve a delta computed from stale weights.
    /// </para>
    /// </remarks>
    public Tensor<T> EffectiveDeltaTransposed(int timestep)
    {
        int effective = EffectiveRank(timestep);
        if (_cachedDelta is not null && _cachedTimestep == timestep) return _cachedDelta;

        int inputDim = DownProjection.Columns;
        int outputDim = UpProjection.Rows;

        // Both branches are ordinary matrix products, so the ENGINE does them. The previous version
        // accumulated the rank-`effective` outer-product sum with nested scalar loops and an
        // Ops.ToDouble/FromDouble round-trip per element — O(r*C^2) managed work per block per
        // timestep, which kept the diffusion suite past its 10-minute ceiling even after the per-token
        // loop was removed. Two matmuls and a subtract replace all of it.
        //
        // Shapes: Am is [effective, inputDim] with row r pre-scaled by s[r]; Bm is [outputDim,
        // effective]. Bm x Am gives dW as [outputDim, inputDim]; the caller wants it transposed.
        var trainedDown = ScaledRowTensor(DownProjection, SingularValues, effective, inputDim);
        var initialDown = ScaledRowTensor(_downInit, _singularInit, effective, inputDim);
        var trainedUp = LeadingColumnTensor(UpProjection, effective, outputDim);
        var initialUp = LeadingColumnTensor(_upInit, effective, outputDim);

        var deltaWeight = Engine.TensorSubtract(
            Engine.TensorMatMul(trainedUp, trainedDown),
            Engine.TensorMatMul(initialUp, initialDown));

        var delta = Engine.TensorPermute(deltaWeight, new[] { 1, 0 });

        _cachedDelta = delta;
        _cachedTimestep = timestep;
        return delta;
    }

    /// <summary>Builds <c>[effective, inputDim]</c> with row r scaled by <c>singular[r]</c>.</summary>
    private static Tensor<T> ScaledRowTensor(Matrix<T> down, Vector<T> singular, int effective, int inputDim)
    {
        // Leading `effective` rows of A, row r scaled by s[r].
        var rows = Engine.TensorSlice(Tensor<T>.FromMatrix(down), new[] { 0, 0 }, new[] { effective, inputDim });
        var scales = singular.GetSubVector(0, effective).AsTensor().Reshape(new[] { effective, 1 });
        return Engine.TensorMultiply(rows, Engine.TensorBroadcastTo(scales, new[] { effective, inputDim }));
    }

    /// <summary>Builds <c>[outputDim, effective]</c> from the leading columns of <paramref name="up"/>.</summary>
    private static Tensor<T> LeadingColumnTensor(Matrix<T> up, int effective, int outputDim)
    {
        // Leading `effective` columns of B.
        return Engine.TensorSlice(Tensor<T>.FromMatrix(up), new[] { 0, 0 }, new[] { outputDim, effective });
    }

    /// <summary>
    /// Copies the COMPLETE state of <paramref name="source"/> — the trainable A, B and S and the frozen
    /// A_init, B_init and S_init — into this adapter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The frozen triplet has to travel too, and that is easy to miss. The adapter's output is
    /// <c>B S M_t A - B_init S_init M_t A_init</c>, so "untrained" does not mean "zero parameters", it
    /// means the two products CANCEL. Copying only A, B and S into an adapter whose _init came from a
    /// different random draw leaves those products unequal, and the adapter then applies the difference
    /// between two unrelated initializations — a large update, not the identity.
    /// </para>
    /// <para>
    /// That is exactly how a freshly cloned model diverged from its source: identical base weights,
    /// identical A/B/S, and output differing by 69 in absolute terms because each side subtracted its
    /// own initialization.
    /// </para>
    /// </remarks>
    public void CopyStateFrom(TimestepDependentLora<T> source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (source._rank != _rank
            || source.DownProjection.Columns != DownProjection.Columns
            || source.UpProjection.Rows != UpProjection.Rows)
        {
            throw new ArgumentException(
                "Adapter shapes differ, so state cannot be copied: this adapter is rank " +
                $"{_rank} over [{UpProjection.Rows}, {DownProjection.Columns}] and the source is rank " +
                $"{source._rank} over [{source.UpProjection.Rows}, {source.DownProjection.Columns}].",
                nameof(source));
        }

        CopyMatrix(source.DownProjection, DownProjection);
        CopyMatrix(source.UpProjection, UpProjection);
        CopyVector(source.SingularValues, SingularValues);

        CopyMatrix(source._downInit, _downInit);
        CopyMatrix(source._upInit, _upInit);
        CopyVector(source._singularInit, _singularInit);

        InvalidateCache();
    }

    private static void CopyMatrix(Matrix<T> from, Matrix<T> to)
    {
        from.AsSpan().CopyTo(to.AsWritableSpan());
    }

    private static void CopyVector(Vector<T> from, Vector<T> to)
    {
        from.AsSpan().CopyTo(to.AsWritableSpan());
    }

    /// <summary>
    /// Drops the cached delta. Call after writing to <see cref="DownProjection"/>,
    /// <see cref="UpProjection"/> or <see cref="SingularValues"/>.
    /// </summary>
    public void InvalidateCache()
    {
        _cachedDelta = null;
        _cachedTimestep = -1;
    }

    /// <summary>
    /// Applies the adapter to a single vector: <c>(B S M_t A - B_init S_init M_t A_init) x</c>.
    /// </summary>
    /// <remarks>
    /// Kept for single-vector callers and for the unit tests that assert the paper's properties
    /// directly. It goes through the same delta as the batched path, so the two cannot disagree.
    /// </remarks>
    public Vector<T> Apply(Vector<T> input, int timestep)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        // delta is dW^T as [inputDim, outputDim], so the output is input (truncated to inputDim) times delta.
        var delta = EffectiveDeltaTransposed(timestep);
        int inputDim = DownProjection.Columns;
        int outputDim = UpProjection.Rows;
        int used = Math.Min(inputDim, input.Length);
        if (used == 0) return new Vector<T>(outputDim);

        var rows = used == inputDim ? delta : Engine.TensorSlice(delta, new[] { 0, 0 }, new[] { used, outputDim });
        var x = (used == input.Length ? input : input.GetSubVector(0, used)).AsTensor().Reshape(new[] { 1, used });
        return Engine.TensorMatMul(x, rows).Reshape(new[] { outputDim }).ToVector();
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
        // THIN construction. Decomposing a full [outputDim, inputDim] random matrix to keep only its
        // trailing `rank` triplet is O(C^3) — at production SD-XL widths (C up to 1280) that is ~2e9
        // operations PER attention block, and it took the diffusion suite from 14 seconds to 51
        // minutes. The properties the paper actually requires of the initialization are: A with
        // orthonormal rows, B with orthonormal columns, and a small NON-ZERO S. All three are obtained
        // here in O(C * rank^2) instead, ~200x cheaper at those widths.
        //
        // Exact: the orthonormality of A and B, and S being non-zero and small. Distributional: the
        // singular values come from the spectrum of a rank x rank Gaussian draw rather than the tail of
        // a C x C one, so their exact law differs from the paper's while their role — a small, varied,
        // trainable scale per direction — does not. This is an implementation deviation, recorded
        // rather than hidden, and it is the only one in this class.
        double stdDev = 1.0 / Math.Sqrt(rank);

        var down = OrthonormalRows(rank, inputDim, random);
        var upTransposed = OrthonormalRows(rank, outputDim, random);

        // B needs orthonormal COLUMNS, so transpose the orthonormal-rows result.
        var up = Engine.TensorTranspose(Tensor<T>.FromMatrix(upTransposed)).ToMatrix();

        // S from the spectrum of a small rank x rank Gaussian: O(rank^3), and genuinely a spread of
        // singular values rather than a constant, so the per-direction scales differ as they do in the
        // paper. Ascending order, so the leading directions the rank mask keeps carry the smaller
        // scales — matching the paper's use of the TRAILING (least-dominant) triplet.
        var seedMatrix = new Matrix<T>(rank, rank);
        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < rank; j++) seedMatrix[i, j] = Ops.FromDouble(NextGaussian(random) * stdDev);
        }

        var spectrum = new SvdDecomposition<T>(seedMatrix).S;
        // SvdDecomposition orders descending; reverse so index 0 is the smallest.
        var singular = Engine.TensorFlip(spectrum.GetSubVector(0, rank).AsTensor(), new[] { 0 }).ToVector();

        return (down, up, singular);
    }

    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Builds a <c>[rows, columns]</c> matrix with ORTHONORMAL ROWS by modified Gram-Schmidt on
    /// Gaussian rows. O(rows^2 * columns), which for rows = rank is linear in the ambient width.
    /// </summary>
    private static Matrix<T> OrthonormalRows(int rows, int columns, Random random)
    {
        // Seeded Gaussian draws, row-major in master's order, one per element (no engine op consumes a caller
        // Random).
        var gaussian = new Matrix<T>(rows, columns);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++) gaussian[r, c] = Ops.FromDouble(NextGaussian(random));
        }

        // Modified Gram-Schmidt: subtract the projections on the accepted rows one at a time, then normalize.
        // rank <= min(inputDim, outputDim) is enforced in the constructor, so a degenerate row here means
        // numerical collapse; the type-aware epsilon keeps it (near) zero for the orthonormality tests to catch.
        var epsilon = Ops.FromDouble(Math.Pow(2.0, -Ops.PrecisionBits));
        var source = Tensor<T>.FromMatrix(gaussian);
        var basis = new List<Tensor<T>>(rows);
        for (int r = 0; r < rows; r++)
        {
            var row = Engine.TensorSlice(source, new[] { r, 0 }, new[] { 1, columns });
            foreach (var previous in basis)
            {
                var dot = Engine.TensorMatMul(row, Engine.TensorTranspose(previous));
                row = Engine.TensorSubtract(row, Engine.TensorMultiply(previous, Engine.TensorBroadcastTo(dot, new[] { 1, columns })));
            }
            basis.Add(Engine.TensorNormalize(row, 1, epsilon));
        }

        return rows == 0 ? new Matrix<T>(0, columns) : Engine.TensorConcatenate(basis.ToArray(), 0).ToMatrix();
    }
}
