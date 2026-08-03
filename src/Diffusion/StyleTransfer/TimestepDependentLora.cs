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
/// noise, where a single training image most easily dictates the output — only the leading few
/// directions of the update survive. At low timesteps, where the model is refining detail, the full
/// rank is available.
/// </para>
/// <code>
///   effective_rank(t) = ceil(R * (1 - t / T))          clamped to at least one direction
///   mask_i(t)         = 1 if i &lt; effective_rank(t), else 0
///   dW(t)             = B * diag(mask(t)) * A
/// </code>
/// <para>
/// <b>Two innovations, and both are here.</b> The rank schedule above, and orthogonal
/// initialization: "a weight parametrization technique that ensures independence between adapter
/// components". Independence is what makes the schedule meaningful — if the adapter's directions
/// were correlated, masking the tail would not actually remove capacity, because the surviving
/// directions would still carry what the masked ones encoded.
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
    private readonly int _totalTimesteps;

    /// <summary>Down-projection A, shape [rank, inputDim], orthogonally initialized.</summary>
    public Matrix<T> DownProjection { get; }

    /// <summary>Up-projection B, shape [outputDim, rank].</summary>
    public Matrix<T> UpProjection { get; }

    /// <summary>Gets the adapter's full rank R, available at timestep zero.</summary>
    public int Rank => _rank;

    /// <summary>Gets the diffusion horizon T used to normalize the timestep.</summary>
    public int TotalTimesteps => _totalTimesteps;

    /// <summary>
    /// Initializes the adapter with orthogonal down-projection rows.
    /// </summary>
    /// <param name="rank">Full rank R.</param>
    /// <param name="inputDim">Width the adapter reads.</param>
    /// <param name="outputDim">Width the adapter writes.</param>
    /// <param name="totalTimesteps">Diffusion horizon T.</param>
    /// <param name="random">RNG for initialization; supply a seeded one for reproducibility.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any dimension is not positive.</exception>
    public TimestepDependentLora(int rank, int inputDim, int outputDim, int totalTimesteps, Random random)
    {
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank), rank, "Rank must be positive.");
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim), inputDim, "Input width must be positive.");
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim), outputDim, "Output width must be positive.");
        if (totalTimesteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalTimesteps), totalTimesteps,
                "The diffusion horizon must be positive; it is the denominator of the rank schedule.");
        if (random is null) throw new ArgumentNullException(nameof(random));

        _rank = rank;
        _totalTimesteps = totalTimesteps;

        DownProjection = OrthogonalRows(rank, inputDim, random);

        // B starts at zero, the standard LoRA convention: the adapter must be the identity before
        // training, or customization would perturb the base model before it has learned anything.
        UpProjection = new Matrix<T>(outputDim, rank);
    }

    /// <summary>
    /// The effective rank at timestep <paramref name="timestep"/>: full at t = 0, shrinking toward
    /// one as t approaches the horizon.
    /// </summary>
    /// <remarks>
    /// Never returns zero. A zero-rank adapter is not "maximally constrained", it is DISCONNECTED —
    /// the update vanishes and the highest timesteps would receive no adaptation at all rather than
    /// a tightly constrained one.
    /// </remarks>
    public int EffectiveRank(int timestep)
    {
        int t = Math.Max(0, Math.Min(timestep, _totalTimesteps));
        double retained = _rank * (1.0 - (double)t / _totalTimesteps);
        return Math.Max(1, Math.Min(_rank, (int)Math.Ceiling(retained)));
    }

    /// <summary>
    /// Applies the adapter at a given timestep: <c>B * diag(mask(t)) * A * x</c>.
    /// </summary>
    public Vector<T> Apply(Vector<T> input, int timestep)
    {
        int effective = EffectiveRank(timestep);
        int inputDim = DownProjection.Columns;
        int outputDim = UpProjection.Rows;

        // Down-project, keeping only the surviving directions.
        var latent = new double[_rank];
        for (int r = 0; r < effective; r++)
        {
            double sum = 0.0;
            for (int i = 0; i < inputDim && i < input.Length; i++)
            {
                sum += Ops.ToDouble(DownProjection[r, i]) * Ops.ToDouble(input[i]);
            }
            latent[r] = sum;
        }

        var output = new Vector<T>(outputDim);
        for (int o = 0; o < outputDim; o++)
        {
            double sum = 0.0;
            for (int r = 0; r < effective; r++) sum += Ops.ToDouble(UpProjection[o, r]) * latent[r];
            output[o] = Ops.FromDouble(sum);
        }
        return output;
    }

    /// <summary>
    /// Builds a matrix whose ROWS are orthonormal, by Gram-Schmidt on Gaussian rows.
    /// </summary>
    /// <remarks>
    /// This is the paper's second innovation, and it is what makes the rank schedule mean anything:
    /// masking the tail of a set of CORRELATED directions does not remove capacity, because the
    /// surviving directions still span what the masked ones encoded. Independence is the
    /// precondition for the constraint to bite.
    /// </remarks>
    private static Matrix<T> OrthogonalRows(int rows, int columns, Random random)
    {
        var basis = new double[rows][];

        for (int r = 0; r < rows; r++)
        {
            var row = new double[columns];
            for (int c = 0; c < columns; c++)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = random.NextDouble();
                row[c] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            }

            // Subtract the projection onto every previously accepted row.
            for (int prev = 0; prev < r; prev++)
            {
                double dot = 0.0;
                for (int c = 0; c < columns; c++) dot += row[c] * basis[prev][c];
                for (int c = 0; c < columns; c++) row[c] -= dot * basis[prev][c];
            }

            double norm = 0.0;
            for (int c = 0; c < columns; c++) norm += row[c] * row[c];
            norm = Math.Sqrt(norm);

            // Rank cannot exceed the ambient width. Beyond that a "new" direction is numerically
            // zero after orthogonalization, so leave it zero rather than normalizing noise into a
            // direction that is not actually independent of the others.
            if (norm > 1e-9)
            {
                for (int c = 0; c < columns; c++) row[c] /= norm;
            }
            basis[r] = row;
        }

        var matrix = new Matrix<T>(rows, columns);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++) matrix[r, c] = Ops.FromDouble(basis[r][c]);
        }
        return matrix;
    }
}
