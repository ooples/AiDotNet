namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the simplex linear-programming solver.
/// </summary>
/// <remarks>
/// <para>
/// Every value has a documented default drawn from standard practice, and every one is settable,
/// so no behaviour of the solver is hard-coded out of the caller's reach.
/// </para>
/// <para><b>Reference:</b> Dantzig, <i>Linear Programming and Extensions</i> (1963), with Bland's
/// anti-cycling rule (1977).</para>
/// <para><b>For Beginners:</b> These settings control how accurately and how long the simplex
/// solver searches the corners of a linear problem before returning.</para>
/// </remarks>
public class SimplexSolverOptions : ModelOptions
{
    private int _maxIterations = 10000;
    private double _tolerance = 1e-9;
    private int _degeneratePivotsBeforeBlandsRule = 20;

    public SimplexSolverOptions() { }

    public SimplexSolverOptions(SimplexSolverOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        MaxIterations = other.MaxIterations;
        Tolerance = other.Tolerance;
        DegeneratePivotsBeforeBlandsRule = other.DegeneratePivotsBeforeBlandsRule;
    }

    /// <summary>
    /// Gets or sets the maximum number of simplex pivots across both phases.
    /// </summary>
    /// <value>The pivot limit, defaulting to 10000.</value>
    /// <remarks>
    /// <para>
    /// The simplex method almost always finishes in a small multiple of the number of constraints,
    /// but the worst case is exponential (Klee and Minty, 1972), so a limit is required to
    /// guarantee termination. Reaching it yields
    /// <see cref="Solvers.LinearProgramming.LinearProgramStatus.IterationLimit"/> with the current
    /// feasible point, never a wrong "optimal" answer.
    /// </para>
    /// <para><b>For Beginners:</b> A safety stop. If the solver somehow keeps working far longer
    /// than any real problem should need, it gives up and says so instead of running forever.
    /// </para>
    /// </remarks>
    public int MaxIterations
    {
        get => _maxIterations;
        set => _maxIterations = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "MaxIterations must be positive.");
    }

    /// <summary>
    /// Gets or sets the magnitude below which a tableau entry is treated as zero.
    /// </summary>
    /// <value>The numerical tolerance, defaulting to 1e-9.</value>
    /// <remarks>
    /// <para>
    /// Repeated pivoting accumulates rounding error, so exact comparisons against zero are unsafe:
    /// a coefficient that should be 0 arrives as 1e-17, and a pivot on it would produce enormous
    /// numbers and destroy the tableau. This threshold decides what counts as zero when selecting
    /// entering and leaving variables and when testing feasibility.
    /// </para>
    /// <para><b>For Beginners:</b> Computer arithmetic leaves tiny crumbs of error behind. This
    /// setting says how small a number has to be before the solver treats it as really zero rather
    /// than as a meaningful quantity.
    /// </para>
    /// </remarks>
    public double Tolerance
    {
        get => _tolerance;
        set => _tolerance = value >= 0 && !double.IsNaN(value) && !double.IsInfinity(value)
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Tolerance must be finite and non-negative.");
    }

    /// <summary>
    /// Gets or sets the number of consecutive degenerate pivots tolerated before the solver
    /// switches to Bland's anti-cycling rule.
    /// </summary>
    /// <value>The degenerate-pivot threshold, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// Dantzig's rule (enter on the most negative reduced cost) is fast but can cycle forever on
    /// degenerate problems, revisiting the same set of bases. Bland's rule (enter on the
    /// lowest-index eligible column, and break leaving-variable ties by lowest index) provably
    /// cannot cycle but converges more slowly. This solver uses Dantzig's rule until it sees this
    /// many pivots in a row that fail to improve the objective, then switches to Bland's rule for
    /// the rest of the solve — the standard hybrid, which keeps Dantzig's speed while retaining
    /// Bland's termination guarantee.
    /// </para>
    /// <para><b>For Beginners:</b> The fast pivot rule can very occasionally get the solver walking
    /// in circles. This setting is how long it is allowed to make no progress before falling back
    /// to a slower rule that mathematically cannot loop.
    /// </para>
    /// </remarks>
    public int DegeneratePivotsBeforeBlandsRule
    {
        get => _degeneratePivotsBeforeBlandsRule;
        set => _degeneratePivotsBeforeBlandsRule = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "DegeneratePivotsBeforeBlandsRule must be positive.");
    }
}
