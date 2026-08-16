namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the active-set quadratic-programming solver.
/// </summary>
/// <remarks>
/// <para><b>Reference:</b> Nocedal and Wright, <i>Numerical Optimization</i>, 2nd ed.,
/// Algorithm 16.3.</para>
/// <para><b>For Beginners:</b> These settings control a solver that keeps track of which limits
/// currently touch the answer, then solves the smaller problem formed by those active limits.</para>
/// </remarks>
public class ActiveSetQuadraticProgramSolverOptions : ModelOptions
{
    public ActiveSetQuadraticProgramSolverOptions() { }

    public ActiveSetQuadraticProgramSolverOptions(ActiveSetQuadraticProgramSolverOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        MaxIterations = other.MaxIterations;
        Tolerance = other.Tolerance;
        SingularityRegularization = other.SingularityRegularization;
        FeasibilityOptions = new SimplexSolverOptions(other.FeasibilityOptions);
    }

    /// <summary>
    /// Gets or sets the maximum number of active-set iterations.
    /// </summary>
    /// <value>The iteration limit, defaulting to 200.</value>
    /// <remarks>
    /// <para>
    /// An active-set method terminates finitely because each iteration either adds or drops a
    /// constraint and the objective never increases, but the number of working sets is combinatorial
    /// in principle. This bound guarantees termination; reaching it returns the current feasible
    /// point flagged as not-certified rather than a false claim of optimality.
    /// </para>
    /// <para><b>For Beginners:</b> A safety stop, so an unusual problem cannot hang the caller.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 200;

    /// <summary>
    /// Gets or sets the magnitude below which a quantity is treated as zero.
    /// </summary>
    /// <value>The numerical tolerance, defaulting to 1e-9.</value>
    /// <remarks>
    /// <para>
    /// Used to decide when a search direction is effectively zero (so the current point solves the
    /// equality-constrained subproblem), when a constraint counts as active, and when a Lagrange
    /// multiplier counts as negative.
    /// </para>
    /// <para><b>For Beginners:</b> Values smaller than this are treated as floating-point noise
    /// rather than meaningful violations.</para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-9;

    /// <summary>
    /// Gets or sets the amount added to the diagonal of <c>Q</c> when the KKT system is singular.
    /// </summary>
    /// <value>The regularization amount, defaulting to 1e-10.</value>
    /// <remarks>
    /// <para>
    /// Convexity only requires <c>Q</c> to be positive <i>semi</i>-definite, and a singular <c>Q</c>
    /// — which arises whenever the quadratic term is <c>XᵀX</c> with fewer rows than columns, as in
    /// non-negative least squares on wide data — makes the KKT system singular along the null
    /// directions. Adding a tiny multiple of the identity restores a unique solve while shifting the
    /// answer by an amount on the order of the regularization itself.
    /// </para>
    /// <para><b>For Beginners:</b> If the bowl is perfectly flat in some direction, there is no
    /// single lowest point along it and the arithmetic has nothing to latch onto. This adds an
    /// almost imperceptible tilt so a unique answer exists.
    /// </para>
    /// </remarks>
    public double SingularityRegularization { get; set; } = 1e-10;

    /// <summary>
    /// Gets or sets the options used for the linear program that finds an initial feasible point.
    /// </summary>
    /// <value>A separately owned copy of the feasibility solver settings.</value>
    /// <remarks>
    /// <para>
    /// An active-set method has to start from a point satisfying every constraint. Finding one is
    /// itself a linear-programming feasibility problem, solved with the simplex method before the
    /// quadratic phase begins.
    /// </para>
    /// <para><b>For Beginners:</b> Before optimizing, the solver first needs to find any point that
    /// obeys all limits. These settings control that preparatory search.</para>
    /// </remarks>
    public SimplexSolverOptions FeasibilityOptions { get; set; } = new();
}
