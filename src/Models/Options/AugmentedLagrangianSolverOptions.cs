namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the augmented Lagrangian solver.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how accurately the solver satisfies
/// constraints and how much work it may spend doing so. The defaults are suitable for most
/// smooth constrained problems.</para>
/// <para><b>Reference:</b>
/// The defaults follow the method of multipliers as presented in J. Nocedal and S. J. Wright,
/// <i>Numerical Optimization</i> (2nd ed., Springer 2006), Framework 17.3 and Algorithm 17.4.
/// </para>
/// </remarks>
public class AugmentedLagrangianSolverOptions : ModelOptions
{
    /// <summary>Initializes the options with documented defaults.</summary>
    public AugmentedLagrangianSolverOptions()
    {
    }

    /// <summary>Initializes the options by copying another configuration.</summary>
    /// <param name="other">The configuration to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public AugmentedLagrangianSolverOptions(AugmentedLagrangianSolverOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        MaxOuterIterations = other.MaxOuterIterations;
        MaxInnerIterations = other.MaxInnerIterations;
        FeasibilityTolerance = other.FeasibilityTolerance;
        StationarityTolerance = other.StationarityTolerance;
        InitialPenalty = other.InitialPenalty;
        PenaltyGrowthFactor = other.PenaltyGrowthFactor;
        MaximumPenalty = other.MaximumPenalty;
        RequiredViolationReduction = other.RequiredViolationReduction;
    }

    /// <summary>
    /// Gets or sets the maximum number of outer iterations.
    /// </summary>
    /// <value>The outer iteration limit, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// One outer iteration is a full unconstrained minimization followed by a multiplier update, so
    /// this is the expensive count — the inner solver runs its own iterations inside each one. The
    /// method typically converges in 10-30 outer iterations, so the default leaves generous room
    /// before giving up.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum number of times the solver may tighten its
    /// handling of the constraints.</para>
    /// </remarks>
    public int MaxOuterIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum number of iterations the inner unconstrained solver may take per
    /// outer iteration.
    /// </summary>
    /// <value>The inner iteration limit, defaulting to 500.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits the work spent improving the candidate point during
    /// each outer constraint update.</para>
    /// </remarks>
    public int MaxInnerIterations { get; set; } = 500;

    /// <summary>
    /// Gets or sets the constraint-violation tolerance that counts as feasible.
    /// </summary>
    /// <value>The feasibility tolerance, defaulting to 1e-8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reaches the constraints from outside rather than
    /// staying inside them, so the answer satisfies them very nearly rather than exactly. This says
    /// how nearly is near enough.
    /// </para>
    /// </remarks>
    public double FeasibilityTolerance { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the tolerance on the augmented Lagrangian's gradient at the inner solution.
    /// </summary>
    /// <value>The stationarity tolerance, defaulting to 1e-8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Smaller values require the objective to be flatter at the final
    /// point, which is more accurate but may take longer.</para>
    /// </remarks>
    public double StationarityTolerance { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the initial penalty weight on constraint violation.
    /// </summary>
    /// <value>The initial penalty, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// Starting too high makes the very first subproblem ill-conditioned for no benefit — the whole
    /// point of the method is that the multipliers, not an ever-growing penalty, are what ultimately
    /// enforce the constraints. Starting too low wastes outer iterations before the penalty bites.
    /// </para>
    /// <para><b>For Beginners:</b> This is how strongly constraint violations matter at the start
    /// of the solve.</para>
    /// </remarks>
    public double InitialPenalty { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the factor by which the penalty grows when an outer iteration fails to reduce
    /// the constraint violation enough.
    /// </summary>
    /// <value>The growth factor, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When constraints are not improving, their penalty is multiplied
    /// by this amount.</para>
    /// </remarks>
    public double PenaltyGrowthFactor { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the ceiling on the penalty weight.
    /// </summary>
    /// <value>The maximum penalty, defaulting to 1e10.</value>
    /// <remarks>
    /// <para>
    /// Past some magnitude the penalty term swamps the objective in floating point and the
    /// subproblem stops being solvable to any useful accuracy. Capping it keeps a genuinely
    /// infeasible problem from degenerating into arithmetic noise rather than reporting honestly
    /// that it could not reach feasibility.
    /// </para>
    /// <para><b>For Beginners:</b> This ceiling prevents an impossible constraint from producing
    /// unusably large numbers.</para>
    /// </remarks>
    public double MaximumPenalty { get; set; } = 1e10;

    /// <summary>
    /// Gets or sets the fraction by which the constraint violation must fall for an outer iteration
    /// to count as progress.
    /// </summary>
    /// <value>The required reduction, defaulting to 0.25.</value>
    /// <remarks>
    /// <para>
    /// An outer iteration that reduces the violation by at least this fraction is taken as evidence
    /// that the current penalty is doing its job, so the multipliers are updated and the penalty is
    /// left alone. Otherwise the penalty is raised instead. This is the test in Nocedal and Wright's
    /// Algorithm 17.4, which trades multiplier updates against penalty growth rather than doing both
    /// every iteration.
    /// </para>
    /// <para><b>For Beginners:</b> This says how much constraint error must shrink before an outer
    /// iteration counts as useful progress.</para>
    /// </remarks>
    public double RequiredViolationReduction { get; set; } = 0.25;
}
