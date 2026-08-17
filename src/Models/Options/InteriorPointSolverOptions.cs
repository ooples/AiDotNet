namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the primal-dual interior-point solver.
/// </summary>
/// <remarks>
/// <para><b>Reference:</b>
/// The defaults follow Mehrotra (1992) and the implementation notes in Nocedal and Wright,
/// <i>Numerical Optimization</i> (2nd ed.), Chapter 14.
/// </para>
/// <para><b>For Beginners:</b> Every setting here has a working default taken from the papers this
/// solver implements. Change them only when you have a reason to — a specific problem that stalls,
/// or an accuracy requirement the default does not meet.
/// </para>
/// </remarks>
public class InteriorPointSolverOptions : ModelOptions
{
    /// <summary>Initializes the options with documented defaults.</summary>
    public InteriorPointSolverOptions()
    {
    }

    /// <summary>Initializes the options by copying another configuration.</summary>
    /// <param name="other">The configuration to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public InteriorPointSolverOptions(InteriorPointSolverOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        MaxIterations = other.MaxIterations;
        Tolerance = other.Tolerance;
        FractionToBoundary = other.FractionToBoundary;
        Regularization = other.Regularization;
        CertificateTolerance = other.CertificateTolerance;
    }

    /// <summary>
    /// Gets or sets the maximum number of interior-point iterations.
    /// </summary>
    /// <value>The iteration limit, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// Each iteration costs one matrix factorization, which dominates the runtime — but interior-point
    /// methods converge in a number of iterations that grows very slowly with problem size, so
    /// hitting even 50 usually means the problem is ill-conditioned rather than large. The default is
    /// deliberately well above the 20-40 iterations a well-scaled problem takes.
    /// </para>
    /// <para><b>For Beginners:</b> This is a safety ceiling on the number of improvement steps.</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance for the primal residual, dual residual and duality gap.
    /// </summary>
    /// <value>The tolerance, defaulting to 1e-8.</value>
    /// <remarks>
    /// <para>
    /// All three measures are relative — each is divided by <c>1 + ‖·‖</c> of the corresponding
    /// problem data — so this tolerance means the same thing whether the objective is measured in
    /// cents or in millions.
    /// </para>
    /// <para><b>For Beginners:</b> How close to the true optimum the answer must be before the solver
    /// stops. Unlike the simplex method, which lands exactly on a corner, an interior-point method
    /// approaches the optimum from inside and never quite reaches it — this decides how close is
    /// close enough.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets how far along a step the iterate may travel toward the boundary.
    /// </summary>
    /// <value>The fraction-to-boundary parameter, defaulting to 0.99.</value>
    /// <remarks>
    /// <para>
    /// The variables that must stay positive would hit zero at step length one, and the algorithm
    /// breaks down there — the next iteration divides by them. So each step is truncated to this
    /// fraction of the distance to the boundary. Larger values converge faster but risk landing so
    /// close to the boundary that the next factorization is badly conditioned.
    /// </para>
    /// <para><b>For Beginners:</b> "Walk 99% of the way to the wall, never into it."
    /// </para>
    /// </remarks>
    public double FractionToBoundary { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the diagonal regularization added to the normal-equations matrix.
    /// </summary>
    /// <value>The regularization, defaulting to 1e-10.</value>
    /// <remarks>
    /// <para>
    /// Redundant constraint rows — two rows saying the same thing, or a row implied by the others —
    /// make the normal-equations matrix singular, and a redundant row is not a user error: the
    /// standard-form rewrite can introduce one. Adding a small multiple of the identity keeps the
    /// factorization well-defined. It perturbs the step slightly, which the convergence test then
    /// absorbs.
    /// </para>
    /// <para><b>For Beginners:</b> This tiny stabilizer keeps redundant constraints from making
    /// the numerical system impossible to solve.</para>
    /// </remarks>
    public double Regularization { get; set; } = 1e-10;

    /// <summary>
    /// Gets or sets the threshold at which a normalized iterate is accepted as an infeasibility or
    /// unboundedness certificate.
    /// </summary>
    /// <value>The certificate tolerance, defaulting to 1e-7.</value>
    /// <remarks>
    /// <para>
    /// An infeasible-start interior-point method does not stall silently on a problem with no answer
    /// — its iterates diverge along a ray that proves why. A direction <c>y</c> with
    /// <c>Aᵀy ≤ 0</c> and <c>bᵀy &gt; 0</c> proves no feasible point exists (Farkas' lemma), and a
    /// direction <c>x ≥ 0</c> with <c>Ax = 0</c> and <c>cᵀx &lt; 0</c> proves the objective runs to
    /// minus infinity. This tolerance decides how exactly those conditions must hold before the
    /// solver reports the corresponding status.
    /// </para>
    /// <para>
    /// Because the certificate is checked rather than inferred, a reported <c>Infeasible</c> or
    /// <c>Unbounded</c> comes with a witness — the solver never guesses from slow progress.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how convincing a mathematical proof of
    /// infeasibility or unboundedness must be before the solver reports it.</para>
    /// </remarks>
    public double CertificateTolerance { get; set; } = 1e-7;
}
