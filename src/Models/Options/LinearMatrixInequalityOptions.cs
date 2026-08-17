namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the linear matrix inequality solver.
/// </summary>
public class LinearMatrixInequalityOptions
{
    /// <summary>
    /// Gets or sets the maximum number of subgradient iterations.
    /// </summary>
    /// <value>The iteration limit, defaulting to 5000.</value>
    /// <remarks>
    /// <para>
    /// Subgradient methods converge slowly — the error falls like one over the square root of the
    /// iteration count, not geometrically — so the limit is high by the standards of the other
    /// solvers here. Each iteration is cheap: one matrix assembly and one eigenvector.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 5000;

    /// <summary>
    /// Gets or sets how strictly inside the feasible set the search aims to land.
    /// </summary>
    /// <value>The margin, defaulting to 1e-8.</value>
    /// <remarks>
    /// <para>
    /// The search stops once the smallest eigenvalue reaches this value, rather than zero. Stopping
    /// exactly at zero would return a point on the boundary, where rounding alone can push the matrix
    /// indefinite — so the answer would satisfy the inequality in the solver and violate it in the
    /// caller. Raise this when the result must survive modelling error as well as arithmetic.
    /// </para>
    /// </remarks>
    public double Margin { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the initial step length for the subgradient iteration.
    /// </summary>
    /// <value>The initial step, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Steps shrink as one over the square root of the iteration count, which is the classical
    /// schedule that guarantees convergence for a convex nonsmooth objective (Shor, 1985). This sets
    /// the scale. Too small and the search cannot cross the distance to the feasible set within the
    /// iteration limit; too large and it spends its early iterations overshooting.
    /// </para>
    /// </remarks>
    public double InitialStepSize { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of power iterations used to extract each eigenvector.
    /// </summary>
    /// <value>The iteration count, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// The subgradient of the largest eigenvalue is built from its eigenvector, which is obtained by
    /// power iteration on a shifted matrix. A rough eigenvector is enough early on — the step is
    /// approximate anyway — so this trades accuracy per step against the number of steps.
    /// </para>
    /// </remarks>
    public int PowerIterations { get; set; } = 100;
}
