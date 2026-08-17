namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the linear matrix inequality solver.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how long the solver searches, how large its
/// first move is, and how carefully it estimates the most troublesome matrix direction. The
/// defaults are safe for ordinary control problems.</para>
/// <para><b>Reference:</b> N. Z. Shor, <i>Minimization Methods for Non-Differentiable Functions</i>,
/// Springer, 1985; S. Boyd et al., <i>Linear Matrix Inequalities in System and Control Theory</i>,
/// SIAM, 1994.</para>
/// </remarks>
public class LinearMatrixInequalityOptions : ModelOptions
{
    private int _maxIterations = 5000;
    private double _margin = 1e-8;
    private double _initialStepSize = 1.0;
    private int _powerIterations = 100;

    /// <summary>Creates options with the documented defaults.</summary>
    public LinearMatrixInequalityOptions()
    {
    }

    /// <summary>Creates an independent copy of another LMI solver configuration.</summary>
    /// <param name="other">The options to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public LinearMatrixInequalityOptions(LinearMatrixInequalityOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        MaxIterations = other.MaxIterations;
        Margin = other.Margin;
        InitialStepSize = other.InitialStepSize;
        PowerIterations = other.PowerIterations;
    }

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
    /// <para><b>For Beginners:</b> This is a safety stop. Increase it only when the reported
    /// smallest eigenvalue is still improving at the limit.</para>
    /// </remarks>
    public int MaxIterations
    {
        get => _maxIterations;
        set => _maxIterations = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "MaxIterations must be positive.");
    }

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
    /// <para><b>For Beginners:</b> A small margin keeps the answer safely inside the allowed region
    /// instead of balancing exactly on its edge.</para>
    /// </remarks>
    public double Margin
    {
        get => _margin;
        set => _margin = value >= 0.0 && !double.IsNaN(value) && !double.IsInfinity(value)
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "Margin must be finite and non-negative.");
    }

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
    /// <para><b>For Beginners:</b> This controls the first move's size. Later moves automatically
    /// become smaller as the answer is refined.</para>
    /// </remarks>
    public double InitialStepSize
    {
        get => _initialStepSize;
        set => _initialStepSize = value > 0.0 && !double.IsNaN(value) && !double.IsInfinity(value)
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "InitialStepSize must be finite and positive.");
    }

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
    /// <para><b>For Beginners:</b> More iterations estimate the most troublesome direction more
    /// carefully, but make each search step more expensive.</para>
    /// </remarks>
    public int PowerIterations
    {
        get => _powerIterations;
        set => _powerIterations = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "PowerIterations must be positive.");
    }
}
