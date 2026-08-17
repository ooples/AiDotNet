namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the algebraic Riccati equation solvers.
/// </summary>
/// <remarks>
/// <para>
/// Both solvers converge quadratically — the doubling algorithm for the discrete equation and the
/// matrix sign-function iteration for the continuous one — so the iteration limits below are far
/// above what a well-posed problem needs. Reaching one of them is a signal that the problem is
/// ill-posed rather than merely large.
/// </para>
/// <para><b>For Beginners:</b> These settings control how long the solver tries, how close its
/// answer must be, and whether the continuous solver uses an acceleration that is safe by default.
/// A failure to converge usually means the control problem is not well posed, so inspect the model
/// before increasing the iteration limit.</para>
/// <para><b>Reference:</b> J. D. Roberts, "Linear model reduction and solution of the algebraic
/// Riccati equation by use of the sign function", <i>International Journal of Control</i> 32(4),
/// 1980, pp. 677-687; R. Byers, "Solving the algebraic Riccati equation with the matrix sign
/// function", <i>Linear Algebra and its Applications</i> 85, 1987, pp. 267-279. The iteration and
/// tolerance defaults are engineering margins derived from quadratic convergence, not constants
/// prescribed by those papers.</para>
/// </remarks>
public class AlgebraicRiccatiSolverOptions : ModelOptions
{
    /// <summary>Creates solver options with the documented defaults.</summary>
    public AlgebraicRiccatiSolverOptions()
    {
    }

    /// <summary>Creates an independent copy of another Riccati solver configuration.</summary>
    /// <param name="other">The options to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public AlgebraicRiccatiSolverOptions(AlgebraicRiccatiSolverOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        MaxIterations = other.MaxIterations;
        Tolerance = other.Tolerance;
        UseSignFunctionScaling = other.UseSignFunctionScaling;
    }

    /// <summary>
    /// Gets or sets the maximum number of iterations.
    /// </summary>
    /// <value>The iteration limit, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// Quadratic convergence roughly doubles the number of correct digits per step, so a well-posed
    /// problem converges in well under 50 iterations regardless of size. The default leaves room for
    /// a badly scaled one without letting a hopeless one spin.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum number of refinement attempts. Raise it only
    /// after checking that the system is stabilizable and the cost matrices are valid.</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance on the change between successive iterates.
    /// </summary>
    /// <value>The tolerance, defaulting to 1e-12.</value>
    /// <remarks>
    /// <para>
    /// This is tight because quadratic convergence makes it nearly free: once the iteration is
    /// close, one more step gains many digits, so demanding them costs a single iteration rather
    /// than many.
    /// </para>
    /// <para><b>For Beginners:</b> Smaller values demand a more stable answer. The default is already
    /// stringent for double precision and normally needs no adjustment.</para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-12;

    /// <summary>
    /// Gets or sets whether to apply norm-based scaling to the sign-function iteration.
    /// </summary>
    /// <value><c>true</c> by default.</value>
    /// <remarks>
    /// <para>
    /// Newton's iteration for the matrix sign function converges quadratically only once it is
    /// close, and can take many wasted steps getting there when the matrix is badly scaled.
    /// Multiplying each iterate by <c>√(‖Z⁻¹‖ / ‖Z‖)</c> — Byers' scaling — costs nothing and
    /// removes most of that initial phase. It affects only the path taken, never the limit.
    /// </para>
    /// <para>
    /// Applies to the continuous-time solver only; the discrete solver's doubling iteration needs no
    /// scaling.
    /// </para>
    /// <para><b>For Beginners:</b> Keep this enabled. It helps badly scaled continuous-time systems
    /// reach the same answer faster and does not alter the mathematical solution.</para>
    /// </remarks>
    public bool UseSignFunctionScaling { get; set; } = true;
}
