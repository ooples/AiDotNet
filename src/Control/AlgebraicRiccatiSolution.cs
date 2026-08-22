using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// The result of solving an algebraic Riccati equation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Riccati equation is the centre of linear optimal control: its solution is the quadratic
/// cost-to-go surface, and the optimal feedback gain reads straight off it. Everything an LQR, a
/// Kalman filter or an infinite-horizon MPC needs comes from here.
/// </para>
/// </remarks>
public sealed class AlgebraicRiccatiSolution<T>
{
    /// <summary>
    /// Gets the stabilizing solution <c>P</c>, a symmetric positive-semidefinite matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>xᵀPx</c> is the total remaining cost of driving the state <c>x</c> to the origin optimally.
    /// The Riccati equation has several solutions in general; the one returned is the stabilizing
    /// one, which is the unique positive-semidefinite one and the only one that corresponds to a
    /// closed loop that does not diverge.
    /// </para>
    /// </remarks>
    public Matrix<T> Solution { get; }

    /// <summary>
    /// Gets how far <see cref="Solution"/> is from satisfying the Riccati equation exactly, as the
    /// Frobenius norm of the residual.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is reported rather than merely checked internally because it is the honest measure of
    /// whether the answer can be trusted: an iteration that stopped changing has not necessarily
    /// stopped changing at the right place, and only substituting back into the equation settles it.
    /// </para>
    /// </remarks>
    public T Residual { get; }

    /// <summary>Gets whether the iteration converged within its limits.</summary>
    public bool Converged { get; }

    /// <summary>Gets the number of iterations performed.</summary>
    public int Iterations { get; }

    /// <summary>
    /// Creates a Riccati solution.
    /// </summary>
    /// <param name="solution">The stabilizing solution <c>P</c>.</param>
    /// <param name="residual">The Frobenius norm of the equation's residual.</param>
    /// <param name="converged">Whether the iteration converged.</param>
    /// <param name="iterations">The number of iterations performed.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="solution"/> is null.</exception>
    public AlgebraicRiccatiSolution(
        Matrix<T> solution, T residual, bool converged, int iterations)
    {
        Solution = solution ?? throw new ArgumentNullException(nameof(solution));
        Residual = residual;
        Converged = converged;
        Iterations = iterations;
    }
}
