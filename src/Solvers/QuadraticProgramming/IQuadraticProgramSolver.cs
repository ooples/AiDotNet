namespace AiDotNet.Solvers.QuadraticProgramming;

/// <summary>
/// Solves convex quadratic programs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implementations differ in algorithm — active-set methods identify which constraints bind and
/// solve an equality-constrained problem, interior-point methods approach the boundary from inside
/// — but agree on this contract, so callers can swap one for the other.
/// </para>
/// </remarks>
public interface IQuadraticProgramSolver<T>
{
    /// <summary>
    /// Solves a convex quadratic program.
    /// </summary>
    /// <param name="program">The problem to solve.</param>
    /// <returns>The outcome, the optimal point when one exists, and the Lagrange multipliers.</returns>
    QuadraticProgramSolution<T> Solve(QuadraticProgram<T> program);
}
