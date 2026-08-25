namespace AiDotNet.Solvers.LinearProgramming;

/// <summary>
/// Solves linear programs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implementations differ in algorithm — the simplex method walks the vertices of the feasible
/// region, while interior-point methods cut through its middle — but agree on this contract, so a
/// caller can swap one for the other without changing anything else.
/// </para>
/// <para><b>For Beginners:</b> You describe the problem (what to minimize, what the limits are) and
/// the solver hands back the best allowed choice, or tells you that no allowed choice exists, or
/// that your limits let the answer run away to infinity.
/// </para>
/// </remarks>
public interface ILinearProgramSolver<T>
{
    /// <summary>
    /// Solves a linear program.
    /// </summary>
    /// <param name="program">The problem to solve.</param>
    /// <returns>
    /// The outcome, the optimal point when one exists, and the dual values of the constraints.
    /// </returns>
    LinearProgramSolution<T> Solve(LinearProgram<T> program);
}
