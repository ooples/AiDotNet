namespace AiDotNet.Solvers.LinearProgramming;

/// <summary>
/// The outcome of attempting to solve a linear program.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> "No answer" comes in two very different flavours, and telling them
/// apart is the first step in fixing a model that did not solve.
/// </para>
/// </remarks>
public enum LinearProgramStatus
{
    /// <summary>
    /// An optimal solution was found. No feasible point has a better objective value.
    /// </summary>
    Optimal,

    /// <summary>
    /// The constraints contradict each other, so no point satisfies all of them at once.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> You asked for something impossible — for example, spend at least
    /// 100 and at most 50. The fix is to relax or correct a constraint, not to try harder.
    /// </para>
    /// </remarks>
    Infeasible,

    /// <summary>
    /// The objective can be improved without limit inside the feasible region.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model says infinite profit is achievable, which in practice
    /// always means a real-world limit was left out — a capacity, a budget, or an upper bound on a
    /// variable.
    /// </para>
    /// </remarks>
    Unbounded,

    /// <summary>
    /// The iteration limit was reached before the search proved optimality. The returned point is
    /// feasible but not certified optimal.
    /// </summary>
    IterationLimit,
}
