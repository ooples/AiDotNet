using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.QuadraticProgramming;

/// <summary>
/// The result of solving a <see cref="QuadraticProgram{T}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The status shares the <see cref="LinearProgramStatus"/> vocabulary, because the possible
/// outcomes are the same: an optimum, contradictory constraints, an objective that runs away, or a
/// search stopped by its iteration budget.
/// </para>
/// </remarks>
public sealed class QuadraticProgramSolution<T>
{
    /// <summary>Gets the outcome of the solve.</summary>
    public LinearProgramStatus Status { get; }

    /// <summary>
    /// Gets the decision variables at the returned point, or <c>null</c> when the problem was
    /// infeasible or unbounded.
    /// </summary>
    public Vector<T>? Solution { get; }

    /// <summary>Gets the objective value <c>½·xᵀQx + cᵀx</c> at <see cref="Solution"/>.</summary>
    public T ObjectiveValue { get; }

    /// <summary>
    /// Gets the Lagrange multiplier of each inequality constraint, or <c>null</c> when the problem
    /// had none.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A multiplier is zero for any constraint that is not active at the optimum, and non-negative
    /// for those that are — the complementary-slackness half of the KKT conditions.
    /// </para>
    /// <para><b>For Beginners:</b> The same "what is this limit worth?" reading as a linear
    /// program's shadow prices. A zero means that limit is not currently binding you.
    /// </para>
    /// </remarks>
    public Vector<T>? InequalityMultipliers { get; }

    /// <summary>
    /// Gets the Lagrange multiplier of each equality constraint, or <c>null</c> when there were
    /// none. Equality multipliers may take either sign.
    /// </summary>
    public Vector<T>? EqualityMultipliers { get; }

    /// <summary>Gets the number of active-set iterations performed.</summary>
    public int Iterations { get; }

    /// <summary>
    /// Creates a quadratic program solution.
    /// </summary>
    /// <param name="status">The outcome of the solve.</param>
    /// <param name="solution">The decision variables, or null when there is no point to report.</param>
    /// <param name="objectiveValue">The objective value at <paramref name="solution"/>.</param>
    /// <param name="iterations">Number of active-set iterations performed.</param>
    /// <param name="inequalityMultipliers">Multipliers of the inequality constraints.</param>
    /// <param name="equalityMultipliers">Multipliers of the equality constraints.</param>
    public QuadraticProgramSolution(
        LinearProgramStatus status,
        Vector<T>? solution,
        T objectiveValue,
        int iterations,
        Vector<T>? inequalityMultipliers = null,
        Vector<T>? equalityMultipliers = null)
    {
        Status = status;
        Solution = solution;
        ObjectiveValue = objectiveValue;
        Iterations = iterations;
        InequalityMultipliers = inequalityMultipliers;
        EqualityMultipliers = equalityMultipliers;
    }
}
