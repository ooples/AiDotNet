using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.Constrained;

/// <summary>
/// The result of solving a <see cref="ConstrainedProblem{T}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The multipliers are as much of the answer as the point itself: they are the prices of the
/// constraints, and they are what the KKT conditions are stated over. A caller that only reads
/// <see cref="Solution"/> is discarding half of what the solver computed.
/// </para>
/// </remarks>
public sealed class ConstrainedSolution<T>
{
    /// <summary>Gets the outcome of the solve.</summary>
    public LinearProgramStatus Status { get; }

    /// <summary>Gets the best point found.</summary>
    public Vector<T> Solution { get; }

    /// <summary>Gets the objective value at <see cref="Solution"/>.</summary>
    public T ObjectiveValue { get; }

    /// <summary>
    /// Gets the multipliers of the equality constraints, or <c>null</c> when there were none.
    /// </summary>
    public Vector<T>? EqualityMultipliers { get; }

    /// <summary>
    /// Gets the multipliers of the inequality constraints, or <c>null</c> when there were none.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are non-negative, and one is zero exactly when its constraint is not binding —
    /// complementary slackness. A non-zero multiplier tells you what relaxing that limit is worth.
    /// </para>
    /// </remarks>
    public Vector<T>? InequalityMultipliers { get; }

    /// <summary>
    /// Gets the largest amount by which any constraint is violated at <see cref="Solution"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An augmented Lagrangian method drives the iterates toward feasibility rather than maintaining
    /// it, so the returned point satisfies the constraints only to a tolerance. This reports how
    /// closely, which is a number a caller sometimes needs to act on rather than merely trust.
    /// </para>
    /// </remarks>
    public T ConstraintViolation { get; }

    /// <summary>Gets the number of outer iterations performed.</summary>
    public int Iterations { get; }

    /// <summary>
    /// Creates a constrained-problem solution.
    /// </summary>
    /// <param name="status">The outcome of the solve.</param>
    /// <param name="solution">The best point found.</param>
    /// <param name="objectiveValue">The objective value at <paramref name="solution"/>.</param>
    /// <param name="constraintViolation">The largest constraint violation.</param>
    /// <param name="iterations">The number of outer iterations performed.</param>
    /// <param name="equalityMultipliers">The equality constraints' multipliers.</param>
    /// <param name="inequalityMultipliers">The inequality constraints' multipliers.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="solution"/> is null.</exception>
    public ConstrainedSolution(
        LinearProgramStatus status,
        Vector<T> solution,
        T objectiveValue,
        T constraintViolation,
        int iterations,
        Vector<T>? equalityMultipliers = null,
        Vector<T>? inequalityMultipliers = null)
    {
        Status = status;
        Solution = solution ?? throw new ArgumentNullException(nameof(solution));
        ObjectiveValue = objectiveValue;
        ConstraintViolation = constraintViolation;
        Iterations = iterations;
        EqualityMultipliers = equalityMultipliers;
        InequalityMultipliers = inequalityMultipliers;
    }
}
