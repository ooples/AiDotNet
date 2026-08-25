using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.LinearProgramming;

/// <summary>
/// The result of solving a <see cref="LinearProgram{T}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// All state is supplied through the constructor and exposed read-only, so a solution cannot be
/// altered after the solver produced it.
/// </para>
/// <para><b>For Beginners:</b> Besides the answer itself, a linear program tells you something
/// most solvers throw away: how much each limit is costing you. See
/// <see cref="InequalityDualValues"/> — those are the numbers that answer "which constraint should
/// I relax first?".
/// </para>
/// </remarks>
public sealed class LinearProgramSolution<T>
{
    /// <summary>Gets the outcome of the solve.</summary>
    public LinearProgramStatus Status { get; }

    /// <summary>
    /// Gets the decision variables at the returned point, or <c>null</c> when the problem was
    /// infeasible or unbounded.
    /// </summary>
    public Vector<T>? Solution { get; }

    /// <summary>
    /// Gets the objective value <c>cᵀx</c> at <see cref="Solution"/>. Only meaningful when
    /// <see cref="Status"/> is <see cref="LinearProgramStatus.Optimal"/> or
    /// <see cref="LinearProgramStatus.IterationLimit"/>.
    /// </summary>
    public T ObjectiveValue { get; }

    /// <summary>
    /// Gets the dual value (shadow price) of each inequality constraint, or <c>null</c> when the
    /// problem had none or was not solved to optimality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The dual value of a constraint is the rate at which the optimal objective improves per unit
    /// of relaxation of that constraint's right-hand side. A dual of zero means the constraint is
    /// slack — it is not currently limiting anything.
    /// </para>
    /// <para><b>For Beginners:</b> If a shift-hours constraint has a dual value of 12, then one
    /// extra hour of that shift is worth 12 to you — so it is worth buying up to that price, and
    /// no more. This is the single most useful by-product of solving a linear program, and it is
    /// why duality (Chapter 3.5 of the optimization literature) gets so much attention.
    /// </para>
    /// </remarks>
    public Vector<T>? InequalityDualValues { get; }

    /// <summary>
    /// Gets the dual value of each equality constraint, or <c>null</c> when the problem had none
    /// or was not solved to optimality.
    /// </summary>
    public Vector<T>? EqualityDualValues { get; }

    /// <summary>Gets the number of simplex pivots performed across both phases.</summary>
    public int Iterations { get; }

    /// <summary>
    /// Creates a linear program solution.
    /// </summary>
    /// <param name="status">The outcome of the solve.</param>
    /// <param name="solution">The decision variables, or null when there is no point to report.</param>
    /// <param name="objectiveValue">The objective value at <paramref name="solution"/>.</param>
    /// <param name="iterations">Number of simplex pivots performed.</param>
    /// <param name="inequalityDualValues">Shadow prices of the inequality constraints.</param>
    /// <param name="equalityDualValues">Shadow prices of the equality constraints.</param>
    public LinearProgramSolution(
        LinearProgramStatus status,
        Vector<T>? solution,
        T objectiveValue,
        int iterations,
        Vector<T>? inequalityDualValues = null,
        Vector<T>? equalityDualValues = null)
    {
        Status = status;
        Solution = solution;
        ObjectiveValue = objectiveValue;
        Iterations = iterations;
        InequalityDualValues = inequalityDualValues;
        EqualityDualValues = equalityDualValues;
    }
}
