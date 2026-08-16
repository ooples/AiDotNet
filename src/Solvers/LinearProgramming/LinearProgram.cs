using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.LinearProgramming;

/// <summary>
/// Describes a linear program: a linear objective minimized subject to linear constraints and
/// bounds on the variables.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The problem solved is
/// <code>
///   minimize    cᵀx
///   subject to  A_ub · x ≤ b_ub
///               A_eq · x = b_eq
///               lower ≤ x ≤ upper
/// </code>
/// Any of the constraint blocks may be omitted. When no bounds are supplied the variables are
/// taken to be non-negative (<c>lower = 0</c>, <c>upper = +∞</c>), which is the convention used by
/// every standard linear-programming interface.
/// </para>
/// <para>
/// Maximization is expressed by negating the objective: maximizing <c>pᵀx</c> is minimizing
/// <c>(−p)ᵀx</c>, and the reported objective value is then the negative of the maximum.
/// </para>
/// <para><b>For Beginners:</b> A linear program is the mathematical form of "get the most out of
/// limited resources". Every quantity you can choose is a variable; the thing you want to make as
/// large or small as possible is the objective; and the limits you cannot exceed are the
/// constraints. Scheduling staff, blending materials, routing deliveries and allocating a budget
/// are all linear programs.
/// </para>
/// <para>
/// The defining feature is that everything is <i>linear</i> — doubling a variable doubles its
/// contribution, and there are no products of variables. That restriction is what makes these
/// problems solvable exactly and quickly, even with thousands of variables.
/// </para>
/// <example>
/// A factory makes tables (profit 30) and chairs (profit 20). A table needs 4 hours of carpentry
/// and 2 of finishing; a chair needs 3 and 1. There are 240 carpentry hours and 100 finishing
/// hours available.
/// <code>
/// // Maximize 30·tables + 20·chairs, so minimize the negated objective.
/// var objective = Vector&lt;double&gt;.FromArray(new[] { -30.0, -20.0 });
///
/// var usage = new Matrix&lt;double&gt;(2, 2);
/// usage[0, 0] = 4; usage[0, 1] = 3;   // carpentry hours
/// usage[1, 0] = 2; usage[1, 1] = 1;   // finishing hours
/// var available = Vector&lt;double&gt;.FromArray(new[] { 240.0, 100.0 });
///
/// var program = new LinearProgram&lt;double&gt;(objective, inequalityMatrix: usage, inequalityBounds: available);
/// </code>
/// </example>
/// </remarks>
public sealed class LinearProgram<T>
{
    /// <summary>
    /// Gets the objective coefficients <c>c</c>. The solver minimizes <c>cᵀx</c>.
    /// </summary>
    public Vector<T> Objective { get; }

    /// <summary>
    /// Gets the inequality constraint matrix <c>A_ub</c>, or <c>null</c> when there are none.
    /// </summary>
    public Matrix<T>? InequalityMatrix { get; }

    /// <summary>
    /// Gets the inequality right-hand side <c>b_ub</c>, or <c>null</c> when there are none.
    /// </summary>
    public Vector<T>? InequalityBounds { get; }

    /// <summary>
    /// Gets the equality constraint matrix <c>A_eq</c>, or <c>null</c> when there are none.
    /// </summary>
    public Matrix<T>? EqualityMatrix { get; }

    /// <summary>
    /// Gets the equality right-hand side <c>b_eq</c>, or <c>null</c> when there are none.
    /// </summary>
    public Vector<T>? EqualityBounds { get; }

    /// <summary>
    /// Gets the lower bound on each variable, or <c>null</c> to mean zero for every variable.
    /// A component may be negative infinity to leave that variable unbounded below.
    /// </summary>
    public Vector<T>? LowerBounds { get; }

    /// <summary>
    /// Gets the upper bound on each variable, or <c>null</c> to mean positive infinity for every
    /// variable.
    /// </summary>
    public Vector<T>? UpperBounds { get; }

    /// <summary>Gets the number of decision variables.</summary>
    public int VariableCount => Objective.Length;

    /// <summary>
    /// Creates a linear program.
    /// </summary>
    /// <param name="objective">Objective coefficients <c>c</c>; the solver minimizes <c>cᵀx</c>.</param>
    /// <param name="inequalityMatrix">Rows of <c>A_ub</c> for the constraints <c>A_ub · x ≤ b_ub</c>.</param>
    /// <param name="inequalityBounds">Right-hand side <c>b_ub</c>.</param>
    /// <param name="equalityMatrix">Rows of <c>A_eq</c> for the constraints <c>A_eq · x = b_eq</c>.</param>
    /// <param name="equalityBounds">Right-hand side <c>b_eq</c>.</param>
    /// <param name="lowerBounds">Lower bound per variable; defaults to zero for every variable.</param>
    /// <param name="upperBounds">Upper bound per variable; defaults to positive infinity.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="objective"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the objective is empty, when a constraint matrix is supplied without its
    /// right-hand side (or the reverse), when a constraint matrix's column count does not match the
    /// number of variables, when a right-hand side length does not match its matrix's row count, or
    /// when a bounds vector's length does not match the number of variables.
    /// </exception>
    public LinearProgram(
        Vector<T> objective,
        Matrix<T>? inequalityMatrix = null,
        Vector<T>? inequalityBounds = null,
        Matrix<T>? equalityMatrix = null,
        Vector<T>? equalityBounds = null,
        Vector<T>? lowerBounds = null,
        Vector<T>? upperBounds = null)
    {
        if (objective is null) throw new ArgumentNullException(nameof(objective));
        if (objective.Length == 0)
        {
            throw new ArgumentException(
                "A linear program must have at least one variable.", nameof(objective));
        }

        ValidateConstraintBlock(
            inequalityMatrix, inequalityBounds, objective.Length,
            nameof(inequalityMatrix), nameof(inequalityBounds));
        ValidateConstraintBlock(
            equalityMatrix, equalityBounds, objective.Length,
            nameof(equalityMatrix), nameof(equalityBounds));
        ValidateBounds(lowerBounds, objective.Length, nameof(lowerBounds));
        ValidateBounds(upperBounds, objective.Length, nameof(upperBounds));

        Objective = objective;
        InequalityMatrix = inequalityMatrix;
        InequalityBounds = inequalityBounds;
        EqualityMatrix = equalityMatrix;
        EqualityBounds = equalityBounds;
        LowerBounds = lowerBounds;
        UpperBounds = upperBounds;
    }

    private static void ValidateConstraintBlock(
        Matrix<T>? matrix, Vector<T>? bounds, int variableCount, string matrixName, string boundsName)
    {
        if (matrix is null && bounds is null) return;

        if (matrix is null)
        {
            throw new ArgumentException(
                $"{boundsName} was supplied without {matrixName}. A right-hand side is meaningless " +
                "without the constraint rows it belongs to.", boundsName);
        }

        if (bounds is null)
        {
            throw new ArgumentException(
                $"{matrixName} was supplied without {boundsName}. Constraint rows are meaningless " +
                "without a right-hand side.", matrixName);
        }

        if (matrix.Columns != variableCount)
        {
            throw new ArgumentException(
                $"{matrixName} has {matrix.Columns} columns but the objective has {variableCount} " +
                "variables. Every constraint must give a coefficient for every variable.", matrixName);
        }

        if (bounds.Length != matrix.Rows)
        {
            throw new ArgumentException(
                $"{boundsName} has {bounds.Length} entries but {matrixName} has {matrix.Rows} rows. " +
                "Every constraint row needs exactly one right-hand side value.", boundsName);
        }
    }

    private static void ValidateBounds(Vector<T>? bounds, int variableCount, string name)
    {
        if (bounds is not null && bounds.Length != variableCount)
        {
            throw new ArgumentException(
                $"{name} has {bounds.Length} entries but there are {variableCount} variables.", name);
        }
    }
}
