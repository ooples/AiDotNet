using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.QuadraticProgramming;

/// <summary>
/// Describes a convex quadratic program: a quadratic objective minimized subject to linear
/// constraints and bounds.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The problem solved is
/// <code>
///   minimize    ½·xᵀQx + cᵀx
///   subject to  A_ub · x ≤ b_ub
///               A_eq · x = b_eq
///               lower ≤ x ≤ upper
/// </code>
/// <c>Q</c> must be symmetric positive semi-definite, which is what makes the problem convex and
/// therefore exactly solvable: any local minimum is the global minimum.
/// </para>
/// <para>
/// This form covers a surprising amount of machine learning. Ridge regression, the support-vector
/// machine dual, non-negative least squares, mean-variance portfolio selection, the projection step
/// of gradient-episodic-memory continual learning, and every step of a linear model-predictive
/// controller are all quadratic programs.
/// </para>
/// <para><b>For Beginners:</b> A linear program has a flat objective, so its answer is always at a
/// corner. A quadratic program's objective is bowl-shaped, so its answer can sit anywhere — in the
/// middle of the bowl if no constraint interferes, or pressed against whichever constraints do.
/// That bowl shape is exactly what "least squares" means, which is why fitting problems with side
/// conditions land here so often.
/// </para>
/// <example>
/// Non-negative least squares — fit <c>Xw ≈ y</c> with all weights required to be non-negative.
/// Expanding <c>½‖Xw − y‖²</c> gives <c>Q = XᵀX</c> and <c>c = −Xᵀy</c>:
/// <code>
/// var program = new QuadraticProgram&lt;double&gt;(
///     quadratic: X.Transpose().Multiply(X),
///     linear: X.Transpose().Multiply(y).Negate(),
///     lowerBounds: Vector&lt;double&gt;.Zeros(featureCount));
/// </code>
/// </example>
/// </remarks>
public sealed class QuadraticProgram<T>
{
    /// <summary>
    /// Gets the symmetric positive semi-definite matrix <c>Q</c> of the quadratic term
    /// <c>½·xᵀQx</c>.
    /// </summary>
    public Matrix<T> Quadratic { get; }

    /// <summary>Gets the linear objective coefficients <c>c</c>.</summary>
    public Vector<T> Linear { get; }

    /// <summary>
    /// Gets the inequality constraint matrix <c>A_ub</c>, or <c>null</c> when there are none.
    /// </summary>
    public Matrix<T>? InequalityMatrix { get; }

    /// <summary>Gets the inequality right-hand side <c>b_ub</c>, or <c>null</c>.</summary>
    public Vector<T>? InequalityBounds { get; }

    /// <summary>
    /// Gets the equality constraint matrix <c>A_eq</c>, or <c>null</c> when there are none.
    /// </summary>
    public Matrix<T>? EqualityMatrix { get; }

    /// <summary>Gets the equality right-hand side <c>b_eq</c>, or <c>null</c>.</summary>
    public Vector<T>? EqualityBounds { get; }

    /// <summary>
    /// Gets the lower bound per variable, or <c>null</c> to leave every variable unbounded below.
    /// </summary>
    /// <remarks>
    /// Unlike a linear program, the default here is <b>unbounded</b> rather than zero: a quadratic
    /// objective is frequently minimized over all of space, and silently imposing non-negativity
    /// would change the answer without the caller asking.
    /// </remarks>
    public Vector<T>? LowerBounds { get; }

    /// <summary>Gets the upper bound per variable, or <c>null</c> for unbounded above.</summary>
    public Vector<T>? UpperBounds { get; }

    /// <summary>Gets the number of decision variables.</summary>
    public int VariableCount => Linear.Length;

    /// <summary>
    /// Creates a quadratic program.
    /// </summary>
    /// <param name="quadratic">The symmetric positive semi-definite matrix <c>Q</c>.</param>
    /// <param name="linear">The linear objective coefficients <c>c</c>.</param>
    /// <param name="inequalityMatrix">Rows of <c>A_ub</c> for <c>A_ub · x ≤ b_ub</c>.</param>
    /// <param name="inequalityBounds">Right-hand side <c>b_ub</c>.</param>
    /// <param name="equalityMatrix">Rows of <c>A_eq</c> for <c>A_eq · x = b_eq</c>.</param>
    /// <param name="equalityBounds">Right-hand side <c>b_eq</c>.</param>
    /// <param name="lowerBounds">Lower bound per variable; defaults to unbounded below.</param>
    /// <param name="upperBounds">Upper bound per variable; defaults to unbounded above.</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="quadratic"/> or <paramref name="linear"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the objective is empty, <c>Q</c> is not square or does not match the variable
    /// count, a constraint matrix is supplied without its right-hand side (or the reverse), a
    /// constraint matrix's column count does not match the variable count, a right-hand side length
    /// does not match its matrix's row count, or a bounds vector's length is wrong.
    /// </exception>
    public QuadraticProgram(
        Matrix<T> quadratic,
        Vector<T> linear,
        Matrix<T>? inequalityMatrix = null,
        Vector<T>? inequalityBounds = null,
        Matrix<T>? equalityMatrix = null,
        Vector<T>? equalityBounds = null,
        Vector<T>? lowerBounds = null,
        Vector<T>? upperBounds = null)
    {
        if (quadratic is null) throw new ArgumentNullException(nameof(quadratic));
        if (linear is null) throw new ArgumentNullException(nameof(linear));

        if (linear.Length == 0)
        {
            throw new ArgumentException(
                "A quadratic program must have at least one variable.", nameof(linear));
        }

        if (quadratic.Rows != quadratic.Columns)
        {
            throw new ArgumentException(
                $"The quadratic term must be square, but is {quadratic.Rows}x{quadratic.Columns}.",
                nameof(quadratic));
        }

        if (quadratic.Rows != linear.Length)
        {
            throw new ArgumentException(
                $"The quadratic term is {quadratic.Rows}x{quadratic.Columns} but there are " +
                $"{linear.Length} variables.", nameof(quadratic));
        }

        ValidateConstraintBlock(
            inequalityMatrix, inequalityBounds, linear.Length,
            nameof(inequalityMatrix), nameof(inequalityBounds));
        ValidateConstraintBlock(
            equalityMatrix, equalityBounds, linear.Length,
            nameof(equalityMatrix), nameof(equalityBounds));
        ValidateBounds(lowerBounds, linear.Length, nameof(lowerBounds));
        ValidateBounds(upperBounds, linear.Length, nameof(upperBounds));

        Quadratic = quadratic;
        Linear = linear;
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
                $"{boundsName} was supplied without {matrixName}.", boundsName);
        }

        if (bounds is null)
        {
            throw new ArgumentException(
                $"{matrixName} was supplied without {boundsName}.", matrixName);
        }

        if (matrix.Columns != variableCount)
        {
            throw new ArgumentException(
                $"{matrixName} has {matrix.Columns} columns but there are {variableCount} variables.",
                matrixName);
        }

        if (bounds.Length != matrix.Rows)
        {
            throw new ArgumentException(
                $"{boundsName} has {bounds.Length} entries but {matrixName} has {matrix.Rows} rows.",
                boundsName);
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
