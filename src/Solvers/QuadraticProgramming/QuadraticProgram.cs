using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
/// var featureCount = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var y = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var X = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
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
/// var featureCount = 32;
/// var y = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var X = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var program = new QuadraticProgram&lt;double&gt;(
///     quadratic: X.Transpose().Multiply(X),
///     linear: X.Transpose().Multiply(y).Multiply(-1.0),
///     lowerBounds: new Vector&lt;double&gt;(featureCount));
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

        var numOps = MathHelper.GetNumericOperations<T>();
        double symmetryTolerance = GetSymmetryTolerance(numOps);
        // One row-major pass in the scalar code stopped at the first (i, j) that is non-finite or, above the
        // diagonal, differs from (j, i) by more than the tolerance; a non-finite (j, i) never trips the symmetry test
        // (its scale is infinite or NaN), so the two flags combine without changing which entry is reported first.
        if (quadratic.Rows > 0)
        {
            var engine = SolverTensorOps<T>.Engine;
            var left = SolverTensorOps<T>.AsTensor(quadratic);
            var right = engine.TensorTranspose(left);
            var nonFinite = SolverTensorOps<T>.NonFinite(left);
            var leftMagnitude = engine.TensorAbs(left);
            var rightMagnitude = engine.TensorAbs(right);
            var scale = engine.TensorMax(engine.TensorMax(leftMagnitude, rightMagnitude), numOps.One);
            var asymmetric = engine.TensorGreaterThan(
                engine.TensorAbs(engine.TensorSubtract(left, right)),
                engine.TensorMultiplyScalar(scale, numOps.FromDouble(symmetryTolerance)));
            // Strictly above the diagonal by integer coordinates (indices held in T merge above 2048 for Half).
            var asymmetricAbove = engine.TensorTriu(asymmetric, 1);
            var first = SolverTensorOps<T>.First(SolverTensorOps<T>.Or(nonFinite, asymmetricAbove));
            if (first is not null)
            {
                int i = first[0], j = first[1];
                double entry = numOps.ToDouble(quadratic[i, j]);
                if (double.IsNaN(entry) || double.IsInfinity(entry))
                {
                    throw new ArgumentException(
                        $"The quadratic term entry ({i}, {j}) must be finite.", nameof(quadratic));
                }

                throw new ArgumentException(
                    $"The quadratic term must be symmetric; entries ({i}, {j}) and ({j}, {i}) differ.",
                    nameof(quadratic));
            }
        }

        var linearFirst = SolverTensorOps<T>.First(SolverTensorOps<T>.NonFinite(SolverTensorOps<T>.AsTensor(linear)));
        if (linearFirst is not null)
        {
            throw new ArgumentException($"The linear objective entry {linearFirst[0]} must be finite.", nameof(linear));
        }

        ValidateConstraintBlock(
            inequalityMatrix, inequalityBounds, linear.Length,
            nameof(inequalityMatrix), nameof(inequalityBounds));
        ValidateConstraintBlock(
            equalityMatrix, equalityBounds, linear.Length,
            nameof(equalityMatrix), nameof(equalityBounds));
        ValidateBounds(lowerBounds, linear.Length, nameof(lowerBounds));
        ValidateBounds(upperBounds, linear.Length, nameof(upperBounds));
        ValidateFiniteConstraintBlock(
            inequalityMatrix, inequalityBounds, nameof(inequalityMatrix), nameof(inequalityBounds), numOps);
        ValidateFiniteConstraintBlock(
            equalityMatrix, equalityBounds, nameof(equalityMatrix), nameof(equalityBounds), numOps);
        ValidateFiniteVector(lowerBounds, nameof(lowerBounds), allowInfinity: true, numOps);
        ValidateFiniteVector(upperBounds, nameof(upperBounds), allowInfinity: true, numOps);
        ValidateDirectionalInfinity(lowerBounds, nameof(lowerBounds), rejectPositiveInfinity: true, numOps);
        ValidateDirectionalInfinity(upperBounds, nameof(upperBounds), rejectPositiveInfinity: false, numOps);
        if (lowerBounds is not null && upperBounds is not null && linear.Length > 0)
        {
            var crossed = SolverTensorOps<T>.First(SolverTensorOps<T>.Engine.TensorGreaterThan(
                SolverTensorOps<T>.AsTensor(lowerBounds), SolverTensorOps<T>.AsTensor(upperBounds)));
            if (crossed is not null)
            {
                throw new ArgumentException(
                    $"Lower bound at index {crossed[0]} exceeds the corresponding upper bound.",
                    nameof(lowerBounds));
            }
        }

        Quadratic = quadratic.Clone();
        Linear = linear.Clone();
        InequalityMatrix = inequalityMatrix?.Clone();
        InequalityBounds = inequalityBounds?.Clone();
        EqualityMatrix = equalityMatrix?.Clone();
        EqualityBounds = equalityBounds?.Clone();
        LowerBounds = lowerBounds?.Clone();
        UpperBounds = upperBounds?.Clone();
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

    private static double GetSymmetryTolerance(INumericOperations<T> numOps)
    {
        // Compare in the precision of T, not in an arbitrary double policy. Eight ULPs allows
        // routine operation ordering noise while still rejecting materially asymmetric objectives.
        if (typeof(T) == typeof(float)) return 8.0 * 1.1920928955078125e-7;
        if (typeof(T) == typeof(double)) return 8.0 * 2.2204460492503131e-16;
        if (typeof(T) == typeof(decimal)) return 0.0;

        // Measure the actual representable spacing around one through the numeric abstraction.
        // This gives Half, fixed-point, and user-defined numeric types their own precision policy
        // without teaching the solver an ever-growing list of concrete CLR types.
        T one = numOps.One;
        if (numOps.Compare(numOps.FromDouble(1.5), one) == 0) return 0.0;

        double epsilon = 1.0;
        for (int iteration = 0; iteration < 1074; iteration++)
        {
            double candidate = epsilon / 2.0;
            if (candidate == 0.0 ||
                numOps.Compare(numOps.FromDouble(1.0 + candidate), one) == 0)
            {
                break;
            }

            epsilon = candidate;
        }

        return 8.0 * epsilon;
    }

    private static void ValidateFiniteConstraintBlock(
        Matrix<T>? matrix,
        Vector<T>? bounds,
        string matrixName,
        string boundsName,
        INumericOperations<T> numOps)
    {
        if (matrix is not null && matrix.Rows > 0 && matrix.Columns > 0)
        {
            var first = SolverTensorOps<T>.First(SolverTensorOps<T>.NonFinite(SolverTensorOps<T>.AsTensor(matrix)));
            if (first is not null)
            {
                throw new ArgumentException(
                    $"{matrixName}[{first[0]}, {first[1]}] must be finite.", matrixName);
            }
        }

        ValidateFiniteVector(bounds, boundsName, allowInfinity: false, numOps);
    }

    private static void ValidateFiniteVector(
        Vector<T>? vector,
        string name,
        bool allowInfinity,
        INumericOperations<T> numOps)
    {
        if (vector is null || vector.Length == 0) return;
        var values = SolverTensorOps<T>.AsTensor(vector);
        var first = SolverTensorOps<T>.First(
            allowInfinity ? SolverTensorOps<T>.NaN(values) : SolverTensorOps<T>.NonFinite(values));
        if (first is not null)
        {
            int i = first[0];
            double value = numOps.ToDouble(vector[i]);
            throw new ArgumentException($"{name}[{i}] has an invalid value {value}.", name);
        }
    }

    private static void ValidateDirectionalInfinity(
        Vector<T>? vector,
        string name,
        bool rejectPositiveInfinity,
        INumericOperations<T> numOps)
    {
        if (vector is null || vector.Length == 0) return;
        var values = SolverTensorOps<T>.AsTensor(vector);
        var engine = SolverTensorOps<T>.Engine;
        var first = SolverTensorOps<T>.First(
            rejectPositiveInfinity ? engine.TensorIsPosInf(values) : engine.TensorIsNegInf(values));
        if (first is not null)
        {
            string direction = rejectPositiveInfinity ? "positive" : "negative";
            throw new ArgumentException(
                $"{name}[{first[0]}] cannot be {direction} infinity because it makes the bound infeasible.",
                name);
        }
    }
}
