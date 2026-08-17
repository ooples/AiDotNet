using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Shape checks the Riccati solvers share.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A control problem is described by four matrices whose dimensions must agree in a specific way,
/// and getting one of them transposed is the most common way to state one wrongly. Checking up front
/// turns that into a message naming the offending matrix instead of an index-out-of-range thrown
/// several matrix products later.
/// </para>
/// </remarks>
internal static class RiccatiValidation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Validates the four matrices of a linear-quadratic problem against each other.
    /// </summary>
    /// <exception cref="ArgumentNullException">Thrown when any matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions are inconsistent.</exception>
    public static void Validate(
        Matrix<T> stateMatrix, Matrix<T> inputMatrix, Matrix<T> stateCost, Matrix<T> inputCost)
    {
        if (stateMatrix is null) throw new ArgumentNullException(nameof(stateMatrix));
        if (inputMatrix is null) throw new ArgumentNullException(nameof(inputMatrix));
        if (stateCost is null) throw new ArgumentNullException(nameof(stateCost));
        if (inputCost is null) throw new ArgumentNullException(nameof(inputCost));

        if (stateMatrix.Rows != stateMatrix.Columns)
        {
            throw new ArgumentException(
                $"The state matrix A must be square; it is {stateMatrix.Rows}-by-" +
                $"{stateMatrix.Columns}.", nameof(stateMatrix));
        }

        int stateCount = stateMatrix.Rows;
        if (stateCount == 0)
        {
            throw new ArgumentException(
                "The system must have at least one state.", nameof(stateMatrix));
        }

        if (inputMatrix.Rows != stateCount)
        {
            throw new ArgumentException(
                $"The input matrix B must have one row per state: expected {stateCount} rows to " +
                $"match A, but it has {inputMatrix.Rows}.", nameof(inputMatrix));
        }

        int inputCount = inputMatrix.Columns;
        if (inputCount == 0)
        {
            throw new ArgumentException(
                "The system must have at least one input; with no inputs there is nothing to " +
                "control.", nameof(inputMatrix));
        }

        if (stateCost.Rows != stateCount || stateCost.Columns != stateCount)
        {
            throw new ArgumentException(
                $"The state cost Q must be {stateCount}-by-{stateCount} to match A; it is " +
                $"{stateCost.Rows}-by-{stateCost.Columns}.", nameof(stateCost));
        }

        if (inputCost.Rows != inputCount || inputCost.Columns != inputCount)
        {
            throw new ArgumentException(
                $"The input cost R must be {inputCount}-by-{inputCount} to match the columns of B; " +
                $"it is {inputCost.Rows}-by-{inputCost.Columns}.", nameof(inputCost));
        }

        RequireSymmetric(stateCost, "The state cost Q", nameof(stateCost));
        RequireSymmetric(inputCost, "The input cost R", nameof(inputCost));
    }

    private static void RequireSymmetric(
        Matrix<T> matrix, string description, string parameterName)
    {
        double tolerance = typeof(T) == typeof(float) ? 1e-5 :
            typeof(T) == typeof(double) ? 1e-12 : 0.0;

        for (int row = 0; row < matrix.Rows; row++)
        {
            for (int column = row + 1; column < matrix.Columns; column++)
            {
                double left = NumOps.ToDouble(matrix[row, column]);
                double right = NumOps.ToDouble(matrix[column, row]);
                double difference = Math.Abs(left - right);
                double scale = Math.Max(1.0, Math.Max(Math.Abs(left), Math.Abs(right)));

                if (double.IsNaN(difference) || double.IsInfinity(difference) ||
                    difference > tolerance * scale)
                {
                    throw new ArgumentException(
                        $"{description} must be symmetric; entry ({row}, {column}) is " +
                        $"{matrix[row, column]} while entry ({column}, {row}) is " +
                        $"{matrix[column, row]}.", parameterName);
                }
            }
        }
    }
}
