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
        RequirePositiveSemidefinite(stateCost, "The state cost Q", nameof(stateCost));
        RequirePositiveDefinite(inputCost, "The input cost R", nameof(inputCost));
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

    private static void RequirePositiveSemidefinite(
        Matrix<T> matrix, string description, string parameterName)
    {
        (double smallestEigenvalue, double tolerance) =
            GetSmallestSymmetricEigenvalue(matrix, parameterName);
        if (smallestEigenvalue < -tolerance)
        {
            throw new ArgumentException(
                $"{description} must be positive-semidefinite; its smallest eigenvalue is " +
                $"{smallestEigenvalue:G17}.", parameterName);
        }
    }

    private static void RequirePositiveDefinite(
        Matrix<T> matrix, string description, string parameterName)
    {
        (double smallestEigenvalue, double tolerance) =
            GetSmallestSymmetricEigenvalue(matrix, parameterName);
        if (smallestEigenvalue <= tolerance)
        {
            throw new ArgumentException(
                $"{description} must be positive-definite; its smallest eigenvalue is " +
                $"{smallestEigenvalue:G17}.", parameterName);
        }
    }

    /// <summary>
    /// Finds the smallest eigenvalue of a real symmetric matrix with the Jacobi algorithm.
    /// </summary>
    /// <remarks>
    /// Cost matrices are normally tiny, so a dependency-free O(n^3) symmetric eigensolve is both
    /// cheaper and more reliable here than using inversion as a proxy for definiteness. Averaging
    /// mirrored entries also keeps the result symmetric after the tolerance-aware symmetry check.
    /// </remarks>
    private static (double SmallestEigenvalue, double Tolerance) GetSmallestSymmetricEigenvalue(
        Matrix<T> matrix, string parameterName)
    {
        int size = matrix.Rows;
        var values = new double[size, size];
        double scale = 0.0;

        for (int row = 0; row < size; row++)
        {
            for (int column = row; column < size; column++)
            {
                double left = NumOps.ToDouble(matrix[row, column]);
                double right = NumOps.ToDouble(matrix[column, row]);
                if (double.IsNaN(left) || double.IsInfinity(left) ||
                    double.IsNaN(right) || double.IsInfinity(right))
                {
                    throw new ArgumentException(
                        "Cost matrices must contain only finite values.", parameterName);
                }

                double value = 0.5 * (left + right);
                values[row, column] = value;
                values[column, row] = value;
                scale = Math.Max(scale, Math.Abs(value));
            }
        }

        double typeEpsilon = typeof(T) == typeof(float)
            ? 1.1920928955078125e-7
            : 2.2204460492503131e-16;
        double tolerance = 32.0 * typeEpsilon * scale * Math.Max(1, size);
        int maximumRotations = Math.Max(1, 50 * size * size);

        for (int rotation = 0; rotation < maximumRotations; rotation++)
        {
            int pivotRow = 0;
            int pivotColumn = 0;
            double largestOffDiagonal = 0.0;
            for (int row = 0; row < size; row++)
            {
                for (int column = row + 1; column < size; column++)
                {
                    double magnitude = Math.Abs(values[row, column]);
                    if (magnitude > largestOffDiagonal)
                    {
                        largestOffDiagonal = magnitude;
                        pivotRow = row;
                        pivotColumn = column;
                    }
                }
            }

            if (largestOffDiagonal <= tolerance) break;

            double diagonalDifference = values[pivotColumn, pivotColumn] - values[pivotRow, pivotRow];
            double angle = 0.5 * Math.Atan2(2.0 * values[pivotRow, pivotColumn], diagonalDifference);
            double cosine = Math.Cos(angle);
            double sine = Math.Sin(angle);

            for (int index = 0; index < size; index++)
            {
                if (index == pivotRow || index == pivotColumn) continue;

                double rowValue = values[index, pivotRow];
                double columnValue = values[index, pivotColumn];
                values[index, pivotRow] = values[pivotRow, index] =
                    cosine * rowValue - sine * columnValue;
                values[index, pivotColumn] = values[pivotColumn, index] =
                    sine * rowValue + cosine * columnValue;
            }

            double rowDiagonal = values[pivotRow, pivotRow];
            double columnDiagonal = values[pivotColumn, pivotColumn];
            double pivot = values[pivotRow, pivotColumn];
            values[pivotRow, pivotRow] =
                cosine * cosine * rowDiagonal - 2.0 * sine * cosine * pivot +
                sine * sine * columnDiagonal;
            values[pivotColumn, pivotColumn] =
                sine * sine * rowDiagonal + 2.0 * sine * cosine * pivot +
                cosine * cosine * columnDiagonal;
            values[pivotRow, pivotColumn] = values[pivotColumn, pivotRow] = 0.0;
        }

        double smallest = values[0, 0];
        for (int index = 1; index < size; index++)
        {
            smallest = Math.Min(smallest, values[index, index]);
        }

        return (smallest, tolerance);
    }
}
