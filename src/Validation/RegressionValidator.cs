namespace AiDotNet.Validation;

/// <summary>
/// Provides validation methods for regression operations.
/// </summary>
public static class RegressionValidator
{
    /// <summary>
    /// Validates that the input matrix has the expected number of feature columns.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The input feature matrix to validate.</param>
    /// <param name="expectedColumns">The expected number of columns.</param>
    /// <param name="regressionType">The name of the regression algorithm for error messages.</param>
    /// <exception cref="InvalidInputDimensionException">
    /// Thrown when the input matrix doesn't have the expected number of columns.
    /// </exception>
    public static void ValidateFeatureCount<T>(Matrix<T> x, int expectedColumns, string regressionType)
    {
        if (x.Columns != expectedColumns)
        {
            throw new InvalidInputDimensionException(
                $"{regressionType} expects exactly {expectedColumns} feature column(s). " +
                $"Input matrix has {x.Columns} columns.");
        }
    }

    /// <summary>
    /// Validates that the input and output data have compatible dimensions for regression.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The input feature matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <exception cref="InvalidInputDimensionException">
    /// Thrown when the number of rows in the input matrix doesn't match the length of the target vector.
    /// </exception>
    public static void ValidateInputOutputDimensions<T>(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new InvalidInputDimensionException(
                $"The number of samples in the input matrix ({x.Rows}) " +
                $"must match the number of target values ({y.Length}).");
        }
    }

    /// <summary>
    /// Validates that the input data doesn't contain any NaN or infinity values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The input feature matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <exception cref="InvalidDataException">
    /// Thrown when the input data contains NaN or infinity values.
    /// </exception>
    public static void ValidateDataValues<T>(Matrix<T> x, Vector<T> y)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Check for NaN or infinity in x
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                if (numOps.IsNaN(x[i, j]) || numOps.IsInfinity(x[i, j]))
                {
                    throw new InvalidDataException(
                        $"Input matrix contains NaN or infinity at position ({i}, {j}).");
                }
            }
        }

        // Check for NaN or infinity in y
        for (int i = 0; i < y.Length; i++)
        {
            if (numOps.IsNaN(y[i]) || numOps.IsInfinity(y[i]))
            {
                throw new InvalidDataException(
                    $"Target vector contains NaN or infinity at position {i}.");
            }
        }
    }
}