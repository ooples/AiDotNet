namespace AiDotNet.Helpers;

/// <summary>
/// Helper class that provides common operations for regression analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Regression is a technique used in AI to find relationships between variables.
/// For example, predicting house prices based on features like size, location, and age.
/// This helper class provides tools to make regression work better with your data.
/// </remarks>
public static class RegressionHelper<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with the specified type T.
    /// </summary>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Standardizes the input features and target values by centering them around zero
    /// and scaling them to have unit variance.
    /// </summary>
    /// <param name="x">The input feature matrix where rows are samples and columns are features.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>
    /// A tuple containing:
    /// - xScaled: The standardized feature matrix
    /// - yScaled: The standardized target values
    /// - xMean: The mean of each feature column
    /// - xStd: The standard deviation of each feature column
    /// - yMean: The mean of the target values
    /// - yStd: The standard deviation of the target values
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method prepares your data for machine learning by making all features
    /// comparable to each other. Think of it like converting different measurements (inches, pounds, etc.)
    /// to a common scale.
    /// 
    /// For example, if one feature ranges from 0-1000 and another from 0-1, the larger one might
    /// dominate the model. Standardization solves this by:
    /// 1. Subtracting the average (centering around zero)
    /// 2. Dividing by the standard deviation (scaling to similar ranges)
    /// 
    /// The method returns both the transformed data and the values used for transformation,
    /// which you'll need later to transform new data or interpret results.
    /// </remarks>
    public static (Matrix<T> xScaled, Vector<T> yScaled, Vector<T> xMean, Vector<T> xStd, T yMean, T yStd) CenterAndScale(Matrix<T> x, Vector<T> y)
    {
        // Calculate mean for each feature column
        Vector<T> xMean = new(x.Columns);
        for (int j = 0; j < x.Columns; j++)
        {
            xMean[j] = x.GetColumn(j).Mean();
        }

        // Calculate mean for target values
        T yMean = y.Mean();

        // Calculate standard deviation for each feature column
        Vector<T> xStd = new(x.Columns);
        for (int j = 0; j < x.Columns; j++)
        {
            xStd[j] = StatisticsHelper<T>.CalculateStandardDeviation(x.GetColumn(j));
        }

        // Calculate standard deviation for target values
        T yStd = StatisticsHelper<T>.CalculateStandardDeviation(y);

        // Standardize feature matrix: (x - mean) / std
        Matrix<T> xScaled = new(x.Rows, x.Columns);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                xScaled[i, j] = _numOps.Divide(_numOps.Subtract(x[i, j], xMean[j]), xStd[j]);
            }
        }

        // Standardize target values: (y - mean) / std
        Vector<T> yScaled = y.Transform(yi => _numOps.Divide(_numOps.Subtract(yi, yMean), yStd));

        return (xScaled, yScaled, xMean, xStd, yMean, yStd);
    }
}
