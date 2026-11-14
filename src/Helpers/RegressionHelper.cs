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
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        int columns = x.Columns;
        var xTensor = Tensor<T>.FromMatrix(x);
        var columnAxis = new[] { 0 };

        // Calculate mean for each feature column
        Vector<T> xMean = engine.ReduceMean(xTensor, columnAxis, false).ToVector();

        // Calculate mean for target values
        T yMean = y.Mean();

        // Calculate the sample (n - 1) standard deviation of each feature column. Fewer than two rows
        // give zero spread, matching CalculateStandardDeviation.
        Tensor<T> columnStd = x.Rows < 2
            ? new Tensor<T>(new[] { columns })
            : engine.TensorStdMean(xTensor, columnAxis, 1, false).Item1;

        if (columns > 0)
        {
            // Report the first non-finite column: score non-finite columns columns - j (unique, so
            // argmax is the lowest such j) and finite columns zero.
            var descending = engine.TensorArange<T>(columns, 0, -1);
            var badScore = engine.TensorWhere(engine.TensorIsFinite(columnStd),
                new Tensor<T>(new[] { columns }), descending);
            int first = engine.TensorArgMax(badScore, 0).GetFlat(0);
            double firstScale = _numOps.ToDouble(columnStd.GetFlat(first));
            if (double.IsNaN(firstScale) || double.IsInfinity(firstScale))
            {
                throw new ArgumentException($"Feature {first} has a non-finite standard deviation.", nameof(x));
            }
        }

        // A constant feature is already centered to zero. Using unit scale keeps it zero and
        // lets callers retain harmless constant columns instead of manufacturing NaN values.
        var unitScale = engine.TensorAddScalar(new Tensor<T>(new[] { columns }), _numOps.One);
        Vector<T> xStd = engine.TensorWhere(engine.TensorEquals(columnStd, _numOps.Zero), unitScale, columnStd)
            .ToVector();

        // Calculate standard deviation for target values
        T yStd = StatisticsHelper<T>.CalculateStandardDeviation(y);
        double targetScale = _numOps.ToDouble(yStd);
        if (double.IsNaN(targetScale) || double.IsInfinity(targetScale))
        {
            throw new ArgumentException("The target has a non-finite standard deviation.", nameof(y));
        }

        // A constant target is a valid regression problem: its standardized response is all zero
        // and the original mean becomes the fitted intercept.
        if (targetScale == 0.0) yStd = _numOps.One;

        // Standardize feature matrix: (x - mean) / std
        var fullShape = new[] { x.Rows, columns };
        var rowShape = new[] { 1, columns };
        var centered = engine.TensorSubtract(xTensor,
            engine.TensorBroadcastTo(Tensor<T>.FromVector(xMean, rowShape), fullShape));
        Matrix<T> xScaled = engine.TensorDivide(centered,
            engine.TensorBroadcastTo(Tensor<T>.FromVector(xStd, rowShape), fullShape)).ToMatrix();

        // Standardize target values: (y - mean) / std
        Vector<T> yScaled = y.Transform(yi => _numOps.Divide(_numOps.Subtract(yi, yMean), yStd));

        return (xScaled, yScaled, xMean, xStd, yMean, yStd);
    }
}
