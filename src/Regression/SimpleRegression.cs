namespace AiDotNet.Regression;

/// <summary>
/// Implements simple linear regression, which predicts a single output value based on a single input feature.
/// This is the most basic form of regression that finds the best-fitting straight line through a set of points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// Simple linear regression models the relationship between two variables by fitting a linear equation:
/// y = mx + b
/// where:
/// - y is the predicted output value
/// - x is the input feature value
/// - m is the slope (coefficient)
/// - b is the y-intercept (where the line crosses the y-axis)
/// </remarks>
public class SimpleRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Creates a new simple regression model.
    /// </summary>
    /// <param name="options">
    /// Optional configuration settings for the regression model. These settings control aspects like:
    /// - Whether to include an intercept term (the "b" in y = mx + b)
    /// - How to handle numerical precision
    /// If not provided, default options will be used.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting. Regularization is a technique that helps
    /// the model perform better on new, unseen data by preventing it from fitting the training data too closely.
    /// If not provided, no regularization will be applied.
    /// </param>
    public SimpleRegression(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Trains the simple regression model using the provided input feature and target values.
    /// </summary>
    /// <param name="x">
    /// The input feature matrix, which must have exactly one column. Each row represents one data sample.
    /// For example, if predicting house prices based on square footage, this would be a single column of square footage values.
    /// </param>
    /// <param name="y">
    /// The target values vector, containing the actual output values that the model should learn to predict.
    /// Each element corresponds to a row in the input matrix.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input matrix has more than one column. Simple regression only works with a single input feature.
    /// </exception>
    /// <remarks>
    /// This method finds the best-fitting line by minimizing the sum of squared differences between
    /// the predicted values and the actual values in the training data.
    /// 
    /// In simple terms, it finds the line that is closest to all the data points when measuring the vertical distance.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        RegressionValidator.ValidateFeatureCount(x, 1, "Simple regression");

        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(Regularization.RegularizeMatrix(xTx));
        var xTy = x.Transpose().Multiply(y);

        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = new Vector<T>([solution[1]]);
        }
        else
        {
            Coefficients = new Vector<T>([solution[0]]);
        }
    }

    /// <summary>
    /// Returns the type identifier for this regression model.
    /// </summary>
    /// <returns>
    /// The model type identifier for simple regression.
    /// </returns>
    /// <remarks>
    /// This method is used internally for model identification and serialization purposes.
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.SimpleRegression;
    }
}