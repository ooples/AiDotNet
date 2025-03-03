namespace AiDotNet.Regression;

/// <summary>
/// Implements multivariate linear regression, which predicts a single output value based on multiple input features.
/// This is an extension of simple linear regression that can handle multiple input variables.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
public class MultivariateRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Creates a new multivariate regression model.
    /// </summary>
    /// <param name="options">
    /// Optional configuration settings for the regression model, such as whether to include an intercept term
    /// or how to handle numerical precision. If not provided, default options will be used.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting. Regularization helps the model generalize better
    /// to new data by penalizing extreme coefficient values. If not provided, no regularization will be applied.
    /// </param>
    public MultivariateRegression(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Trains the multivariate regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">
    /// The input features matrix, where each row represents a data point and each column represents a feature.
    /// For example, if predicting house prices, columns might include square footage, number of bedrooms, etc.
    /// </param>
    /// <param name="y">
    /// The target values vector, containing the actual output values that the model should learn to predict.
    /// Each element corresponds to a row in the input matrix.
    /// </param>
    /// <remarks>
    /// This method uses the normal equation approach to find the optimal coefficients:
    /// coefficients = (X^T * X + regularization)^(-1) * X^T * y
    /// where X^T is the transpose of the input matrix X.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(Regularization.RegularizeMatrix(xTx));
        var xTy = x.Transpose().Multiply(y);

        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = solution.Slice(1, solution.Length - 1);
        }
        else
        {
            Coefficients = solution;
        }
    }

    /// <summary>
    /// Makes predictions using the trained model for new input data.
    /// </summary>
    /// <param name="input">
    /// The input features matrix for which predictions should be made. Each row represents a data point
    /// to predict, and each column represents a feature. The columns must match the features used during training.
    /// </param>
    /// <returns>
    /// A vector of predicted values, one for each row in the input matrix.
    /// </returns>
    /// <remarks>
    /// Predictions are made using the formula: y = X * coefficients (+ intercept if enabled)
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Options.UseIntercept)
            input = input.AddConstantColumn(NumOps.One);

        return input.Multiply(Coefficients);
    }

    /// <summary>
    /// Returns the type of this regression model.
    /// </summary>
    /// <returns>
    /// The model type identifier for multivariate regression.
    /// </returns>
    protected override ModelType GetModelType()
    {
        return ModelType.MultivariateRegression;
    }
}