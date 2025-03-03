namespace AiDotNet.Regression;

/// <summary>
/// Implements multiple linear regression, which predicts a single output value based on multiple input features.
/// This model finds the best-fitting linear relationship between several input variables and one output variable.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
public class MultipleRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Creates a new multiple regression model.
    /// </summary>
    /// <param name="options">
    /// Optional configuration settings for the regression model. These settings control aspects like:
    /// - Whether to include an intercept term (the "baseline" value when all inputs are zero)
    /// - How to handle numerical precision
    /// If not provided, default options will be used.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting. Regularization is a technique that helps
    /// the model generalize better to new data by preventing the model from becoming too complex.
    /// If not provided, no regularization will be applied.
    /// </param>
    public MultipleRegression(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Trains the multiple regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">
    /// The input features matrix, where:
    /// - Each row represents one data sample (example)
    /// - Each column represents one feature (input variable)
    /// For example, if predicting house prices, columns might include square footage, number of bedrooms, etc.
    /// </param>
    /// <param name="y">
    /// The target values vector, containing the actual output values that the model should learn to predict.
    /// Each element corresponds to a row in the input matrix.
    /// </param>
    /// <remarks>
    /// This method uses the "normal equation" approach to find the optimal coefficients:
    /// coefficients = (X^T * X + regularization)^(-1) * X^T * y
    /// where X^T is the transpose of the input matrix X.
    /// 
    /// In simpler terms, the method finds the coefficients that minimize the difference between
    /// the predicted values and the actual values in the training data.
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
            Coefficients = new Vector<T>([.. solution.Skip(1)]);
        }
        else
        {
            Coefficients = new Vector<T>(solution);
        }
    }

    /// <summary>
    /// Returns the type identifier for this regression model.
    /// </summary>
    /// <returns>
    /// The model type identifier for multiple regression.
    /// </returns>
    /// <remarks>
    /// This method is used internally for model identification and serialization purposes.
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.MultipleRegression;
    }
}