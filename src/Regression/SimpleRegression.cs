namespace AiDotNet.Regression;

/// <summary>
/// Implements simple linear regression, which predicts a single output value based on a single input feature.
/// This is the most basic form of regression that finds the best-fitting straight line through a set of points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Simple linear regression models the relationship between two variables by fitting a linear equation:
/// y = mx + b
/// where:
/// - y is the predicted output value
/// - x is the input feature value
/// - m is the slope (coefficient)
/// - b is the y-intercept (where the line crosses the y-axis)
/// </para>
/// <para><b>For Beginners:</b> Simple linear regression is like drawing the best straight line through a set of points.
/// 
/// Think of it like this:
/// - You have data points on a graph (like house sizes and their prices)
/// - You want to find the line that best represents the relationship
/// - This line helps you predict new values (like the price of a house based on its size)
/// 
/// For example, if you plot people's heights and weights, simple regression would find
/// the line that shows how weight typically increases with height, allowing you to
/// estimate someone's weight if you only know their height.
/// </para>
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
    /// <remarks>
    /// <para>
    /// This constructor creates a new simple regression model with the specified configuration options and
    /// regularization method. If options are not provided, default values are used. Regularization helps prevent
    /// overfitting by adding penalties for model complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up a new simple regression model.
    /// 
    /// Think of it like setting up a new tool:
    /// - You can use the default settings (by not specifying options)
    /// - Or you can customize how it works (by providing specific options)
    /// - You can also add regularization, which acts like a safeguard to prevent the model
    ///   from memorizing the data instead of learning the general pattern
    /// 
    /// For example, when setting up a simple regression to predict house prices based on size,
    /// you might want to include an intercept (base price) or use regularization if you have
    /// limited data samples.
    /// </para>
    /// </remarks>
    public SimpleRegression(RegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
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
    /// <para>
    /// This method finds the best-fitting line by minimizing the sum of squared differences between
    /// the predicted values and the actual values in the training data. It computes the coefficient (slope)
    /// and intercept values that define the regression line.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to make predictions based on your data.
    /// 
    /// The training process works like this:
    /// 
    /// 1. The model looks at all your data points (like house sizes and prices)
    /// 2. It tries different straight lines to see which one fits the points best
    /// 3. "Best fit" means the line that has the smallest total distance from all points
    /// 4. Once found, this line gives you the formula to predict new values
    /// 
    /// For example, after training on house data, the model might learn that:
    /// price = $100,000 + ($100 × square_footage)
    /// This means a house has a base price of $100,000 plus $100 for each square foot.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        RegressionValidator.ValidateFeatureCount(x, 1, "Simple regression");

        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(Regularization.Regularize(xTx));
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
    /// Creates a new instance of the simple regression model with the same options.
    /// </summary>
    /// <returns>A new instance of the simple regression model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the simple regression model with the same configuration
    /// options and regularization method as the current instance, but without copying the trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters.
    /// 
    /// Think of it like getting a clean notepad with the same paper type and line spacing, but 
    /// without any writing on it yet. The new model has the same settings (like whether to include
    /// an intercept term), but hasn't learned any coefficients from data.
    /// 
    /// This is primarily used internally by the framework when doing things like:
    /// - Cross-validation (testing the model on different data splits)
    /// - Building model ensembles
    /// - Creating copies of models for experimentation
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        return new SimpleRegression<T>(Options, Regularization);
    }

    /// <summary>
    /// Returns the type identifier for this regression model.
    /// </summary>
    /// <returns>
    /// The model type identifier for simple regression.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method is used internally for model identification and serialization purposes.
    /// It returns an enum value that identifies this model as a simple regression model.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply tells the system what kind of model this is.
    /// 
    /// It's like a name tag for the model that says "I am a simple regression model."
    /// This is useful when:
    /// - Saving the model to a file
    /// - Loading a model from a file
    /// - Logging information about the model
    /// 
    /// You generally won't need to call this method directly in your code.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.SimpleRegression;
    }
}
