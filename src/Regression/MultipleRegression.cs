namespace AiDotNet.Regression;

/// <summary>
/// Represents a multiple linear regression model that predicts a target value based on multiple input features.
/// </summary>
/// <remarks>
/// <para>
/// Multiple linear regression extends simple linear regression to incorporate multiple input features. It models the
/// relationship between several independent variables and one dependent variable by fitting a linear equation to the
/// observed data. The model assumes that the relationship between inputs and the output is linear, meaning that the output
/// can be calculated as a weighted sum of the input features plus a constant term (intercept).
/// </para>
/// <para><b>For Beginners:</b> Multiple regression is like a formula that predicts one value based on several inputs.
/// 
/// Think of it like a house price calculator:
/// - You provide information like square footage, number of bedrooms, neighborhood rating, etc.
/// - Each feature has a certain importance (called a coefficient)
/// - The model combines all these factors with their importances to make a prediction
/// 
/// For example, the formula might be:
/// House Price = $50,000 + ($100 × Square Footage) + ($15,000 × Number of Bedrooms) + ($25,000 × Neighborhood Rating)
/// 
/// The model learns the best values for these coefficients from your training data to make accurate predictions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MultipleRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MultipleRegression{T}"/> class with optional custom options and regularization.
    /// </summary>
    /// <param name="options">Custom options for the regression algorithm. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new multiple regression model with the specified options and regularization. If no options
    /// are provided, default values are used. Regularization helps prevent overfitting by penalizing large coefficient values.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new multiple regression model with your chosen settings.
    /// 
    /// When creating a regression model:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps prevent the model from becoming too specialized to the training data
    /// 
    /// Regularization is like adding a penalty for complexity - it encourages the model to keep the coefficient values
    /// smaller, which typically results in a more general model that performs better on new data.
    /// </para>
    /// </remarks>
    public MultipleRegression(RegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Trains the multiple regression model using the provided features and target values.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
    /// <remarks>
    /// <para>
    /// This method trains the multiple regression model using the normal equation approach, which finds the optimal
    /// coefficients by solving a system of linear equations. The normal equation is given by:
    /// coefficients = (X^T * X + regularization)^(-1) * X^T * y, where X^T is the transpose of the feature matrix X.
    /// This approach directly computes the optimal coefficients without requiring iterative optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model learns from your data.
    /// 
    /// During training:
    /// 1. If an intercept is used, an extra column of 1's is added to the input features
    /// 2. The model applies matrix operations to find the best coefficients
    /// 3. These coefficients determine how much each feature contributes to the prediction
    /// 
    /// Unlike some other models that learn gradually through many iterations, multiple regression
    /// finds the optimal solution in one step by solving a system of equations.
    /// 
    /// For example, the model might learn that for house prices:
    /// - Each additional square foot adds $100 to the price
    /// - Each bedroom adds $15,000 to the price 
    /// - Each point in neighborhood rating adds $25,000 to the price
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (Options.UseIntercept)
            x = x.AddConstantColumn(NumOps.One);
        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(Regularization.Regularize(xTx));
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
    /// Creates a new instance of the Multiple Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Multiple Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Multiple Regression model, including its coefficients,
    /// intercept, options, and regularization. The new instance is completely independent of the original,
    /// allowing modifications without affecting the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of the current regression model.
    /// 
    /// The copy includes:
    /// - The same coefficients (the importance values for each feature)
    /// - The same intercept (the starting point value)
    /// - The same options (settings like whether to use an intercept)
    /// - The same regularization (settings that help prevent overfitting)
    /// 
    /// This is useful when you want to:
    /// - Create a backup before modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new MultipleRegression with the same options and regularization
        var newModel = new MultipleRegression<T>(Options, Regularization);

        // Copy the coefficients
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        // Copy the intercept
        newModel.Intercept = Intercept;

        return newModel;
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type, in this case, MultipleRegression.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an enumeration value indicating that this is a multiple regression model. This is used
    /// for type identification when working with different regression models in a unified manner.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply identifies what kind of model this is.
    /// 
    /// It returns a label (MultipleRegression) that:
    /// - Identifies this specific type of model
    /// - Helps other code handle the model appropriately
    /// - Is used for model identification and categorization
    /// 
    /// It's like a name tag that lets other parts of the program know what kind of model they're working with.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.MultipleRegression;
    }
}
