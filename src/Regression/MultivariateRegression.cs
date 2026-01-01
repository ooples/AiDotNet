namespace AiDotNet.Regression;

/// <summary>
/// Represents a multivariate linear regression model that predicts a target value based on multiple input features.
/// </summary>
/// <remarks>
/// <para>
/// Multivariate linear regression is a statistical method that models the relationship between multiple independent
/// variables and a dependent variable by fitting a linear equation to the observed data. The model assumes that the
/// relationship between inputs and the output is linear, meaning that the output can be calculated as a weighted sum
/// of the input features plus a constant term (intercept) if included.
/// </para>
/// <para><b>For Beginners:</b> Multivariate regression is like a recipe that combines several ingredients to predict an outcome.
/// 
/// Think of it like a car's fuel efficiency calculator:
/// - You provide information like car weight, engine size, aerodynamics, etc.
/// - Each factor has a certain importance (coefficient) in determining fuel efficiency
/// - The model combines all these factors to make a prediction
/// 
/// For example, the formula might be:
/// Miles per gallon = 35 - (0.005 × Car Weight) - (2 × Engine Size) + (3 × Aerodynamic Rating)
/// 
/// The model learns the best values for these coefficients from your training data to make accurate predictions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MultivariateRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MultivariateRegression{T}"/> class with optional custom options and regularization.
    /// </summary>
    /// <param name="options">Custom options for the regression algorithm. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new multivariate regression model with the specified options and regularization. If no options
    /// are provided, default values are used. Regularization helps prevent overfitting by penalizing large coefficient values.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new multivariate regression model with your chosen settings.
    /// 
    /// When creating a regression model:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps prevent the model from fitting too closely to the training data
    /// 
    /// Regularization is like adding a rule that says "keep things simple unless there's strong evidence." 
    /// This typically helps the model perform better on new data it hasn't seen before.
    /// </para>
    /// </remarks>
    public MultivariateRegression(RegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Trains the multivariate regression model using the provided features and target values.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the values to predict.</param>
    /// <remarks>
    /// <para>
    /// This method trains the multivariate regression model using the normal equation approach, which finds the optimal
    /// coefficients by solving a system of linear equations. The normal equation is given by:
    /// coefficients = (X^T * X + regularization)^(-1) * X^T * y, where X^T is the transpose of the feature matrix X.
    /// This approach directly computes the optimal coefficients without requiring iterative optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model learns from your data.
    /// 
    /// During training:
    /// 1. If an intercept is used, an extra column of 1's is added to the input features
    /// 2. The model applies mathematical operations to find the best coefficients
    /// 3. These coefficients determine how much each feature contributes to the prediction
    /// 
    /// Unlike some other models that learn gradually through many iterations, multivariate regression
    /// finds the optimal solution in one step by solving a mathematical equation.
    /// 
    /// For example, the model might learn that for car fuel efficiency:
    /// - Each additional pound of weight reduces efficiency by 0.005 mpg
    /// - Each liter of engine size reduces efficiency by 2 mpg
    /// - Each point in aerodynamic rating improves efficiency by 3 mpg
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
            Coefficients = solution.Slice(1, solution.Length - 1);
        }
        else
        {
            Coefficients = solution;
        }
    }

    /// <summary>
    /// Makes predictions for new data points using the trained multivariate regression model.
    /// </summary>
    /// <param name="input">The feature matrix where each row is a sample to predict.</param>
    /// <returns>A vector containing the predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method makes predictions by applying the learned coefficients to the input features. If an intercept term
    /// is used, it is added to the feature matrix before multiplication. The prediction is calculated as:
    /// y = X * coefficients (+ intercept if enabled).
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model uses what it learned to make predictions on new data.
    /// 
    /// The prediction process:
    /// 1. For each data point, take its feature values
    /// 2. Multiply each feature by its corresponding coefficient
    /// 3. Add all these products together (plus the intercept if used)
    /// 4. The result is the predicted value
    /// 
    /// It's like following a recipe with precise measurements:
    /// - 0.005 units of effect for each unit of the first feature
    /// - 2 units of effect for each unit of the second feature
    /// - And so on for each feature
    /// 
    /// The model combines all these effects to produce the final prediction.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Multiply input by coefficients and add intercept
        var predictions = input.Multiply(Coefficients);
        if (Options.UseIntercept)
        {
            predictions = predictions.Add(Intercept);
        }
        return predictions;
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type, in this case, MultivariateRegression.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an enumeration value indicating that this is a multivariate regression model. This is used
    /// for type identification when working with different regression models in a unified manner.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply identifies what kind of model this is.
    /// 
    /// It returns a label (MultivariateRegression) that:
    /// - Identifies this specific type of model
    /// - Helps other code handle the model appropriately
    /// - Is used when saving or loading models
    /// 
    /// It's like a name tag that lets other parts of the program know what kind of model they're working with.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.MultivariateRegression;
    }

    /// <summary>
    /// Creates a new instance of the Multivariate Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Multivariate Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Multivariate Regression model, including its coefficients,
    /// intercept, and configuration options. The new instance is completely independent of the original,
    /// allowing modifications without affecting the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate of your recipe:
    /// - It copies all the configuration settings (like whether to use an intercept)
    /// - It preserves the coefficients (the importance values for each feature)
    /// - It maintains the intercept (the starting point or base value)
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newModel = new MultivariateRegression<T>(Options, Regularization);

        // Copy coefficients if they exist
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        // Copy the intercept
        newModel.Intercept = Intercept;

        return newModel;
    }
}
