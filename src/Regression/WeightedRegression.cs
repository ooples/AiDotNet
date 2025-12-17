global using AiDotNet.Extensions;

namespace AiDotNet.Regression;

/// <summary>
/// Implements weighted regression, a variation of linear regression where each data point has a different 
/// level of importance (weight) in determining the model parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Weighted regression extends standard regression by allowing each data point to have a different level
/// of influence on the model. This is particularly useful in scenarios where data points have varying
/// reliability, importance, or error variance.
/// </para>
/// <para><b>For Beginners:</b> Weighted regression is like giving different voting power to different data points.
/// 
/// Think of it like this:
/// - Regular regression treats all data points equally - each point gets one "vote" on where the line should go
/// - Weighted regression lets some points have more "votes" than others
/// - Points with higher weights have more influence on the final model
/// - Points with lower weights have less influence
/// 
/// For example, if you're predicting house prices:
/// - Recent sales might get higher weights because they reflect current market conditions better
/// - Unusual properties might get lower weights to prevent them from skewing the model
/// - More reliable measurements might get higher weights than less reliable ones
/// 
/// This helps you build models that focus more on the data points you trust or care about most.
/// </para>
/// </remarks>
public class WeightedRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// The weights assigned to each data point, determining their influence on the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the weight for each observation in the training data. Each weight determines
    /// how much influence its corresponding data point has during the model fitting process.
    /// </para>
    /// <para><b>For Beginners:</b> This stores how important each data point is.
    /// 
    /// The weights determine:
    /// - How much each point influences the final model
    /// - Higher weights = more influence
    /// - Lower weights = less influence
    /// - Zero weight = point is completely ignored
    /// 
    /// Think of weights like adjusting the volume for different speakers in a discussion - 
    /// you can turn up the volume for those with important information and turn down the volume
    /// for those who might be less reliable.
    /// </para>
    /// </remarks>
    private readonly Vector<T> _weights;

    /// <summary>
    /// The polynomial order for feature expansion.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This integer determines the highest power to which the input features will be raised during polynomial
    /// feature expansion. A value of 1 means linear features only, 2 adds squared terms, 3 adds cubic terms, and so on.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how complex the patterns can be that the model looks for.
    /// 
    /// The order value controls:
    /// - Order 1: The model only looks for straight-line relationships (linear)
    /// - Order 2: The model can also detect curved relationships (quadratic)
    /// - Order 3: The model can detect even more complex curves (cubic)
    /// - And so on...
    /// 
    /// Higher orders can fit more complex patterns but may also be more prone to overfitting
    /// (fitting the noise rather than the true pattern).
    /// </para>
    /// </remarks>
    private readonly int _order;

    /// <summary>
    /// Creates a new instance of the weighted regression model.
    /// </summary>
    /// <param name="options">
    /// Configuration options for the weighted regression model, including the weights for each data point
    /// and the polynomial order. The weights determine how much influence each data point has on the model.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting. Regularization adds a penalty for large 
    /// coefficient values, which helps create a more generalizable model.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when the options parameter is null or doesn't contain weights.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new weighted regression model with the specified options and regularization.
    /// The weights vector must be provided in the options, as it's a fundamental component of weighted regression.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your weighted regression model with your chosen settings.
    /// 
    /// When creating a weighted regression model:
    /// - You must provide weights for each data point (how important each point is)
    /// - You can specify the order (complexity) of the model
    /// - You can add regularization to prevent overfitting (making the model too specific to training data)
    /// 
    /// If you don't provide weights, you'll get an error because weights are essential to weighted regression.
    /// </para>
    /// </remarks>
    public WeightedRegression(WeightedRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _weights = options?.Weights ?? throw new ArgumentNullException(nameof(options), "Weights must be provided for weighted regression.");
        _order = options.Order;
    }

    /// <summary>
    /// Trains the weighted regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">
    /// The input feature matrix where each row represents a data point and each column represents a feature.
    /// </param>
    /// <param name="y">
    /// The target values vector where each element corresponds to a row in the input matrix.
    /// </param>
    /// <remarks>
    /// <para>
    /// This method implements the weighted least squares algorithm to find the optimal coefficients for the model.
    /// It first expands the features to the specified polynomial order, then applies the weights to give different
    /// importance to each data point, and finally solves the weighted normal equations to find the coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to recognize patterns in your data, respecting
    /// the importance of each point.
    /// 
    /// During training:
    /// - The features are expanded based on the order (to capture more complex patterns)
    /// - The weights are applied to each data point (giving more influence to higher-weighted points)
    /// - The model solves a complex equation to find the best fit considering these weights
    /// - If you included an intercept, it finds both the slope(s) and the y-intercept
    /// 
    /// After training is complete, the model can make predictions for new data points.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        var expandedX = ExpandFeatures(x);

        if (Options.UseIntercept)
            expandedX = expandedX.AddConstantColumn(NumOps.One);

        var weightMatrix = Matrix<T>.CreateDiagonal(_weights);
        var xTWx = expandedX.Transpose().Multiply(weightMatrix).Multiply(expandedX);
        var regularizedXTWx = xTWx.Add(Regularization.Regularize(xTWx));
        var xTWy = expandedX.Transpose().Multiply(weightMatrix).Multiply(y);

        var solution = SolveSystem(regularizedXTWx, xTWy);

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
    /// Makes predictions using the trained weighted regression model.
    /// </summary>
    /// <param name="input">
    /// The input feature matrix where each row represents a data point to predict and each column represents a feature.
    /// </param>
    /// <returns>
    /// A vector of predicted values, one for each row in the input matrix.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method generates predictions for new data points using the coefficients learned during training.
    /// It first expands the input features to the same polynomial order used during training, then applies
    /// the base class prediction method to calculate the output values.
    /// </para>
    /// <para><b>For Beginners:</b> Once the model is trained, this method uses the discovered patterns
    /// to predict outcomes for new data.
    /// 
    /// When making predictions:
    /// - The new data is transformed the same way as during training (expanded to polynomial features)
    /// - The model applies the equation it learned to generate predictions
    /// - Each input row gets one predicted value
    /// 
    /// Note that the weights only matter during training - when predicting, all we need are the 
    /// patterns (coefficients) the model discovered.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var expandedInput = ExpandFeatures(input);
        return base.Predict(expandedInput);
    }

    /// <summary>
    /// Expands the input features to include polynomial terms up to the specified order.
    /// </summary>
    /// <param name="x">The original input feature matrix.</param>
    /// <returns>
    /// A new matrix with expanded polynomial features.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method transforms the input features by adding polynomial terms up to the order specified
    /// in the model options. For each original feature, it generates additional features representing
    /// the feature raised to powers from 1 to the specified order.
    /// </para>
    /// <para><b>For Beginners:</b> This method enriches your data to help find more complex patterns.
    /// 
    /// Feature expansion adds new columns to your data:
    /// - Original features: x
    /// - If order = 2: Adds x²
    /// - If order = 3: Adds x² and x³
    /// - And so on...
    /// 
    /// For example, if your original data has one feature (height) and order = 2:
    /// - Original: [5]
    /// - Expanded: [5, 25] (height and height-squared)
    /// 
    /// This allows the model to detect non-linear relationships (curves) in your data,
    /// even though the model itself is linear.
    /// </para>
    /// </remarks>
    private Matrix<T> ExpandFeatures(Matrix<T> x)
    {
        var expandedX = new Matrix<T>(x.Rows, x.Columns * _order);

        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                for (int k = 0; k < _order; k++)
                {
                    expandedX[i, j * _order + k] = NumOps.Power(x[i, j], NumOps.FromDouble(k + 1));
                }
            }
        }

        return expandedX;
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type identifier for weighted regression.</returns>
    /// <remarks>
    /// <para>
    /// This method identifies the type of this regression model as weighted regression, which
    /// helps with model type checking and serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply identifies what kind of model this is.
    /// 
    /// It returns a label that identifies this as a weighted regression model, which helps the system
    /// recognize and handle it correctly when saving, loading, or processing models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.WeightedRegression;
    }

    /// <summary>
    /// Creates a new instance of the weighted regression model with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="WeightedRegression{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new weighted regression model that has the same configuration as the current instance.
    /// It's used for model persistence, cloning, and transferring the model's configuration to new instances.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like making a blueprint copy of your model that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar models
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its settings
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new WeightedRegression<T>((WeightedRegressionOptions<T>)Options, Regularization);
    }
}
