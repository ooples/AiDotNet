namespace AiDotNet.Models.Inputs;

/// <summary>
/// Represents the input data required for calculating error statistics between actual and predicted values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class encapsulates the essential data needed to compute various error metrics that evaluate 
/// model performance. It contains the actual (ground truth) values, the model's predicted values, 
/// and the number of features used in the model, which is necessary for calculating certain 
/// adjusted error metrics like adjusted R-squared.
/// </para>
/// <para><b>For Beginners:</b> This class holds the data needed to measure how accurate your model is.
/// 
/// To evaluate a machine learning model, you need:
/// - The actual values (what you're trying to predict)
/// - The predicted values (what your model outputs)
/// - The number of features (variables) used in your model
/// 
/// With this information, AiDotNet can calculate various error metrics like:
/// - Mean Squared Error (MSE)
/// - Mean Absolute Error (MAE)
/// - R-squared
/// - And many others
/// 
/// These metrics help you understand how well your model is performing and where it might need improvement.
/// </para>
/// </remarks>
public class ErrorStatsInputs<T>
{
    /// <summary>
    /// Gets or sets the actual (ground truth) values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the correct values you're trying to predict.</para>
    /// </remarks>
    public Vector<T> Actual { get; set; } = Vector<T>.Empty();

    /// <summary>
    /// Gets or sets the values predicted by the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the values your model outputs.</para>
    /// </remarks>
    public Vector<T> Predicted { get; set; } = Vector<T>.Empty();

    /// <summary>
    /// Gets or sets the number of features (independent variables) used in the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the number of input variables your model uses to make predictions.
    /// It's needed for calculating certain adjusted error metrics that account for model complexity.</para>
    /// </remarks>
    public int FeatureCount { get; set; }
}