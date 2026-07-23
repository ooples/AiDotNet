namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for weighted regression models, which assign different importance to different observations.
/// </summary>
/// <remarks>
/// <para>
/// Weighted regression is an extension of standard regression techniques that allows different observations to have 
/// different levels of influence on the model. By assigning weights to each observation, you can control how much 
/// each data point contributes to the parameter estimation process. This approach is particularly useful when 
/// dealing with heteroscedasticity (non-constant variance in errors), outliers, or when some observations are known 
/// to be more reliable or important than others. Common applications include time series analysis where recent 
/// observations may be weighted more heavily than older ones, or situations where observations have different 
/// measurement precision. This class inherits from RegressionOptions and adds parameters specific to weighted 
/// regression.
/// </para>
/// <para><b>For Beginners:</b> Weighted regression lets you give some data points more influence than others.
/// 
/// In standard regression:
/// - All observations have equal influence on the model
/// - This assumes all data points are equally important and reliable
/// 
/// Weighted regression solves this by:
/// - Allowing you to assign different weights to different observations
/// - Giving more influence to observations with higher weights
/// - Reducing the impact of less reliable or less important data points
/// 
/// This approach is useful when:
/// - Some observations are more reliable than others
/// - Recent data is more relevant than older data
/// - Certain observations are known to be outliers
/// - Data points have different levels of measurement precision
/// 
/// For example, in time series forecasting, you might assign higher weights to
/// recent observations to make your model more responsive to recent trends.
/// 
/// This class lets you configure how the weighted regression model is structured.
/// </para>
/// </remarks>
public class WeightedRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the order of the regression model.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the order of the regression model, which determines the highest power of the 
    /// independent variable included in the model. For example, an order of 1 corresponds to a linear regression 
    /// (y = a + bx), an order of 2 corresponds to a quadratic regression (y = a + bx + cx²), and so on. A higher 
    /// order allows the model to capture more complex nonlinear relationships but increases the risk of overfitting. 
    /// The default value of 1 provides a simple linear model suitable for many applications. The optimal order 
    /// depends on the underlying relationship between the variables and can be determined using techniques such as 
    /// cross-validation or information criteria.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the complexity of the relationship between variables.
    /// 
    /// The regression order:
    /// - Defines the highest power of the independent variable in the equation
    /// - Determines the shape of the curve that can be fitted to your data
    /// - Higher orders can model more complex, nonlinear relationships
    /// 
    /// The default value of 1 means:
    /// - The model is a straight line (linear regression): y = a + bx
    /// - This is appropriate for many relationships that are approximately linear
    /// 
    /// Common values and their equations:
    /// - 1: Linear (y = a + bx)
    /// - 2: Quadratic (y = a + bx + cx²)
    /// - 3: Cubic (y = a + bx + cx² + dx³)
    /// 
    /// When to adjust this value:
    /// - Increase it when the relationship between variables is clearly nonlinear
    /// - Keep at 1 for simple linear relationships
    /// - Be cautious with high orders as they can lead to overfitting
    /// 
    /// For example, if plotting your data shows a clear curved relationship,
    /// you might increase this to 2 to fit a quadratic curve.
    /// </para>
    /// </remarks>
    public int Order { get; set; } = 1;

    /// <summary>
    /// Gets or sets the weights assigned to each observation.
    /// </summary>
    /// <value>A vector of weights, defaulting to an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the weights assigned to each observation in the regression model. Each weight 
    /// determines the relative influence of the corresponding observation on the parameter estimation process. 
    /// Higher weights give more influence to the corresponding observations, while lower weights reduce their 
    /// influence. The weights should be non-negative, with zero weights effectively excluding the corresponding 
    /// observations from the model. The default value is an empty vector, which means that weights must be 
    /// explicitly provided before fitting the model. The length of the weights vector should match the number of 
    /// observations in the dataset. Common weighting schemes include equal weights (standard regression), inverse 
    /// variance weights (to account for heteroscedasticity), exponential decay weights (for time series), and 
    /// robust weights (to reduce the influence of outliers).
    /// </para>
    /// <para><b>For Beginners:</b> This setting specifies how much influence each data point has on the model.
    /// 
    /// The weights vector:
    /// - Contains a weight value for each observation in your dataset
    /// - Higher weights give that observation more influence on the model
    /// - Lower weights reduce an observation's influence
    /// - Zero weights effectively remove observations from consideration
    /// 
    /// The default value is an empty vector, which means:
    /// - You must provide weights before fitting the model
    /// - If you don't set weights, all observations will have equal influence
    /// 
    /// Common weighting approaches:
    /// - Equal weights (1, 1, 1, ...): Standard regression with equal influence
    /// - Time-based weights (e.g., 0.2, 0.4, 0.7, 1.0): More recent data has more influence
    /// - Reliability weights: More reliable measurements get higher weights
    /// - Robust weights: Automatically reduce weights for potential outliers
    /// 
    /// When to adjust this value:
    /// - Set specific weights when some observations should have more influence than others
    /// - Use time-based weights for time series where recency matters
    /// - Use inverse variance weights when observations have different precision
    /// 
    /// For example, in a time series model of monthly sales, you might use weights like
    /// [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] to give more recent months more influence.
    /// </para>
    /// </remarks>
    public Vector<T> Weights { get; set; } = Vector<T>.Empty();
}
