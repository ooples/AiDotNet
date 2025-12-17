namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, 
/// Trend, and Seasonal components) time series forecasting model.
/// </summary>
/// <remarks>
/// <para>
/// TBATS is an advanced time series forecasting model designed to handle complex seasonal patterns, including 
/// multiple seasonal periods. It combines several powerful techniques: Box-Cox transformation for stabilizing 
/// variance, Fourier terms for handling multiple seasonal patterns, ARMA models for capturing short-term 
/// dependencies, and trend components with optional damping. TBATS is particularly effective for time series 
/// with multiple seasonal patterns of different lengths (e.g., daily, weekly, and yearly patterns) and can 
/// automatically select the appropriate components based on the data. This class provides configuration options 
/// for controlling the various components and optimization parameters of the TBATS model.
/// </para>
/// <para><b>For Beginners:</b> TBATS is a powerful forecasting model for time series with complex seasonal patterns.
/// 
/// When forecasting time series data:
/// - Simple models struggle with multiple seasonal patterns (e.g., daily, weekly, and yearly cycles)
/// - Traditional approaches may require separate modeling for each seasonal pattern
/// 
/// TBATS solves this by:
/// - Handling multiple seasonal periods simultaneously
/// - Using Fourier series to efficiently represent seasonal patterns
/// - Transforming data to stabilize variance (Box-Cox transformation)
/// - Modeling short-term dependencies (ARMA components)
/// - Incorporating trend with optional damping
/// 
/// This approach offers several benefits:
/// - Effectively captures complex seasonal patterns
/// - Handles irregular seasonality (e.g., varying month lengths)
/// - Produces accurate forecasts for data with multiple cycles
/// - Automatically selects appropriate components
/// 
/// This class lets you configure how the TBATS model analyzes and forecasts your time series data.
/// </para>
/// </remarks>
public class TBATSModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the Box-Cox transformation parameter.
    /// </summary>
    /// <value>An integer value, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the parameter for the Box-Cox transformation, which is used to stabilize the 
    /// variance in the time series. The Box-Cox transformation is a power transformation defined as 
    /// (y^? - 1)/? for ? ? 0 and log(y) for ? = 0. A value of 1 means no transformation is applied. A value 
    /// of 0 corresponds to a logarithmic transformation, which is useful for data with multiplicative patterns. 
    /// Other common values include 0.5 (square root transformation) and -1 (reciprocal transformation). The 
    /// optimal value depends on the characteristics of the data, particularly the relationship between the 
    /// level and the variance of the series. In practice, the value is often estimated from the data rather 
    /// than specified manually.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the data is transformed before modeling.
    /// 
    /// The Box-Cox lambda parameter:
    /// - Transforms your data to make it more suitable for forecasting
    /// - Helps stabilize variance (make fluctuations more consistent across the series)
    /// - Different values create different transformations
    /// 
    /// The default value of 1 means:
    /// - No transformation is applied to the data
    /// - The model works with the original values
    /// 
    /// Common values include:
    /// - 0: Log transformation (good for exponential growth or multiplicative patterns)
    /// - 0.5: Square root transformation (moderately right-skewed data)
    /// - -1: Reciprocal transformation (severely right-skewed data)
    /// 
    /// When to adjust this value:
    /// - Set to 0 when your data shows multiplicative patterns (variance increases with level)
    /// - Set to 0.5 for moderately skewed data
    /// - Leave at 1 when your data already has stable variance
    /// 
    /// For example, for retail sales data that grows exponentially over time,
    /// you might set this to 0 to apply a log transformation.
    /// </para>
    /// </remarks>
    public int BoxCoxLambda { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of the ARMA (AutoRegressive Moving Average) component.
    /// </summary>
    /// <value>A positive integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the order of the ARMA (AutoRegressive Moving Average) component in the TBATS model, 
    /// which captures short-term dependencies in the time series. The ARMA component combines an autoregressive 
    /// (AR) model, where the current value depends on previous values, and a moving average (MA) model, where 
    /// the current value depends on previous errors. A higher order allows the model to capture more complex 
    /// dependencies but increases the risk of overfitting and the computational cost. The default value of 1 
    /// provides a simple ARMA component suitable for many applications, capturing first-order dependencies. The 
    /// optimal value depends on the autocorrelation structure of the time series after removing seasonal and 
    /// trend components.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the model captures short-term patterns in your data.
    /// 
    /// The ARMA order:
    /// - Determines how many past values and errors influence the current prediction
    /// - Helps the model capture short-term dependencies and patterns
    /// - Higher values can model more complex relationships but risk overfitting
    /// 
    /// The default value of 1 means:
    /// - The model considers the immediate past values and errors
    /// - This is sufficient for many time series with simple short-term patterns
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 2 or 3): Can capture more complex short-term patterns
    /// - Lower values (e.g., 0): Ignore short-term dependencies entirely
    /// 
    /// When to adjust this value:
    /// - Increase it when your data shows complex short-term dependencies
    /// - Decrease it or set to 0 when your data doesn't show significant short-term patterns
    /// - Consider using model selection criteria (AIC, BIC) to determine the optimal value
    /// 
    /// For example, for financial time series with complex short-term dependencies,
    /// you might increase this to 2 or 3 to better capture those patterns.
    /// </para>
    /// </remarks>
    public int ARMAOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the damping factor for the trend component.
    /// </summary>
    /// <value>An integer value, typically between 0 and 1, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the damping factor for the trend component in the TBATS model. Trend damping 
    /// reduces the influence of the trend over time, which can lead to more conservative and often more realistic 
    /// long-term forecasts. A value of 1 means no damping is applied, and the trend continues at a constant rate. 
    /// A value between 0 and 1 applies damping, with smaller values causing the trend to flatten out more quickly. 
    /// The default value of 1 provides no damping, which is suitable for many applications where the trend is 
    /// expected to continue. The optimal value depends on the characteristics of the data and the forecast horizon, 
    /// with damping often being more important for longer-term forecasts.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the trend component behaves in long-term forecasts.
    /// 
    /// The trend damping factor:
    /// - Determines whether and how quickly the trend flattens out over time
    /// - Helps prevent unrealistic forecasts when projecting far into the future
    /// - Values between 0 and 1 create damped trends that gradually flatten
    /// 
    /// The default value of 1 means:
    /// - No damping is applied
    /// - The trend continues at the same rate indefinitely
    /// 
    /// Think of it like this:
    /// - Value of 1: Trend continues unchanged (linear growth or decline)
    /// - Values close to 1 (e.g., 0.98): Very slow damping, trend flattens very gradually
    /// - Values further from 1 (e.g., 0.8): Faster damping, trend flattens more quickly
    /// 
    /// When to adjust this value:
    /// - Decrease it when making long-term forecasts to prevent unrealistic projections
    /// - Keep at 1 for short-term forecasts or when you believe the trend will continue
    /// - Values between 0.9 and 0.98 are common for moderate damping
    /// 
    /// For example, when forecasting technology adoption that will eventually saturate,
    /// you might set this to 0.9 to gradually flatten the growth curve.
    /// </para>
    /// </remarks>
    public int TrendDampingFactor { get; set; } = 1;

    /// <summary>
    /// Gets or sets the seasonal periods to model.
    /// </summary>
    /// <value>An array of positive integers, defaulting to [7, 30, 365].</value>
    /// <remarks>
    /// <para>
    /// This property specifies the lengths of the seasonal periods to be modeled. TBATS can handle multiple 
    /// seasonal patterns simultaneously, making it particularly suitable for time series with nested seasonality. 
    /// Each value in the array represents the length of a seasonal cycle in terms of the time unit of the data. 
    /// For example, with daily data, common values might include 7 (weekly seasonality), 30 or 30.44 (monthly 
    /// seasonality), and 365 or 365.25 (yearly seasonality). The default values of [7, 30, 365] are suitable 
    /// for daily data with weekly, monthly, and yearly patterns. For hourly data, additional values like 24 
    /// (daily seasonality) might be included. The optimal values depend on the known or expected seasonal 
    /// patterns in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting specifies the seasonal cycles present in your data.
    /// 
    /// The seasonal periods:
    /// - Define the lengths of repeating patterns in your data
    /// - Allow the model to capture multiple seasonal cycles simultaneously
    /// - Should match the known or expected patterns in your data
    /// 
    /// The default value of [7, 30, 365] means:
    /// - The model will look for weekly (7-day), monthly (30-day), and yearly (365-day) patterns
    /// - This is appropriate for daily data with these common seasonal cycles
    /// 
    /// Common seasonal periods for different data frequencies:
    /// - Hourly data: [24, 168, 8766] (daily, weekly, yearly cycles)
    /// - Daily data: [7, 30, 365] (weekly, monthly, yearly cycles)
    /// - Weekly data: [52] (yearly cycle)
    /// - Monthly data: [12] (yearly cycle)
    /// 
    /// When to adjust this value:
    /// - Set based on your knowledge of the data and its natural cycles
    /// - Include all relevant seasonal periods for your specific domain
    /// - Remove periods that aren't relevant to your data
    /// 
    /// For example, for hourly electricity consumption data, you might use
    /// [24, 168] to capture daily and weekly patterns.
    /// </para>
    /// </remarks>
    public int[] SeasonalPeriods { get; set; } = new int[] { 7, 30, 365 };

    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization algorithm.
    /// </summary>
    /// <value>A positive integer, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of iterations for the optimization algorithm used to estimate 
    /// the TBATS model parameters. The optimization process iteratively refines the parameter estimates until 
    /// convergence is reached or the maximum number of iterations is exceeded. A larger value allows for more 
    /// iterations and potentially better parameter estimates but increases the computation time. The default 
    /// value of 1000 provides a reasonable upper limit for many applications, allowing sufficient iterations 
    /// for convergence while preventing excessive computation. The optimal value depends on the complexity of 
    /// the data and the desired trade-off between accuracy and computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how long the model will spend trying to find the best parameters.
    /// 
    /// The maximum iterations:
    /// - Sets an upper limit on the number of optimization steps
    /// - Prevents the algorithm from running indefinitely
    /// - Balances computation time with parameter quality
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will perform at most 1000 iterations to find optimal parameters
    /// - This is sufficient for many time series problems
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 2000): More opportunity to find better parameters, but longer runtime
    /// - Lower values (e.g., 500): Faster results, but potentially suboptimal parameters
    /// 
    /// When to adjust this value:
    /// - Increase it for complex time series with multiple seasonal patterns
    /// - Decrease it when computational resources are limited or for simpler time series
    /// - If the algorithm frequently hits this limit, consider increasing it
    /// 
    /// For example, for a complex time series with multiple strong seasonal patterns,
    /// you might increase this to 2000 to ensure the model has enough iterations to converge.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance for the optimization algorithm.
    /// </summary>
    /// <value>A positive double value, defaulting to 1e-6.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the convergence tolerance for the optimization algorithm used to estimate the 
    /// TBATS model parameters. The optimization process is considered to have converged when the relative change 
    /// in the objective function (typically the likelihood or the sum of squared errors) between iterations is 
    /// less than this tolerance. A smaller value requires more precise convergence, potentially leading to better 
    /// parameter estimates but requiring more iterations. The default value of 1e-6 (0.000001) provides a 
    /// relatively strict convergence criterion suitable for many applications, ensuring good parameter estimates 
    /// while allowing the algorithm to terminate in a reasonable number of iterations. The optimal value depends 
    /// on the desired precision of the parameter estimates and the computational resources available.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precise the optimization must be before stopping.
    /// 
    /// The tolerance:
    /// - Defines when the optimization algorithm considers itself "done"
    /// - Smaller values require more precise results before stopping
    /// - Affects both accuracy and computation time
    /// 
    /// The default value of 1e-6 (0.000001) means:
    /// - The algorithm stops when improvements between iterations are very small
    /// - This provides good precision for most applications
    /// 
    /// Think of it like this:
    /// - Smaller values (e.g., 1e-8): More precise results, but may take longer
    /// - Larger values (e.g., 1e-4): Faster results, but potentially less precise
    /// 
    /// When to adjust this value:
    /// - Decrease it when you need very precise parameter estimates
    /// - Increase it when faster computation is more important than precision
    /// - For most applications, the default value works well
    /// 
    /// For example, if you're developing a critical forecasting system where precision is paramount,
    /// you might decrease this to 1e-8 for more precise parameter estimates.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the type of matrix decomposition used in the optimization algorithm.
    /// </summary>
    /// <value>A value from the MatrixDecompositionType enumeration, defaulting to MatrixDecompositionType.Cholesky.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the type of matrix decomposition used in the optimization algorithm for estimating 
    /// the TBATS model parameters. Matrix decomposition is used to solve linear systems that arise during the 
    /// optimization process. Different decomposition methods have different numerical properties and computational 
    /// requirements. Cholesky decomposition (the default) is efficient and numerically stable for positive definite 
    /// matrices, which are common in many statistical models. Other options might include QR decomposition, which is 
    /// more stable for ill-conditioned matrices, or SVD (Singular Value Decomposition), which is the most stable but 
    /// also the most computationally expensive. The optimal choice depends on the numerical properties of the specific 
    /// problem and the desired trade-off between numerical stability and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical method used to solve equations during model fitting.
    /// 
    /// The matrix decomposition type:
    /// - Affects how certain mathematical operations are performed during optimization
    /// - Different methods have different trade-offs between speed and numerical stability
    /// - Most users don't need to change this setting
    /// 
    /// The default value of Cholesky means:
    /// - The algorithm uses Cholesky decomposition for matrix operations
    /// - This method is efficient and works well for most time series problems
    /// 
    /// Common alternatives include:
    /// - QR: More stable for difficult numerical problems, but slower
    /// - SVD: Most numerically stable, but significantly slower
    /// - LU: Another alternative with different numerical properties
    /// 
    /// When to adjust this value:
    /// - Change to QR or SVD if you encounter numerical stability issues
    /// - Keep the default for most applications
    /// - This is an advanced setting that rarely needs adjustment
    /// 
    /// For example, if your model fitting process fails with numerical errors,
    /// you might try changing this to MatrixDecompositionType.SVD for better stability.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}
