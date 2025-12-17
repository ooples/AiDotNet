namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Exponential Smoothing, a time series forecasting method that gives
/// exponentially decreasing weights to older observations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Exponential smoothing is a popular forecasting method for time series data that works by applying
/// weighted averages where the weights decrease exponentially as observations get older. This means
/// recent observations have more influence on forecasts than older observations.
/// </para>
/// <para><b>For Beginners:</b> Exponential smoothing is like a weighted average that gives more importance
/// to recent data points and less importance to older ones. Imagine predicting tomorrow's temperature:
/// you might care more about today's temperature than what happened two weeks ago. This method works
/// similarly, gradually "forgetting" older data while focusing on newer trends. It's particularly good
/// for data that changes over time and where recent observations are more relevant for prediction.
/// This class lets you configure how quickly the model "forgets" old data and how it handles trends
/// and seasonal patterns.</para>
/// </remarks>
public class ExponentialSmoothingOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the initial smoothing factor (alpha) that controls the weight given to recent observations.
    /// </summary>
    /// <value>The initial alpha value, defaulting to 0.3.</value>
    /// <remarks>
    /// <para>
    /// Alpha is the primary smoothing parameter that determines how quickly the influence of older observations
    /// decreases. Values range from 0 to 1, with higher values giving more weight to recent observations
    /// and less weight to older ones.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the model "forgets" older data points.
    /// With the default value of 0.3, the model gives moderate importance to recent observations while
    /// still considering older data. A higher value (closer to 1) would make the model focus almost
    /// entirely on the most recent data points, making it react quickly to changes but potentially
    /// overreacting to random fluctuations. A lower value (closer to 0) would make the model consider
    /// a longer history, resulting in smoother forecasts that might be slower to detect new trends.
    /// Think of it as adjusting how "short-term focused" your predictions should be.</para>
    /// </remarks>
    public double InitialAlpha { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the initial trend smoothing factor (beta) that controls the weight given to the trend component.
    /// </summary>
    /// <value>The initial beta value, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// Beta is the trend smoothing parameter that determines how the model responds to changes in the trend
    /// of the time series. Values range from 0 to 1, with higher values making the trend component more
    /// responsive to recent changes in the trend.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the model adapts to changes in the direction
    /// of your data. With the default value of 0.1, the model gradually adjusts its understanding of trends.
    /// For example, if your sales data has been increasing by 5% each month but suddenly starts increasing
    /// by 10% each month, a higher beta value would make the model recognize and adapt to this change in
    /// trend more quickly. Lower values create more stable trend estimates that don't change dramatically
    /// with short-term fluctuations. This is only used when <see cref="UseTrend"/> is set to true.</para>
    /// </remarks>
    public double InitialBeta { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the initial seasonal smoothing factor (gamma) that controls the weight given to the seasonal component.
    /// </summary>
    /// <value>The initial gamma value, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// Gamma is the seasonal smoothing parameter that determines how the model responds to changes in the
    /// seasonal pattern of the time series. Values range from 0 to 1, with higher values making the seasonal
    /// component more responsive to recent changes in the seasonal pattern.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the model adapts to changes in seasonal patterns
    /// in your data. With the default value of 0.1, the model gradually updates its understanding of seasonal
    /// effects. For example, if your retail business always had a December sales spike of 50% but this year
    /// it's 70%, a higher gamma value would make the model adjust its seasonal expectations more quickly.
    /// Lower values create more stable seasonal estimates that don't change dramatically with one unusual
    /// season. This is only used when <see cref="UseSeasonal"/> is set to true.</para>
    /// </remarks>
    public double InitialGamma { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to include a trend component in the exponential smoothing model.
    /// </summary>
    /// <value>True to include a trend component, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the model will account for trends in the data (consistent upward or downward movement).
    /// This corresponds to using either Simple Exponential Smoothing (when false) or Holt's Linear Method
    /// (when true).
    /// </para>
    /// <para><b>For Beginners:</b> This determines whether the model should look for and predict continuing
    /// upward or downward movement in your data. With the default value of true, the model will identify
    /// trends (like steadily increasing sales or gradually decreasing costs) and use them in predictions.
    /// If your data doesn't show clear directional movement over time, or if you want simpler predictions
    /// that don't assume trends will continue, you might set this to false.</para>
    /// </remarks>
    public bool UseTrend { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include a seasonal component in the exponential smoothing model.
    /// </summary>
    /// <value>True to include a seasonal component, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the model will account for seasonal patterns in the data (regular fluctuations that
    /// repeat at fixed intervals). This corresponds to using Holt-Winters' Method when both trend and
    /// seasonality are enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This determines whether the model should look for and predict repeating
    /// patterns in your data. With the default value of false, the model ignores potential seasonal effects.
    /// If your data has regular patterns that repeat (like higher sales every December, or higher website
    /// traffic every weekend), you should set this to true. You'll also need to specify the seasonal period
    /// in the base <see cref="TimeSeriesRegressionOptions{T}.SeasonalPeriod"/> property (like 12 for monthly
    /// data with yearly patterns, or 7 for daily data with weekly patterns).</para>
    /// </remarks>
    public bool UseSeasonal { get; set; } = false;

    /// <summary>
    /// Gets or sets the step size for grid search when optimizing smoothing parameters.
    /// </summary>
    /// <value>The grid search step size, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// When the model is being trained, it can search for optimal values of alpha, beta, and gamma by
    /// testing different combinations. This parameter controls how finely the search grid is divided,
    /// with smaller values providing more precise but more computationally expensive optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how thoroughly the model searches for the best smoothing
    /// parameters. With the default value of 0.1, the model will try values like 0.1, 0.2, 0.3, etc.
    /// A smaller value like 0.01 would make it try more values (0.01, 0.02, 0.03, etc.), potentially
    /// finding better parameters but taking longer to train. Think of it as the precision of the model's
    /// search for optimal settings - smaller steps mean a more detailed search but require more
    /// computation time.</para>
    /// </remarks>
    public double GridSearchStep { get; set; } = 0.1;
}
