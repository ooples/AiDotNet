namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for additive decomposition of time series data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Additive decomposition is a technique used to break down time series data 
/// (data collected over time, like daily temperatures or monthly sales) into separate components:
/// 
/// 1. Trend - The long-term direction of the data (going up, down, or staying flat over time)
/// 2. Seasonality - Regular patterns that repeat at fixed intervals (like higher sales during holidays)
/// 3. Residual - The random fluctuations left after accounting for trend and seasonality
/// 
/// It's called "additive" because we assume these components add together to form the original data:
/// Original Data = Trend + Seasonality + Residual
/// 
/// This enum lists different algorithms that can perform this decomposition, each with its own
/// approach to separating these components.
/// </para>
/// </remarks>
public enum AdditiveDecompositionAlgorithmType
{
    /// <summary>
    /// Uses a moving average approach to decompose time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Moving Average method works by calculating averages over fixed windows of time
    /// to smooth out short-term fluctuations and highlight longer-term trends. It's like looking at the 
    /// "big picture" by averaging out the day-to-day noise in your data.
    /// 
    /// For example, instead of looking at daily sales which might vary a lot, a 7-day moving average
    /// would give you the average sales for each 7-day period, creating a smoother line that makes
    /// the overall trend easier to see.
    /// </para>
    /// </remarks>
    MovingAverage,

    /// <summary>
    /// Uses exponential smoothing to decompose time series data, giving more weight to recent observations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Exponential Smoothing is like a weighted average that gives more importance to recent data
    /// and less importance to older data. It's similar to how you might naturally pay more attention to what 
    /// happened yesterday than what happened last month.
    /// 
    /// This method is particularly useful when recent changes in your data are more important for predicting
    /// future values. For example, when forecasting product demand, recent sales trends might be more relevant
    /// than older patterns.
    /// </para>
    /// </remarks>
    ExponentialSmoothing,

    /// <summary>
    /// Uses Seasonal and Trend decomposition using Loess (STL) to break down time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> STL (Seasonal and Trend decomposition using Loess) is a more sophisticated method
    /// that can handle complex seasonal patterns that might change over time.
    /// 
    /// Imagine you're analyzing ice cream sales over several years. Not only do sales increase in summer
    /// and decrease in winter (seasonality), but maybe the overall popularity of ice cream is growing year
    /// by year (trend). STL can separate these patterns even if the summer peaks are getting higher each year.
    /// 
    /// "Loess" refers to a special statistical technique used in this method that helps fit smooth curves
    /// to different parts of your data.
    /// </para>
    /// </remarks>
    STL
}
