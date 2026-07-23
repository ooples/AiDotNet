namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different multiplicative algorithm types for time series analysis and forecasting.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Multiplicative algorithms are special methods used when analyzing data that changes over time 
/// (time series data), especially when the pattern of change depends on the current value.
/// 
/// Think about the difference between:
/// 
/// 1. Adding $100 to your savings each month (additive growth)
/// 2. Growing your savings by 5% each month (multiplicative growth)
/// 
/// With multiplicative patterns, the changes get larger as the base value gets larger. For example, 5% of $1000 
/// is $50, but 5% of $10,000 is $500 - the same percentage creates bigger absolute changes as the value grows.
/// 
/// Multiplicative algorithms are especially useful for:
/// 
/// 1. Financial data (stock prices, sales figures)
/// 2. Population growth
/// 3. Seasonal patterns that grow or shrink proportionally to the overall trend
/// 4. Any data where percentage changes are more important than absolute changes
/// 
/// In contrast to additive methods (which use addition and subtraction), multiplicative methods use 
/// multiplication and division to model changes. They often work with data on a logarithmic scale or 
/// with ratios rather than differences.
/// 
/// This enum specifies which specific multiplicative algorithm to use for analyzing or forecasting time series data.
/// </para>
/// </remarks>
public enum MultiplicativeAlgorithmType
{
    /// <summary>
    /// Uses a Geometric Moving Average to analyze and forecast time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Geometric Moving Average (GMA) is like a regular moving average, but instead of adding 
    /// values and dividing (arithmetic mean), it multiplies values and takes the nth root (geometric mean).
    /// 
    /// Imagine you're tracking the growth of an investment:
    /// - A regular average might tell you the average value over time
    /// - A geometric average tells you the consistent growth rate that would achieve the same final result
    /// 
    /// For example, if an investment grows by 10% one year and 20% the next:
    /// - The arithmetic average is (10% + 20%)/2 = 15%
    /// - The geometric average is v(1.10 × 1.20) - 1 = 14.89%
    /// 
    /// The geometric average is slightly lower but more accurate for compounding growth.
    /// 
    /// The Geometric Moving Average:
    /// 
    /// 1. Is better than simple averages for data with growth rates (like financial returns)
    /// 
    /// 2. Reduces the impact of outliers and volatility
    /// 
    /// 3. Preserves the multiplicative relationships in the data
    /// 
    /// 4. Is commonly used in financial analysis and stock market technical indicators
    /// 
    /// In machine learning applications, GMA is useful for preprocessing financial time series data, 
    /// analyzing growth patterns, or creating features that capture multiplicative trends.
    /// </para>
    /// </remarks>
    GeometricMovingAverage,

    /// <summary>
    /// Uses Multiplicative Exponential Smoothing to analyze and forecast time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiplicative Exponential Smoothing is a forecasting method that works well for data 
    /// with trends and seasonal patterns that change proportionally to the overall level.
    /// 
    /// Imagine a retail store where sales increase during holidays. If overall sales double over five years, 
    /// the holiday boost might also double (from +$5,000 to +$10,000). This is a multiplicative pattern.
    /// 
    /// This method breaks down your data into three components:
    /// 1. Level (the base value)
    /// 2. Trend (the overall direction)
    /// 3. Seasonality (repeating patterns)
    /// 
    /// But instead of adding these components (Level + Trend + Seasonality), it multiplies them 
    /// (Level × Trend × Seasonality).
    /// 
    /// Multiplicative Exponential Smoothing:
    /// 
    /// 1. Works well when seasonal variations increase as the trend increases
    /// 
    /// 2. Uses "smoothing parameters" that determine how quickly the model adapts to changes
    /// 
    /// 3. Gives more weight to recent observations and less weight to older ones (that's the "exponential" part)
    /// 
    /// 4. Is also known as Holt-Winters multiplicative method
    /// 
    /// In machine learning and forecasting, this method is particularly useful for sales forecasting, 
    /// demand planning, stock market analysis, and any time series where the seasonal variation is 
    /// proportional to the level of the series.
    /// </para>
    /// </remarks>
    MultiplicativeExponentialSmoothing,

    /// <summary>
    /// Uses a log-transformed Seasonal and Trend decomposition using Loess (STL) to analyze time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Log-Transformed STL is a technique that first converts your data using logarithms, 
    /// then applies a powerful decomposition method to separate different patterns in your time series.
    /// 
    /// Imagine you have a photo that contains multiple layers (background, people, objects). STL is like 
    /// a tool that can separate these layers. The log transformation makes it easier to separate these 
    /// layers when they have multiplicative relationships.
    /// 
    /// The process works in two main steps:
    /// 1. Transform the data by taking the logarithm of each value
    /// 2. Apply STL (Seasonal and Trend decomposition using Loess) to break down the transformed data into:
    ///    - Trend component (long-term direction)
    ///    - Seasonal component (repeating patterns)
    ///    - Remainder component (what's left after removing trend and seasonality)
    /// 
    /// After analysis, you can transform back to the original scale using exponentiation (the opposite of logarithm).
    /// 
    /// Log-Transformed STL:
    /// 
    /// 1. Handles complex seasonal patterns that can change over time
    /// 
    /// 2. Is robust against outliers (unusual data points)
    /// 
    /// 3. Works well for data with multiplicative relationships between components
    /// 
    /// 4. Provides a flexible way to decompose time series with multiple seasonal patterns
    /// 
    /// 5. "Loess" refers to a special smoothing technique used in the decomposition
    /// 
    /// In machine learning applications, this method is valuable for preprocessing time series data before 
    /// feeding it into other algorithms, for anomaly detection (finding unusual patterns), or for understanding 
    /// the underlying components driving your time series.
    /// </para>
    /// </remarks>
    LogTransformedSTL
}
