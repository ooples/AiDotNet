namespace AiDotNet.Enums;

/// <summary>
/// Represents the different components that can be extracted when decomposing a time series.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Time series decomposition is like breaking down a complex song into its individual instruments.
/// 
/// When analyzing data that changes over time (like stock prices, temperature readings, or website traffic),
/// it's often helpful to separate the data into simpler components to better understand what's happening.
/// 
/// For example, retail sales data might contain:
/// - A general upward trend due to business growth
/// - Seasonal patterns (higher sales during holidays)
/// - Random fluctuations due to unpredictable factors
/// 
/// Decomposing the data helps you see each of these patterns separately, making it easier to:
/// - Understand what's driving changes in your data
/// - Make better forecasts
/// - Identify unusual events or anomalies
/// </para>
/// </remarks>
public enum DecompositionComponentType
{
    /// <summary>
    /// The long-term progression or general direction of the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Trend component represents the long-term movement in the data, showing whether values are generally 
    /// increasing, decreasing, or staying stable over time.
    /// 
    /// Think of it as the "big picture" direction of your data when you ignore short-term fluctuations.
    /// 
    /// Examples:
    /// - Population growth showing a steady increase over decades
    /// - A company's revenue gradually increasing year over year
    /// - Global temperature rising slowly over many years
    /// 
    /// Identifying the trend helps you understand the fundamental direction of your data and make long-term forecasts.
    /// </para>
    /// </remarks>
    Trend,

    /// <summary>
    /// Repeating patterns or cycles with a fixed, known period (e.g., daily, weekly, monthly, yearly).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Seasonal component captures regular, predictable patterns that repeat at fixed intervals.
    /// 
    /// Think of it as patterns that occur at specific times, like:
    /// - Higher retail sales during December holidays
    /// - Higher ice cream sales in summer months
    /// - Higher website traffic during business hours
    /// - Lower energy usage on weekends
    /// 
    /// The key characteristic of seasonality is that it happens at known, fixed intervals (daily, weekly, monthly, quarterly, yearly).
    /// 
    /// Identifying seasonality helps you plan for predictable fluctuations and adjust your forecasts accordingly.
    /// </para>
    /// </remarks>
    Seasonal,

    /// <summary>
    /// The irregular variation or "noise" remaining after other components have been extracted.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Residual component (sometimes called "noise" or "error") represents the random, unpredictable fluctuations 
    /// in your data that can't be explained by trend, seasonality, or cycles.
    /// 
    /// Think of it as the "unexplained" part of your data - the random ups and downs that don't follow any pattern.
    /// 
    /// Examples of what might cause residuals:
    /// - Unexpected events (like a surprise promotion causing a sales spike)
    /// - Measurement errors
    /// - Random consumer behavior
    /// - Small factors that individually aren't significant enough to model
    /// 
    /// Analyzing residuals helps you:
    /// - Check if your decomposition captured all important patterns
    /// - Identify unusual events or outliers
    /// - Assess the randomness and unpredictability in your data
    /// </para>
    /// </remarks>
    Residual,

    /// <summary>
    /// Repeating patterns with a variable or changing period, unlike the fixed periods of seasonal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Cycle component represents repeating patterns that don't have a fixed time period.
    /// 
    /// Unlike seasonality (which occurs at fixed intervals like every December), cycles have varying lengths:
    /// - Business cycles (boom and bust) might last 2-10 years
    /// - Housing market cycles might expand and contract over varying timeframes
    /// - Sunspot activity follows cycles of approximately 11 years, but the exact length varies
    /// 
    /// Think of cycles as repeating patterns where you can't precisely predict when the next one will occur.
    /// 
    /// Identifying cycles helps you understand medium to long-term patterns that aren't tied to the calendar.
    /// </para>
    /// </remarks>
    Cycle,

    /// <summary>
    /// A combined component that includes both the long-term trend and cyclical patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The TrendCycle component combines the long-term trend and cyclical patterns into a single component.
    /// 
    /// This is often used when:
    /// - It's difficult to separate the trend from cycles
    /// - You're more interested in the overall direction including medium-term fluctuations
    /// - The decomposition method doesn't distinguish between trends and cycles
    /// 
    /// Think of it as the "smoothed" version of your data with short-term seasonality and noise removed,
    /// but keeping both the long-term direction and medium-term fluctuations.
    /// 
    /// For example, in economic data, the TrendCycle might show both the general economic growth (trend)
    /// and the business cycles of expansion and recession together.
    /// </para>
    /// </remarks>
    TrendCycle,

    /// <summary>
    /// Random, unpredictable fluctuations in the data (similar to Residual but used in specific decomposition methods).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Irregular component represents random variations that can't be attributed to trend, seasonality, or cycles.
    /// 
    /// This term is often used in specific decomposition methods (especially X-12-ARIMA and SEATS) and is 
    /// functionally similar to the Residual component.
    /// 
    /// Think of it as the unpredictable "noise" in your data after accounting for all identifiable patterns.
    /// 
    /// Analyzing the irregular component helps you:
    /// - Identify outliers or unusual events
    /// - Assess the volatility or unpredictability in your data
    /// - Evaluate how well your decomposition has captured the systematic patterns
    /// </para>
    /// </remarks>
    Irregular,

    /// <summary>
    /// Intrinsic Mode Functions - components extracted using Empirical Mode Decomposition (EMD) methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> IMF stands for Intrinsic Mode Function, which is a special type of component extracted 
    /// using a technique called Empirical Mode Decomposition (EMD).
    /// 
    /// Unlike traditional decomposition that separates data into trend, seasonality, and residuals, EMD breaks 
    /// down the data into a collection of IMFs, each representing oscillations at different time scales.
    /// 
    /// Think of IMFs as layers of waves with different frequencies:
    /// - Lower-order IMFs (IMF1, IMF2) capture fast oscillations (high-frequency components)
    /// - Higher-order IMFs capture slower oscillations (low-frequency components)
    /// - The final residual typically represents the overall trend
    /// 
    /// EMD and IMFs are particularly useful for:
    /// - Analyzing non-linear and non-stationary time series
    /// - Data where patterns change over time
    /// - Complex signals with multiple overlapping cycles
    /// 
    /// This approach is more advanced but can reveal patterns that traditional decomposition methods might miss.
    /// </para>
    /// </remarks>
    IMF
}
