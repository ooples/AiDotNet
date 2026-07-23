namespace AiDotNet.Enums;

/// <summary>
/// Specifies whether to use an upper or lower envelope in signal processing and data analysis operations.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An envelope in data analysis is like drawing a line that follows the peaks or valleys of your data.
/// 
/// Think of it like tracing the outline of a mountain range:
/// - The upper envelope follows the tops of the mountains (the highest points)
/// - The lower envelope follows the bottoms of the valleys (the lowest points)
/// 
/// Envelopes are useful for:
/// - Identifying trends in noisy data
/// - Finding the boundaries of oscillating signals
/// - Detecting peaks and valleys in time series data
/// - Creating confidence intervals around predictions
/// 
/// For example, if you have stock price data that goes up and down, the upper envelope would connect all the highest prices,
/// while the lower envelope would connect all the lowest prices.
/// </para>
/// </remarks>
public enum EnvelopeType
{
    /// <summary>
    /// Represents the upper envelope that follows the maximum values or peaks in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The upper envelope connects the highest points in a dataset or signal.
    /// 
    /// Think of it as drawing a smooth line that touches only the peaks of your data,
    /// like tracing the tops of mountains in a mountain range.
    /// 
    /// Use cases:
    /// - Finding the maximum amplitude of an oscillating signal
    /// - Creating upper confidence bounds for predictions
    /// - Identifying resistance levels in financial data
    /// - Detecting the highest points in cyclical data
    /// 
    /// For example, in audio processing, the upper envelope might represent the maximum volume levels over time.
    /// </para>
    /// </remarks>
    Upper,

    /// <summary>
    /// Represents the lower envelope that follows the minimum values or valleys in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The lower envelope connects the lowest points in a dataset or signal.
    /// 
    /// Think of it as drawing a smooth line that touches only the valleys of your data,
    /// like tracing the bottoms of valleys in a mountain range.
    /// 
    /// Use cases:
    /// - Finding the minimum amplitude of an oscillating signal
    /// - Creating lower confidence bounds for predictions
    /// - Identifying support levels in financial data
    /// - Detecting the lowest points in cyclical data
    /// 
    /// For example, in environmental monitoring, the lower envelope might represent the minimum temperature readings over a season.
    /// </para>
    /// </remarks>
    Lower
}
