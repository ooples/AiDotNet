namespace AiDotNet.Enums;

/// <summary>
/// Defines different methods for detecting outliers in datasets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Outliers are data points that differ significantly from other observations in your dataset.
/// Think of them as unusual values that stand out from the pattern - like if most people in a classroom are 
/// between 5'0" and 6'0" tall, someone who is 7'5" would be an outlier. Detecting outliers is important because 
/// they can skew your analysis or cause your AI model to learn incorrect patterns. These methods help you 
/// identify which data points might be outliers so you can decide whether to investigate them further or 
/// remove them from your analysis.
/// </para>
/// </remarks>
public enum OutlierDetectionMethod
{
    /// <summary>
    /// Detects outliers based on how many standard deviations a value is from the mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Z-Score method measures how far away a data point is from the average (mean) 
    /// in terms of standard deviations. Standard deviation is just a measure of how spread out the data is.
    /// 
    /// Imagine a classroom where the average height is 5'6" with a standard deviation of 2 inches. 
    /// Using Z-Score:
    /// - Someone 5'10" tall has a Z-Score of +2 (2 standard deviations above average)
    /// - Someone 5'2" tall has a Z-Score of -2 (2 standard deviations below average)
    /// - Someone 6'4" tall has a Z-Score of +5 (5 standard deviations above average) - likely an outlier!
    /// 
    /// Typically, values with Z-Scores beyond +/-3 are considered potential outliers. This method works 
    /// best when your data follows a bell curve (normal distribution).
    /// </para>
    /// </remarks>
    ZScore,

    /// <summary>
    /// Detects outliers using the Interquartile Range method, which is resistant to extreme values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The IQR (Interquartile Range) method is like focusing on the middle 50% of your data 
    /// and then identifying values that fall too far outside this range.
    /// 
    /// Here's how it works:
    /// 1. Arrange all your data from smallest to largest
    /// 2. Find Q1 (the value 25% of the way through your sorted data)
    /// 3. Find Q3 (the value 75% of the way through your sorted data)
    /// 4. Calculate IQR = Q3 - Q1
    /// 5. Any value below Q1 - 1.5×IQR or above Q3 + 1.5×IQR is considered an outlier
    /// 
    /// The advantage of IQR is that it's not affected by extreme values, making it more robust than Z-Score 
    /// for skewed data or when you're not sure if your data follows a bell curve.
    /// </para>
    /// </remarks>
    IQR,

    /// <summary>
    /// Uses both Z-Score and IQR methods together to identify outliers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Combined method uses both Z-Score and IQR approaches together for more reliable 
    /// outlier detection. A data point is flagged as an outlier only if both methods identify it as unusual.
    /// 
    /// This is like getting a second opinion - if only one doctor thinks something is wrong but another 
    /// doesn't, you might want more information before taking action. But if both doctors agree there's an 
    /// issue, you can be more confident.
    /// 
    /// The Combined approach reduces the chance of incorrectly flagging normal data points as outliers, 
    /// making it more conservative but potentially more accurate, especially when you're not sure which 
    /// method is best for your specific data.
    /// </para>
    /// </remarks>
    Combined
}
