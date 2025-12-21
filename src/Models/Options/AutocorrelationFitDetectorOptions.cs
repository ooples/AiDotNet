namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for detecting autocorrelation in time series data and regression residuals.
/// </summary>
/// <remarks>
/// <para>
/// This class provides threshold values used to interpret the Durbin-Watson statistic, which measures
/// autocorrelation in the residuals (errors) of regression and time series models. The Durbin-Watson
/// statistic typically ranges from 0 to 4, with different values indicating different types of autocorrelation.
/// </para>
/// <para><b>For Beginners:</b> Autocorrelation is a pattern where data points in a time series are related to 
/// their own past values. Think of it like weather patterns - if it's been raining for several days, there's a 
/// higher chance it will rain tomorrow (positive autocorrelation). This class helps determine if your data has 
/// such patterns by setting thresholds for the Durbin-Watson test, which is like a thermometer for measuring 
/// autocorrelation. Values around 2 suggest no autocorrelation, values closer to 0 suggest positive autocorrelation 
/// (each value tends to be similar to previous values), and values closer to 4 suggest negative autocorrelation 
/// (each value tends to be opposite to previous values). Understanding autocorrelation helps choose the right 
/// prediction model for your data.</para>
/// </remarks>
public class AutocorrelationFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting strong positive autocorrelation.
    /// </summary>
    /// <value>The threshold value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Durbin-Watson statistic values below this threshold indicate strong positive autocorrelation in the data.
    /// Positive autocorrelation means that values tend to be followed by similar values.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a warning level for detecting patterns where each data point is 
    /// similar to the previous one. If the test result is below 1.0 (the default threshold), it suggests your data 
    /// has strong positive autocorrelation - meaning values tend to follow similar values (like warm days typically 
    /// following warm days). This pattern affects how you should analyze and predict your data, as simple models 
    /// might not capture this relationship correctly.</para>
    /// </remarks>
    public double StrongPositiveAutocorrelationThreshold { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the threshold for detecting strong negative autocorrelation.
    /// </summary>
    /// <value>The threshold value, defaulting to 3.0.</value>
    /// <remarks>
    /// <para>
    /// Durbin-Watson statistic values above this threshold indicate strong negative autocorrelation in the data.
    /// Negative autocorrelation means that values tend to be followed by opposite values.
    /// </para>
    /// <para><b>For Beginners:</b> This is a threshold for detecting patterns where each data point tends to be 
    /// opposite to the previous one. If the test result is above 3.0 (the default threshold), it suggests your data 
    /// has strong negative autocorrelation - meaning high values tend to be followed by low values and vice versa 
    /// (like a very hot day often being followed by a cooler day). This zigzag pattern requires special consideration 
    /// when building prediction models.</para>
    /// </remarks>
    public double StrongNegativeAutocorrelationThreshold { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the lower bound of the range indicating no autocorrelation.
    /// </summary>
    /// <value>The lower bound value, defaulting to 1.5.</value>
    /// <remarks>
    /// <para>
    /// Durbin-Watson statistic values above this threshold and below the upper bound suggest
    /// that there is no significant autocorrelation in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the minimum value for the "safe zone" where your data likely doesn't 
    /// have significant autocorrelation patterns. If your test result is between this lower bound (1.5 by default) 
    /// and the upper bound, it suggests your data points aren't strongly influenced by previous values. This is 
    /// often ideal for many standard statistical methods, as they typically assume independence between observations.</para>
    /// </remarks>
    public double NoAutocorrelationLowerBound { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the upper bound of the range indicating no autocorrelation.
    /// </summary>
    /// <value>The upper bound value, defaulting to 2.5.</value>
    /// <remarks>
    /// <para>
    /// Durbin-Watson statistic values below this threshold and above the lower bound suggest
    /// that there is no significant autocorrelation in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum value for the "safe zone" where your data likely doesn't 
    /// have significant autocorrelation patterns. If your test result is between the lower bound and this upper bound 
    /// (2.5 by default), it suggests your data points aren't strongly influenced by previous values. When your test 
    /// result falls in this range (approximately 1.5-2.5), you can generally proceed with standard statistical methods 
    /// without special adjustments for autocorrelation.</para>
    /// </remarks>
    public double NoAutocorrelationUpperBound { get; set; } = 2.5;
}
