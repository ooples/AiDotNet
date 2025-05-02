namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of statistical intervals for predictions.
/// </summary>
/// <remarks>
/// <para>
/// This enum provides a comprehensive list of all supported interval types,
/// each representing a different statistical approach to quantifying prediction uncertainty.
/// </para>
/// <para>
/// <b>For Beginners:</b> Different interval types tell you different things about the uncertainty
/// in your predictions. For example, prediction intervals tell you where individual future values
/// are likely to fall, while confidence intervals tell you about the uncertainty in the average prediction.
/// </para>
/// </remarks>
public enum IntervalType
{
    /// <summary>
    /// A range that likely contains future individual observations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A prediction interval gives you a range where future individual values 
    /// are likely to fall. For example, if your model predicts a value of 100 and the prediction 
    /// interval is (90, 110), you can be reasonably confident that the actual value will be between 
    /// 90 and 110.
    /// </para>
    /// <para>
    /// Unlike confidence intervals (which are about the average), prediction intervals account for
    /// both the uncertainty in the average prediction and the natural variability of individual values.
    /// </para>
    /// </remarks>
    Prediction,

    /// <summary>
    /// A range that likely contains the true mean of the predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A confidence interval tells you the range where the true average prediction 
    /// is likely to be. It helps you understand the precision of your model's average prediction.
    /// </para>
    /// <para>
    /// For example, if your model predicts an average of 50 and the confidence interval is (48, 52),
    /// you can be reasonably confident that the true average is between 48 and 52.
    /// The interval gets narrower with more data, indicating more precise estimates.
    /// </para>
    /// </remarks>
    Confidence,

    /// <summary>
    /// A Bayesian interval that contains the true value with a certain probability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A credible interval is the Bayesian version of a confidence interval. 
    /// While they look similar, they have different interpretations.
    /// </para>
    /// <para>
    /// You can say "There's a 95% chance that the true value lies within this credible interval,"
    /// which is a more intuitive interpretation than confidence intervals provide.
    /// </para>
    /// <para>
    /// Credible intervals incorporate prior knowledge about the parameter being estimated,
    /// which can be beneficial when you have domain expertise or previous data.
    /// </para>
    /// </remarks>
    Credible,

    /// <summary>
    /// A range that contains a specified proportion of the population with a certain confidence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tolerance interval is different from both prediction and confidence intervals.
    /// It gives you a range that contains a specific percentage of all possible values
    /// (not just the average) with a certain level of confidence.
    /// </para>
    /// <para>
    /// For example, a 95/99 tolerance interval means you're 95% confident that the interval
    /// contains 99% of all possible values from the population.
    /// </para>
    /// <para>
    /// These are useful when you need to understand the range of almost all possible values,
    /// such as in quality control or setting specification limits.
    /// </para>
    /// </remarks>
    Tolerance,

    /// <summary>
    /// A prediction interval specifically for time series forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forecast intervals are similar to prediction intervals but are specifically 
    /// designed for time series data (data collected over time, like monthly sales or daily temperatures).
    /// </para>
    /// <para>
    /// They account for the unique characteristics of time series data, like increasing uncertainty
    /// the further you forecast into the future and potential seasonal patterns.
    /// </para>
    /// <para>
    /// A wider forecast interval indicates less certainty about future values.
    /// </para>
    /// </remarks>
    Forecast,

    /// <summary>
    /// An interval created using resampling techniques.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bootstrap intervals use a technique called "bootstrapping," where many samples 
    /// are randomly drawn (with replacement) from your data to estimate the variability in your predictions.
    /// </para>
    /// <para>
    /// This approach is powerful because it doesn't assume any particular distribution for your data,
    /// making it robust when your data doesn't follow common patterns like the normal distribution.
    /// </para>
    /// <para>
    /// Bootstrap intervals are especially useful when you have limited data or when the theoretical
    /// assumptions for other interval types might not be valid.
    /// </para>
    /// </remarks>
    Bootstrap,

    /// <summary>
    /// A prediction interval that accounts for multiple predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you make multiple predictions at once, there's a higher chance that 
    /// at least one of them will fall outside a standard prediction interval just by random chance.
    /// </para>
    /// <para>
    /// Simultaneous prediction intervals account for this by creating wider intervals that are guaranteed
    /// to contain a certain percentage of all predictions made simultaneously.
    /// </para>
    /// <para>
    /// These are important when you need to ensure that all (or most) of your predictions are within
    /// the intervals, not just each one individually.
    /// </para>
    /// </remarks>
    SimultaneousPrediction,

    /// <summary>
    /// An interval created by systematically leaving out one observation at a time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The jackknife method creates an interval by repeatedly recalculating 
    /// your statistics while leaving out one data point each time. This helps assess how sensitive 
    /// your results are to individual data points.
    /// </para>
    /// <para>
    /// It's particularly useful for detecting outliers or influential points that might be
    /// skewing your results, and for creating intervals when sample sizes are small.
    /// </para>
    /// <para>
    /// Like bootstrap intervals, jackknife intervals don't make strong assumptions about
    /// the distribution of your data.
    /// </para>
    /// </remarks>
    Jackknife,

    /// <summary>
    /// An interval based directly on the percentiles of the prediction distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A percentile interval is one of the simplest types of intervals, created 
    /// by taking the percentiles directly from your prediction distribution.
    /// </para>
    /// <para>
    /// For example, a 95% percentile interval might use the 2.5th and 97.5th percentiles
    /// of your predictions as the lower and upper bounds.
    /// </para>
    /// <para>
    /// These intervals are intuitive and don't require many statistical assumptions,
    /// making them useful for quick assessments of your prediction range.
    /// </para>
    /// </remarks>
    Percentile
}