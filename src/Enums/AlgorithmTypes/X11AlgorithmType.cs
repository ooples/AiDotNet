namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different variants of the X-11 seasonal adjustment algorithm used in time series analysis.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> X-11 is a statistical method used to remove seasonal patterns from time series data.
/// 
/// Imagine you run an ice cream shop and want to understand your true business growth. Your sales naturally 
/// spike in summer and drop in winter due to seasonal effects. The X-11 algorithm helps separate:
/// 
/// 1. The seasonal component (predictable patterns that repeat, like summer sales spikes)
/// 2. The trend component (your long-term growth or decline)
/// 3. The irregular component (random fluctuations that don't follow patterns)
/// 
/// By removing seasonal effects, you can see if your business is truly growing year-over-year, 
/// regardless of these predictable seasonal patterns.
/// 
/// X-11 was developed by the U.S. Census Bureau and is widely used by government agencies and 
/// businesses worldwide to produce "seasonally adjusted" economic indicators like unemployment rates, 
/// retail sales, and GDP figures that you might hear about in the news.
/// </para>
/// </remarks>
public enum X11AlgorithmType
{
    /// <summary>
    /// The standard implementation of the X-11 seasonal adjustment algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Standard X-11 approach uses an iterative process to gradually separate 
    /// a time series into its components.
    /// 
    /// It works like this:
    /// 
    /// 1. Makes an initial estimate of the trend using moving averages
    /// 2. Removes this trend to get preliminary seasonal and irregular components
    /// 3. Refines the seasonal factors by averaging across years
    /// 4. Applies these improved seasonal factors to get a better trend
    /// 5. Repeats these steps until the results stabilize
    /// 
    /// This method:
    /// - Works well for most economic and business time series
    /// - Handles both monthly and quarterly data
    /// - Automatically adjusts for outliers and extreme values
    /// - Produces results that are easy to interpret
    /// - Is the default choice for many statistical agencies
    /// </para>
    /// </remarks>
    Standard,

    /// <summary>
    /// A variant of X-11 that uses multiplicative adjustments for seasonal patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Multiplicative Adjustment approach assumes that seasonal patterns 
    /// grow or shrink proportionally with the overall level of the series.
    /// 
    /// For example, if your ice cream sales are generally $10,000 per month but increase by 50% in summer, 
    /// that's a multiplicative pattern (summer = regular sales × 1.5).
    /// 
    /// This method:
    /// 
    /// 1. Expresses seasonal factors as percentages or ratios (like "sales in July are 150% of average")
    /// 2. Works by dividing the original series by seasonal factors (rather than subtracting)
    /// 3. Is appropriate when the size of seasonal fluctuations changes proportionally with the level of the series
    /// 
    /// Multiplicative adjustment is best for:
    /// - Data that shows larger seasonal swings during periods of higher overall values
    /// - Many economic time series like retail sales, where seasonal patterns tend to grow with the economy
    /// - Series that can't go below zero (like sales figures)
    /// </para>
    /// </remarks>
    MultiplicativeAdjustment,

    /// <summary>
    /// A variant of X-11 that applies additive adjustments after logarithmic transformation of the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Log-Additive Adjustment is a hybrid approach that combines benefits of both 
    /// additive and multiplicative methods.
    /// 
    /// It works like this:
    /// 
    /// 1. First transforms the data by taking its logarithm (a mathematical operation that compresses large values)
    /// 2. Applies additive adjustments to this transformed data
    /// 3. Then transforms back to the original scale using the exponential function (reverse of logarithm)
    /// 
    /// This approach:
    /// - Handles data with changing seasonal patterns better than purely additive methods
    /// - Is more stable than purely multiplicative methods when values are close to zero
    /// - Works well for data with exponential growth trends
    /// - Is particularly useful for economic time series with strong growth
    /// - Helps manage extreme values and outliers effectively
    /// 
    /// Log-Additive adjustment is often a good compromise when neither purely additive nor 
    /// purely multiplicative models seem appropriate.
    /// </para>
    /// </remarks>
    LogAdditiveAdjustment
}
