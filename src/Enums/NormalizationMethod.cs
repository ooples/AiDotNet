namespace AiDotNet.Enums;

/// <summary>
/// Defines different methods for normalizing data before processing in machine learning algorithms.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Normalization is like converting different measurements to a common scale. 
/// Imagine you have data about people's heights (in feet) and weights (in pounds) - these numbers 
/// are on very different scales. Normalization transforms all your data to similar ranges (like 0-1) 
/// so that one feature doesn't overwhelm others just because it uses bigger numbers. This helps 
/// machine learning algorithms work better and faster.
/// </para>
/// </remarks>
public enum NormalizationMethod
{
    /// <summary>
    /// No normalization is applied to the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This option means "don't change my data at all." Use this when your data 
    /// is already on a similar scale or when you specifically want to preserve the original values.
    /// </para>
    /// </remarks>
    None,

    /// <summary>
    /// Scales all values to a range between 0 and 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MinMax scaling takes your data and squeezes it to fit between 0 and 1. 
    /// The smallest value becomes 0, the largest becomes 1, and everything else falls in between. 
    /// It's like converting heights of people in a classroom to percentages of the tallest person.
    /// Formula: (x - min) / (max - min)
    /// </para>
    /// </remarks>
    MinMax,

    /// <summary>
    /// Standardizes values to have a mean of 0 and a standard deviation of 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Z-Score (also called standardization) centers your data around 0, with most values 
    /// falling between -3 and +3. It tells you how many "standard deviations" away from average each value is. 
    /// For example, a Z-Score of 2 means "this value is 2 standard deviations above average." This method 
    /// works well when your data follows a bell curve pattern.
    /// Formula: (x - mean) / standard_deviation
    /// </para>
    /// </remarks>
    ZScore,

    /// <summary>
    /// Scales values by dividing by powers of 10 to bring all values between 0 and 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decimal scaling moves the decimal point in your numbers until all values are between 
    /// -1 and 1. For example, if your largest number is 1,000, you'd divide everything by 1,000, so that 
    /// number becomes 1.0. This is useful when your data has different numbers of digits but is otherwise similar.
    /// Formula: x / 10^j where j is the smallest integer such that max(|x|) &lt; 1
    /// </para>
    /// </remarks>
    Decimal,

    /// <summary>
    /// Applies a logarithmic transformation to compress the range of values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Log transformation is useful when your data has a few very large values that would 
    /// otherwise dominate. It compresses high values while spreading out smaller values. For example, the 
    /// difference between 1 and 10 becomes the same as the difference between 10 and 100. This is often 
    /// used for data that grows exponentially, like population counts or prices.
    /// Formula: log(x) (Note: typically requires handling zero or negative values specially)
    /// </para>
    /// </remarks>
    Log,

    /// <summary>
    /// Groups continuous data into discrete bins or categories.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Binning is like grouping people by age ranges (0-10, 11-20, 21-30, etc.) instead of 
    /// using their exact ages. It converts continuous numbers into categorical groups. This can help reduce 
    /// noise in the data and reveal patterns that might be hidden when looking at exact values.
    /// </para>
    /// </remarks>
    Binning,

    /// <summary>
    /// Normalizes data by adjusting contrast across the entire dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Global Contrast Normalization is like adjusting the contrast on a photo to make 
    /// features more visible. It's especially useful for image data, where it helps highlight important 
    /// patterns while reducing the impact of variations in lighting or brightness.
    /// </para>
    /// </remarks>
    GlobalContrast,

    /// <summary>
    /// Applies logarithmic transformation followed by mean-variance normalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a two-step process: first, a log transformation compresses very large values 
    /// (like turning 1,000,000 into 6), then the result is standardized around a mean of 0. This is useful 
    /// for data with both exponential growth patterns and the need for standardization, such as financial 
    /// or scientific measurements that vary by orders of magnitude.
    /// </para>
    /// </remarks>
    LogMeanVariance,

    /// <summary>
    /// Normalizes data using the Lp norm (typically L1 or L2 norm).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Lp Norm scaling is like making sure the total "energy" of your data equals 1. 
    /// The most common version (L2 norm) is similar to making sure the length of a vector equals 1. 
    /// This is useful in text analysis and when working with high-dimensional data, as it helps compare 
    /// documents or data points of different lengths fairly.
    /// Formula for L2 norm: x / sqrt(sum(x^2))
    /// </para>
    /// </remarks>
    LpNorm,

    /// <summary>
    /// Standardizes data to have a specified mean and variance (typically mean=0, variance=1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean-Variance normalization is similar to Z-Score but gives you more control. 
    /// It adjusts your data to have a specific average (usually 0) and spread (usually 1). This makes 
    /// different features comparable and helps many machine learning algorithms perform better.
    /// Formula: (x - mean) / standard_deviation
    /// </para>
    /// </remarks>
    MeanVariance,

    /// <summary>
    /// Scales features using statistics that are robust to outliers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Robust Scaling is designed to handle data with outliers (extreme values that don't
    /// follow the pattern). Instead of using the minimum and maximum values (which can be skewed by outliers),
    /// it uses the median and quartiles. It's like saying "ignore the extremely tall and short people when
    /// standardizing heights." This is useful when your data contains unusual values that shouldn't influence
    /// the overall scaling.
    /// Formula: (x - median) / (Q3 - Q1) where Q1 is the 25th percentile and Q3 is the 75th percentile
    /// </para>
    /// </remarks>
    RobustScaling,

    /// <summary>
    /// Scales features to the range [-1, 1] by dividing by the maximum absolute value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MaxAbsScaler is like MinMax scaling, but instead of using both the minimum and
    /// maximum values, it only uses the maximum absolute value (the largest value ignoring signs). This
    /// method preserves zeros and the sign of values, which is important for sparse data where many values
    /// are zero. For example, if your largest value is 100 and smallest is -50, everything gets divided by 100,
    /// so results fall between -1 and 1.
    /// Formula: x / max(|x|)
    /// </para>
    /// </remarks>
    MaxAbsScaler,

    /// <summary>
    /// Transforms features to follow a uniform or normal distribution using quantiles.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> QuantileTransformer is a powerful technique that changes your data's distribution
    /// to be either uniform (flat, where all ranges have equal numbers of values) or normal (bell-shaped).
    /// It works by ranking values and mapping them to a target distribution. This is extremely effective at
    /// handling outliers because it spreads them out across the distribution. Think of it as redistributing
    /// your data so that it matches a desired pattern, regardless of the original distribution.
    /// </para>
    /// </remarks>
    QuantileTransformer
}
