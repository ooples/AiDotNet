namespace AiDotNet.Enums;

/// <summary>
/// Specifies the target distribution for quantile transformation.
/// </summary>
/// <remarks>
/// <para>
/// This enum defines the available output distributions for the QuantileTransformer.
/// Each distribution has different characteristics and use cases in machine learning.
/// </para>
/// <para><b>For Beginners:</b> Think of this as choosing the shape you want your data to take:
/// - Uniform: Spreads values evenly across the range (like a flat distribution)
/// - Normal: Creates a bell curve pattern (most values in the middle, fewer at extremes)
/// </para>
/// </remarks>
public enum OutputDistribution
{
    /// <summary>
    /// Maps data to a uniform distribution where all values are equally likely.
    /// Values are spread evenly across the [0, 1] range.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This makes your data look like a flat line - every value
    /// is equally common. Good for algorithms that don't assume any particular distribution.
    /// </para>
    /// <para>
    /// Use this when:
    /// - You want to reduce the impact of outliers
    /// - Your algorithm works best with uniformly distributed features
    /// - You want a simple, predictable transformation
    /// </para>
    /// </remarks>
    Uniform,

    /// <summary>
    /// Maps data to a normal (Gaussian) distribution with mean 0 and standard deviation 1.
    /// Values follow a bell curve pattern.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This makes your data look like a bell curve - most values
    /// are near the middle, with fewer values at the extremes. Many statistical methods
    /// assume this distribution.
    /// </para>
    /// <para>
    /// Use this when:
    /// - Your algorithm assumes normally distributed features (e.g., linear regression, LDA)
    /// - You want to reduce the impact of outliers while maintaining statistical properties
    /// - You need compatibility with methods that expect Gaussian distributions
    /// </para>
    /// </remarks>
    Normal
}
