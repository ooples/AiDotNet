namespace AiDotNet.Enums;

/// <summary>
/// Specifies the method used to sample or combine values when reducing data dimensions.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Sampling is how we summarize a group of numbers into a single value.
/// 
/// In AI, we often need to take a collection of values (like a grid of pixels in an image)
/// and represent them with fewer values. This process is called "downsampling" or "pooling".
/// 
/// Think of it like summarizing a neighborhood on a map:
/// - You could pick the tallest building (Max)
/// - You could calculate the average building height (Average)
/// - You could use a special mathematical formula (L2Norm)
/// 
/// Different sampling types give different results and are useful in different situations.
/// </remarks>
public enum SamplingType
{
    /// <summary>
    /// Takes the maximum value from the input region.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Max sampling simply picks the largest number from a group of values.
    /// 
    /// For example, if you have these numbers: [2, 5, 1, 3], Max sampling would give you 5.
    /// 
    /// This is commonly used in neural networks for:
    /// - Detecting if a feature is present anywhere in the region
    /// - Reducing the size of images while preserving important details
    /// - Making the model less sensitive to the exact position of features
    /// 
    /// Think of it like looking at a group of mountains and recording only the height of the tallest one.
    /// It's good at preserving strong signals and ignoring weaker ones.
    /// </remarks>
    Max,

    /// <summary>
    /// Takes the average (mean) value from the input region.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Average sampling calculates the mean of all values in a group.
    /// 
    /// For example, if you have these numbers: [2, 5, 1, 3], Average sampling would give you 2.75.
    /// 
    /// This is useful for:
    /// - Smoothing out noise in the data
    /// - Capturing the general trend of all values in the region
    /// - Reducing the impact of outliers or extreme values
    /// 
    /// Think of it like measuring the average temperature across a city instead of just the hottest spot.
    /// It gives you a more balanced representation of the entire region.
    /// </remarks>
    Average,

    /// <summary>
    /// Calculates the L2 norm (Euclidean norm) of the values in the input region.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> L2Norm sampling uses a special mathematical formula to combine values.
    /// 
    /// It works by:
    /// 1. Squaring each number
    /// 2. Adding up all the squared values
    /// 3. Taking the square root of the sum
    /// 
    /// For example, if you have these numbers: [2, 5, 1, 3], L2Norm sampling would give you:
    /// v(2² + 5² + 1² + 3²) = v(4 + 25 + 1 + 9) = v39 ˜ 6.24
    /// 
    /// This is useful for:
    /// - Measuring the overall "energy" or "strength" of a signal
    /// - Giving more weight to larger values without ignoring smaller ones
    /// - Certain specialized neural network architectures
    /// 
    /// Think of it like measuring how "impactful" a group of values is collectively,
    /// with larger values having more influence than smaller ones.
    /// </remarks>
    L2Norm
}
