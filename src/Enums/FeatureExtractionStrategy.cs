namespace AiDotNet.Enums;

/// <summary>
/// Defines strategies for extracting features from higher-dimensional tensors.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This defines different ways to handle complex data.
/// 
/// When data has multiple values for each feature (like pixels in an image),
/// we need a strategy to condense these into a single value for analysis.
/// Different strategies work better for different types of data.
/// </para>
/// </remarks>
public enum FeatureExtractionStrategy
{
    /// <summary>
    /// Uses the average value across all dimensions.
    /// </summary>
    Mean,

    /// <summary>
    /// Uses the maximum value across all dimensions.
    /// </summary>
    Max,

    /// <summary>
    /// Uses the first element as a representative value.
    /// </summary>
    Flatten,

    /// <summary>
    /// Uses a weighted sum with configurable weights.
    /// </summary>
    WeightedSum
}
