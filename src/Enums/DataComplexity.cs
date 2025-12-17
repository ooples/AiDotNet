namespace AiDotNet.Enums;

/// <summary>
/// Represents the level of complexity in a dataset, which helps determine appropriate model selection and preprocessing.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Data complexity refers to how difficult it is for a machine learning model to find patterns in your data.
/// 
/// Think of it like solving puzzles:
/// - Simple data is like a basic jigsaw puzzle with few, large pieces
/// - Complex data is like an advanced puzzle with many tiny pieces and subtle patterns
/// 
/// Understanding your data's complexity helps you choose the right model:
/// - Simple data often works well with basic models
/// - Complex data usually requires more sophisticated models
/// 
/// This enum helps you categorize your data to make better decisions about which algorithms to use.
/// </para>
/// </remarks>
public enum DataComplexity
{
    /// <summary>
    /// Indicates data with clear patterns, few features, and minimal noise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Simple data typically has these characteristics:
    /// - Linear or easily separable patterns
    /// - Few relevant features (typically less than 10)
    /// - Clear relationships between inputs and outputs
    /// - Minimal noise or outliers
    /// - Little to no missing data
    /// - Balanced classes (for classification problems)
    /// 
    /// Examples include:
    /// - Height vs. weight relationships
    /// - Basic customer segmentation with clear groups
    /// - Simple time series with obvious trends
    /// 
    /// Simple data often works well with linear models, decision trees, or basic statistical methods.
    /// </para>
    /// </remarks>
    Simple,

    /// <summary>
    /// Indicates data with somewhat complex patterns, a moderate number of features, and some noise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Moderate complexity data typically has these characteristics:
    /// - Somewhat non-linear patterns
    /// - Moderate number of features (typically 10-50)
    /// - Some feature interactions
    /// - Moderate amount of noise
    /// - Some missing data
    /// - Slight class imbalance (for classification problems)
    /// 
    /// Examples include:
    /// - Customer purchase prediction with demographic and behavioral data
    /// - Housing price prediction with multiple factors
    /// - Seasonal time series with multiple patterns
    /// 
    /// Moderate complexity data often requires ensemble methods, shallow neural networks, or more sophisticated algorithms.
    /// </para>
    /// </remarks>
    Moderate,

    /// <summary>
    /// Indicates data with intricate patterns, many features, significant noise, or complex dependencies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Complex data typically has these characteristics:
    /// - Highly non-linear patterns
    /// - Many features (typically 50+)
    /// - Significant feature interactions
    /// - High dimensionality
    /// - Substantial noise or outliers
    /// - Significant missing data
    /// - Severe class imbalance (for classification problems)
    /// - Temporal or spatial dependencies
    /// 
    /// Examples include:
    /// - Natural language processing tasks
    /// - Image or video recognition
    /// - Financial market prediction
    /// - Genomic data analysis
    /// - Complex sensor data with multiple sources
    /// 
    /// Complex data often requires deep learning models, sophisticated feature engineering, or specialized algorithms 
    /// designed for the specific domain.
    /// </para>
    /// </remarks>
    Complex
}
