namespace AiDotNet.Enums;

/// <summary>
/// Represents the overall performance quality of a machine learning model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This enum provides a simple way to categorize how well your AI model is performing.
/// Instead of dealing with complex numerical metrics, it gives you a straightforward assessment
/// that's easy to understand. The library automatically determines this rating based on various
/// statistical measures of your model's accuracy.
/// </para>
/// </remarks>
public enum ModelPerformance
{
    /// <summary>
    /// Indicates that the model performs well on the given data, with high accuracy and reliability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A "Good" performance typically means the model makes predictions that closely match the actual values,
    /// with metrics like R² generally above 0.8 (or equivalent thresholds for other metrics).
    /// </para>
    /// </remarks>
    Good,

    /// <summary>
    /// Indicates that the model performs acceptably but has room for improvement.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A "Moderate" performance suggests the model captures some patterns in the data but misses others.
    /// It's usable but could benefit from refinement, with metrics typically in the middle ranges
    /// (e.g., R² between 0.5 and 0.8).
    /// </para>
    /// </remarks>
    Moderate,

    /// <summary>
    /// Indicates that the model performs poorly and may not be suitable for the given data or task.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A "Poor" performance means the model struggles to make accurate predictions, with metrics showing
    /// low values (e.g., R² below 0.5). This could indicate that the model needs significant improvements,
    /// more training data, or that a different type of model might be more appropriate for the task.
    /// </para>
    /// </remarks>
    Poor
}
