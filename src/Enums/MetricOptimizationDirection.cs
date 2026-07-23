namespace AiDotNet.Enums;

/// <summary>
/// Specifies the direction for metric optimization (whether lower or higher values are better).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When tracking metrics during training, you need to specify whether
/// you want to minimize the metric (lower is better, like loss) or maximize it (higher is better,
/// like accuracy). This enum lets you tell the system which direction represents improvement.
///
/// Examples:
/// - Loss functions: Use Minimize (you want loss to go DOWN)
/// - Accuracy: Use Maximize (you want accuracy to go UP)
/// - Error rate: Use Minimize (you want errors to go DOWN)
/// - F1 score: Use Maximize (you want F1 to go UP)
/// </remarks>
public enum MetricOptimizationDirection
{
    /// <summary>
    /// Lower metric values are better (e.g., loss, error rate, MSE).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this when tracking metrics where smaller numbers indicate
    /// better performance. For example, if you're tracking loss and it goes from 0.5 to 0.3,
    /// that's an improvement.
    /// </remarks>
    Minimize = 0,

    /// <summary>
    /// Higher metric values are better (e.g., accuracy, F1 score, AUC).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this when tracking metrics where larger numbers indicate
    /// better performance. For example, if accuracy goes from 85% to 90%, that's an improvement.
    /// </remarks>
    Maximize = 1
}
