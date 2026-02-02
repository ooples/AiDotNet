namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies whether higher or lower values are better for a metric.
/// </summary>
/// <remarks>
/// <para>
/// Different metrics have different optimization directions:
/// <list type="bullet">
/// <item>Accuracy, F1, AUC → Higher is better</item>
/// <item>Error, Loss, MSE → Lower is better</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> This tells the system how to interpret a metric value:
/// <list type="bullet">
/// <item><b>Higher is better:</b> 0.95 accuracy is better than 0.80</item>
/// <item><b>Lower is better:</b> 0.05 error is better than 0.20</item>
/// </list>
/// Knowing this is essential for model comparison, early stopping, and hyperparameter tuning.
/// </para>
/// </remarks>
public enum MetricDirection
{
    /// <summary>
    /// Higher values indicate better performance.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Maximize this metric. Examples:</para>
    /// <list type="bullet">
    /// <item>Accuracy (higher = more correct predictions)</item>
    /// <item>Precision (higher = fewer false positives)</item>
    /// <item>Recall (higher = fewer false negatives)</item>
    /// <item>F1 Score (higher = better balance)</item>
    /// <item>AUC-ROC (higher = better discrimination)</item>
    /// <item>R² (higher = better fit)</item>
    /// </list>
    /// </remarks>
    HigherIsBetter = 0,

    /// <summary>
    /// Lower values indicate better performance.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Minimize this metric. Examples:</para>
    /// <list type="bullet">
    /// <item>Mean Squared Error (lower = smaller errors)</item>
    /// <item>Mean Absolute Error (lower = smaller errors)</item>
    /// <item>Log Loss (lower = better calibration)</item>
    /// <item>Brier Score (lower = better probabilities)</item>
    /// <item>Expected Calibration Error (lower = better calibrated)</item>
    /// <item>False Positive Rate (lower = fewer false alarms)</item>
    /// </list>
    /// </remarks>
    LowerIsBetter = 1,

    /// <summary>
    /// Target a specific value (e.g., calibration metrics targeting 0 or 1).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Aim for a specific target value. Examples:</para>
    /// <list type="bullet">
    /// <item>Residual mean (target: 0, meaning unbiased predictions)</item>
    /// <item>Hosmer-Lemeshow p-value (target: > 0.05 for good calibration)</item>
    /// <item>Durbin-Watson (target: ~2 for no autocorrelation)</item>
    /// </list>
    /// </remarks>
    TargetValue = 2,

    /// <summary>
    /// Direction depends on context or is not applicable.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some metrics don't have a clear better/worse direction:</para>
    /// <list type="bullet">
    /// <item>Feature importance (just shows which features matter)</item>
    /// <item>Prediction counts (descriptive, not evaluative)</item>
    /// <item>Correlation coefficients (depends on expected sign)</item>
    /// </list>
    /// </remarks>
    NotApplicable = 3
}
