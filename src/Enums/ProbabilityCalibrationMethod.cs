namespace AiDotNet.Enums;

/// <summary>
/// Defines probability calibration strategies for classification-like outputs.
/// </summary>
/// <remarks>
/// <para>
/// Calibration transforms predicted probabilities to better reflect empirical correctness likelihoods.
/// </para>
/// <para><b>For Beginners:</b> Calibration helps ensure that "80% confident" means "correct about 80% of the time".</para>
/// </remarks>
public enum ProbabilityCalibrationMethod
{
    /// <summary>
    /// Automatically selects a suitable calibration method based on the task and output shape.
    /// </summary>
    Auto,

    /// <summary>
    /// Disables probability calibration.
    /// </summary>
    None,

    /// <summary>
    /// Uses temperature scaling (typically best for multiclass neural network probabilities/logits).
    /// </summary>
    TemperatureScaling,

    /// <summary>
    /// Uses Platt scaling (logistic calibration, typically best for binary classification).
    /// </summary>
    PlattScaling,

    /// <summary>
    /// Uses isotonic regression calibration (non-parametric monotonic calibration, typically for binary classification).
    /// </summary>
    IsotonicRegression
}

