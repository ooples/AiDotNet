namespace AiDotNet.Enums;

/// <summary>
/// Defines conformal prediction calibration modes.
/// </summary>
/// <remarks>
/// <para>
/// Conformal prediction produces prediction sets (classification) or intervals (regression) with statistical guarantees under exchangeability.
/// Different modes trade off compute cost, stability, and adaptivity.
/// </para>
/// </remarks>
public enum ConformalPredictionMode
{
    /// <summary>
    /// Standard split conformal calibration using a single calibration set.
    /// </summary>
    Split,

    /// <summary>
    /// Cross-conformal style calibration using K folds of a single calibration set.
    /// </summary>
    CrossConformal,

    /// <summary>
    /// Adaptive conformal calibration that adjusts thresholds based on confidence buckets.
    /// </summary>
    Adaptive
}

