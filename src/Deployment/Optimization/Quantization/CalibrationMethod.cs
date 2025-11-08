namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// Calibration methods for quantization.
/// </summary>
public enum CalibrationMethod
{
    /// <summary>No calibration</summary>
    None,

    /// <summary>Min-Max calibration (simple range-based)</summary>
    MinMax,

    /// <summary>Histogram-based calibration (percentile-based)</summary>
    Histogram,

    /// <summary>Entropy-based calibration (KL divergence)</summary>
    Entropy,

    /// <summary>Mean Squared Error (MSE) based calibration</summary>
    MSE,

    /// <summary>Percentile-based calibration</summary>
    Percentile
}
