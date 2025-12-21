using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.Results;

/// <summary>
/// Stores uncertainty-calibration artifacts computed at build-time (not raw calibration data).
/// </summary>
/// <remarks>
/// This is internal by design to avoid expanding the public API surface area and to keep implementation details hidden.
/// </remarks>
internal sealed class UncertaintyCalibrationArtifacts<T>
{
    internal bool HasConformalRegression { get; set; }
    internal T ConformalRegressionQuantile { get; set; } = default!;

    internal bool HasConformalClassification { get; set; }
    internal T ConformalClassificationThreshold { get; set; } = default!;
    internal int ConformalClassificationNumClasses { get; set; }

    internal bool HasAdaptiveConformalClassification { get; set; }
    internal double[]? ConformalClassificationAdaptiveBinEdges { get; set; }
    internal Vector<T>? ConformalClassificationAdaptiveThresholds { get; set; }

    internal bool HasTemperatureScaling { get; set; }
    internal T TemperatureScalingTemperature { get; set; } = default!;

    internal bool HasPlattScaling { get; set; }
    internal Vector<T>? PlattScalingA { get; set; }
    internal Vector<T>? PlattScalingB { get; set; }

    internal bool HasIsotonicRegressionCalibration { get; set; }
    internal Vector<T>? IsotonicCalibrationX { get; set; }
    internal Vector<T>? IsotonicCalibrationY { get; set; }

    internal bool HasLaplacePosterior { get; set; }
    internal Vector<T>? LaplacePosteriorMean { get; set; }
    internal Vector<T>? LaplacePosteriorVarianceDiag { get; set; }

    internal bool HasSwagPosterior { get; set; }
    internal Vector<T>? SwagPosteriorMean { get; set; }
    internal Vector<T>? SwagPosteriorVarianceDiag { get; set; }

    internal bool HasExpectedCalibrationError { get; set; }
    internal T ExpectedCalibrationError { get; set; } = default!;
}
