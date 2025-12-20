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

    internal bool HasTemperatureScaling { get; set; }
    internal T TemperatureScalingTemperature { get; set; } = default!;

    internal bool HasExpectedCalibrationError { get; set; }
    internal T ExpectedCalibrationError { get; set; } = default!;
}
