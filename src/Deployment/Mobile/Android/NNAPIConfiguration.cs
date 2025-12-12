namespace AiDotNet.Deployment.Mobile.Android;

/// <summary>
/// Configuration for NNAPI backend.
/// </summary>
public class NNAPIConfiguration
{
    /// <summary>
    /// Gets or sets the preferred acceleration device.
    /// </summary>
    public NNAPIDevice PreferredDevice { get; set; } = NNAPIDevice.Auto;

    /// <summary>
    /// Gets or sets whether to allow fallback to CPU (default: true).
    /// </summary>
    public bool AllowCpuFallback { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use relaxed float32 precision (default: true).
    /// </summary>
    public bool UseRelaxedFloat32 { get; set; } = true;

    /// <summary>
    /// Gets or sets the execution preference.
    /// </summary>
    public NNAPIExecutionPreference ExecutionPreference { get; set; } = NNAPIExecutionPreference.FastSingleAnswer;

    /// <summary>
    /// Gets or sets the maximum number of concurrent executions (default: 1).
    /// </summary>
    public int MaxConcurrentExecutions { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to cache compiled models (default: true).
    /// </summary>
    public bool EnableModelCaching { get; set; } = true;

    /// <summary>
    /// Gets or sets the model cache directory.
    /// </summary>
    public string? ModelCacheDirectory { get; set; }

    /// <summary>
    /// Creates a configuration for maximum performance.
    /// </summary>
    public static NNAPIConfiguration ForMaxPerformance()
    {
        return new NNAPIConfiguration
        {
            PreferredDevice = NNAPIDevice.GPU,
            ExecutionPreference = NNAPIExecutionPreference.FastSingleAnswer,
            UseRelaxedFloat32 = true,
            AllowCpuFallback = false
        };
    }

    /// <summary>
    /// Creates a configuration for low power consumption.
    /// </summary>
    public static NNAPIConfiguration ForLowPower()
    {
        return new NNAPIConfiguration
        {
            PreferredDevice = NNAPIDevice.DSP,
            ExecutionPreference = NNAPIExecutionPreference.SustainedSpeed,
            UseRelaxedFloat32 = true,
            AllowCpuFallback = true
        };
    }
}
