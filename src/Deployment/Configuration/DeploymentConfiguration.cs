namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Aggregates all deployment-related configurations.
/// Used to pass deployment settings from PredictionModelBuilder to PredictionModelResult.
/// </summary>
public class DeploymentConfiguration
{
    /// <summary>
    /// Gets or sets the quantization configuration (null = no quantization).
    /// </summary>
    public QuantizationConfig? Quantization { get; set; }

    /// <summary>
    /// Gets or sets the caching configuration (null = use defaults).
    /// </summary>
    public CacheConfig? Caching { get; set; }

    /// <summary>
    /// Gets or sets the versioning configuration (null = use defaults).
    /// </summary>
    public VersioningConfig? Versioning { get; set; }

    /// <summary>
    /// Gets or sets the A/B testing configuration (null = disabled).
    /// </summary>
    public ABTestingConfig? ABTesting { get; set; }

    /// <summary>
    /// Gets or sets the telemetry configuration (null = use defaults).
    /// </summary>
    public TelemetryConfig? Telemetry { get; set; }

    /// <summary>
    /// Gets or sets the export configuration (null = use defaults).
    /// </summary>
    public ExportConfig? Export { get; set; }

    /// <summary>
    /// Gets or sets the GPU acceleration configuration (null = use defaults).
    /// </summary>
    public GpuAccelerationConfig? GpuAcceleration { get; set; }

    /// <summary>
    /// Gets or sets the compression configuration (null = no compression).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When configured, compression is automatically applied during
    /// model serialization (saving) and reversed during deserialization (loading).
    /// This reduces model file sizes by 50-90% with minimal accuracy impact.
    /// </para>
    /// </remarks>
    public CompressionConfig? Compression { get; set; }

    /// <summary>
    /// Gets or sets the profiling configuration (null = disabled).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When configured, profiling measures the performance of training
    /// and inference operations. The profiling report will be available in PredictionModelResult.
    /// </para>
    /// </remarks>
    public ProfilingConfig? Profiling { get; set; }

    /// <summary>
    /// Creates a deployment configuration from individual config objects.
    /// </summary>
    public static DeploymentConfiguration Create(
        QuantizationConfig? quantization,
        CacheConfig? caching,
        VersioningConfig? versioning,
        ABTestingConfig? abTesting,
        TelemetryConfig? telemetry,
        ExportConfig? export,
        GpuAccelerationConfig? gpuAcceleration,
        CompressionConfig? compression = null,
        ProfilingConfig? profiling = null)
    {
        return new DeploymentConfiguration
        {
            Quantization = quantization,
            Caching = caching,
            Versioning = versioning,
            ABTesting = abTesting,
            Telemetry = telemetry,
            Export = export,
            GpuAcceleration = gpuAcceleration,
            Compression = compression,
            Profiling = profiling
        };
    }
}
