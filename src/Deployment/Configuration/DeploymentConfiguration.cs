namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Internal class that aggregates all deployment-related configurations.
/// Used to pass deployment settings from PredictionModelBuilder to PredictionModelResult.
/// </summary>
internal class DeploymentConfiguration
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
    /// Creates a deployment configuration from individual config objects.
    /// </summary>
    public static DeploymentConfiguration Create(
        QuantizationConfig? quantization,
        CacheConfig? caching,
        VersioningConfig? versioning,
        ABTestingConfig? abTesting,
        TelemetryConfig? telemetry,
        ExportConfig? export)
    {
        return new DeploymentConfiguration
        {
            Quantization = quantization,
            Caching = caching,
            Versioning = versioning,
            ABTesting = abTesting,
            Telemetry = telemetry,
            Export = export
        };
    }
}
