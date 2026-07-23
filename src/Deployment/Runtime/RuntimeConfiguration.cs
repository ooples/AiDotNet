namespace AiDotNet.Deployment.Runtime;

/// <summary>
/// Configuration for the deployment runtime environment.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> RuntimeConfiguration provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class RuntimeConfiguration
{
    /// <summary>
    /// Gets or sets whether to enable telemetry collection (default: true).
    /// </summary>
    public bool EnableTelemetry { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable model caching (default: true).
    /// </summary>
    public bool EnableCaching { get; set; } = true;

    /// <summary>
    /// Gets or sets the cache size in megabytes (default: 100 MB).
    /// </summary>
    public double CacheSizeMB { get; set; } = 100.0;

    /// <summary>
    /// Gets or sets the cache eviction policy.
    /// </summary>
    public Enums.CacheEvictionPolicy CacheEvictionPolicy { get; set; } = Enums.CacheEvictionPolicy.LRU;

    /// <summary>
    /// Gets or sets whether to enable automatic model warm-up on registration (default: true).
    /// </summary>
    public bool AutoWarmUp { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of warm-up iterations (default: 10).
    /// </summary>
    public int WarmUpIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to enable versioning (default: true).
    /// </summary>
    public bool EnableVersioning { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of versions to keep per model (default: 3).
    /// </summary>
    public int MaxVersionsPerModel { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to enable A/B testing support (default: false).
    /// </summary>
    public bool EnableABTesting { get; set; } = false;

    /// <summary>
    /// Gets or sets the telemetry sampling rate (0.0 to 1.0, default: 1.0 = 100%).
    /// </summary>
    public double TelemetrySamplingRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the telemetry buffer size (default: 1000 events).
    /// </summary>
    public int TelemetryBufferSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the telemetry flush interval in seconds (default: 60).
    /// </summary>
    public int TelemetryFlushIntervalSeconds { get; set; } = 60;

    /// <summary>
    /// Gets or sets whether to enable performance monitoring (default: true).
    /// </summary>
    public bool EnablePerformanceMonitoring { get; set; } = true;

    /// <summary>
    /// Gets or sets the performance alert threshold in milliseconds (default: 1000).
    /// </summary>
    public double PerformanceAlertThresholdMs { get; set; } = 1000.0;

    /// <summary>
    /// Gets or sets whether to enable health checks (default: true).
    /// </summary>
    public bool EnableHealthChecks { get; set; } = true;

    /// <summary>
    /// Gets or sets the health check interval in seconds (default: 300 = 5 minutes).
    /// </summary>
    public int HealthCheckIntervalSeconds { get; set; } = 300;

    /// <summary>
    /// Gets or sets the model load timeout in seconds (default: 60).
    /// </summary>
    public int ModelLoadTimeoutSeconds { get; set; } = 60;

    /// <summary>
    /// Gets or sets the inference timeout in seconds (default: 30).
    /// </summary>
    public int InferenceTimeoutSeconds { get; set; } = 30;

    /// <summary>
    /// Gets or sets whether to enable GPU acceleration for inference (default: true).
    /// Falls back to CPU if GPU is not available.
    /// </summary>
    public bool EnableGpuAcceleration { get; set; } = true;

    /// <summary>
    /// Creates a configuration for production deployment.
    /// </summary>
    public static RuntimeConfiguration ForProduction()
    {
        return new RuntimeConfiguration
        {
            EnableTelemetry = true,
            EnableCaching = true,
            CacheSizeMB = 500.0,
            AutoWarmUp = true,
            WarmUpIterations = 20,
            EnableVersioning = true,
            MaxVersionsPerModel = 5,
            EnableABTesting = true,
            TelemetrySamplingRate = 0.1, // Sample 10% in production
            EnablePerformanceMonitoring = true,
            PerformanceAlertThresholdMs = 500.0,
            EnableHealthChecks = true
        };
    }

    /// <summary>
    /// Creates a configuration for development/testing.
    /// </summary>
    public static RuntimeConfiguration ForDevelopment()
    {
        return new RuntimeConfiguration
        {
            EnableTelemetry = true,
            EnableCaching = true,
            CacheSizeMB = 100.0,
            AutoWarmUp = false,
            EnableVersioning = true,
            MaxVersionsPerModel = 2,
            EnableABTesting = false,
            TelemetrySamplingRate = 1.0, // Sample all in development
            EnablePerformanceMonitoring = true,
            EnableHealthChecks = false
        };
    }

    /// <summary>
    /// Creates a minimal configuration for edge devices.
    /// </summary>
    public static RuntimeConfiguration ForEdge()
    {
        return new RuntimeConfiguration
        {
            EnableTelemetry = true,
            TelemetrySamplingRate = 0.01, // Sample 1% on edge
            EnableCaching = true,
            CacheSizeMB = 10.0,
            AutoWarmUp = true,
            WarmUpIterations = 5,
            EnableVersioning = false,
            EnableABTesting = false,
            EnablePerformanceMonitoring = false,
            EnableHealthChecks = false
        };
    }
}
