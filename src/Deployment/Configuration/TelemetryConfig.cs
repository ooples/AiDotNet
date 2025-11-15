namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for telemetry - tracking and monitoring model inference metrics.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Telemetry collects data about how your model is performing in production.
/// Think of it like a fitness tracker for your AI model - it tracks important health metrics so
/// you know when something goes wrong or when performance degrades.
///
/// **What gets tracked:**
/// - **Latency**: How long each inference takes (helps identify slowdowns)
/// - **Throughput**: How many inferences per second (measures capacity)
/// - **Errors**: When predictions fail (helps identify issues)
/// - **Cache hits/misses**: How often cached models are used (optimizes memory)
/// - **Version usage**: Which model versions are being used (helps with rollouts)
///
/// **Why it's important:**
/// - Detect performance degradation before users complain
/// - Understand usage patterns to optimize resources
/// - Debug production issues with real data
/// - Make data-driven decisions about model updates
///
/// **Privacy Note:** Telemetry doesn't collect user data or predictions, only metadata
/// like timing and version information.
/// </remarks>
public class TelemetryConfig
{
    /// <summary>
    /// Gets or sets whether telemetry is enabled (default: true).
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track inference latency (default: true).
    /// </summary>
    public bool TrackLatency { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track throughput metrics (default: true).
    /// </summary>
    public bool TrackThroughput { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track errors and exceptions (default: true).
    /// </summary>
    public bool TrackErrors { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track cache hit/miss rates (default: true).
    /// </summary>
    public bool TrackCacheMetrics { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track model version usage (default: true).
    /// </summary>
    public bool TrackVersionUsage { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track detailed timing breakdowns (default: false).
    /// Includes pre-processing, inference, and post-processing times separately.
    /// More detailed but slight performance overhead.
    /// </summary>
    public bool TrackDetailedTiming { get; set; } = false;

    /// <summary>
    /// Gets or sets the sampling rate for telemetry (default: 1.0 = 100%).
    /// Values between 0.0 and 1.0. Use lower values to reduce overhead.
    /// Example: 0.1 = track only 10% of requests.
    /// </summary>
    public double SamplingRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the telemetry export endpoint URL (optional).
    /// If set, telemetry data will be sent to this endpoint for centralized monitoring.
    /// </summary>
    public string? ExportEndpoint { get; set; }

    /// <summary>
    /// Gets or sets the telemetry flush interval in seconds (default: 60).
    /// How often to export telemetry data to the endpoint.
    /// </summary>
    public int FlushIntervalSeconds { get; set; } = 60;

    /// <summary>
    /// Gets or sets custom tags to include with all telemetry events.
    /// Useful for identifying different deployments or environments.
    /// </summary>
    public Dictionary<string, string> CustomTags { get; set; } = new();

    /// <summary>
    /// Creates a minimal telemetry configuration (only errors, low overhead).
    /// </summary>
    public static TelemetryConfig Minimal()
    {
        return new TelemetryConfig
        {
            Enabled = true,
            TrackLatency = false,
            TrackThroughput = false,
            TrackErrors = true,
            TrackCacheMetrics = false,
            TrackVersionUsage = false,
            TrackDetailedTiming = false,
            SamplingRate = 0.1 // Only 10% of requests
        };
    }

    /// <summary>
    /// Creates a comprehensive telemetry configuration (all metrics, full sampling).
    /// </summary>
    public static TelemetryConfig Comprehensive()
    {
        return new TelemetryConfig
        {
            Enabled = true,
            TrackLatency = true,
            TrackThroughput = true,
            TrackErrors = true,
            TrackCacheMetrics = true,
            TrackVersionUsage = true,
            TrackDetailedTiming = true,
            SamplingRate = 1.0 // 100% sampling
        };
    }

    /// <summary>
    /// Creates a disabled telemetry configuration (no tracking).
    /// </summary>
    public static TelemetryConfig Disabled()
    {
        return new TelemetryConfig
        {
            Enabled = false
        };
    }
}
