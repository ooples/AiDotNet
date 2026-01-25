namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for telemetry - tracking and monitoring model inference metrics.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Telemetry collects data about how your model is performing in production.
/// Think of it like a fitness tracker for your AI model - it tracks important health metrics so
/// you know when something goes wrong or when performance degrades.
///
/// What gets tracked:
/// - Latency: How long each inference takes (helps identify slowdowns)
/// - Throughput: How many inferences per second (measures capacity)
/// - Errors: When predictions fail (helps identify issues)
/// - Cache hits/misses: How often cached models are used (optimizes memory)
/// - Version usage: Which model versions are being used (helps with rollouts)
///
/// Why it's important:
/// - Detect performance degradation before users complain
/// - Understand usage patterns to optimize resources
/// - Debug production issues with real data
/// - Make data-driven decisions about model updates
///
/// Privacy Note: Telemetry doesn't collect user data or predictions, only metadata
/// like timing and version information.
/// </para>
/// </remarks>
public class TelemetryConfig
{
    /// <summary>
    /// Gets or sets whether telemetry is enabled (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to true to collect performance metrics, false to disable telemetry.
    /// Recommended to keep enabled for production monitoring.
    /// </para>
    /// </remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track inference latency (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tracks how long each prediction takes.
    /// Helps identify performance problems and slow requests.
    /// </para>
    /// </remarks>
    public bool TrackLatency { get; set; } = true;

    /// <summary>
    /// Alias for TrackLatency for more intuitive access.
    /// </summary>
    public bool CollectLatency
    {
        get => TrackLatency;
        set => TrackLatency = value;
    }

    /// <summary>
    /// Gets or sets whether to track throughput metrics (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tracks how many predictions per second your system handles.
    /// Helps understand capacity and plan for scaling.
    /// </para>
    /// </remarks>
    public bool TrackThroughput { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track errors and exceptions (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records when predictions fail or errors occur.
    /// Critical for debugging and maintaining system health.
    /// </para>
    /// </remarks>
    public bool TrackErrors { get; set; } = true;

    /// <summary>
    /// Alias for TrackErrors for more intuitive access.
    /// </summary>
    public bool CollectErrors
    {
        get => TrackErrors;
        set => TrackErrors = value;
    }

    /// <summary>
    /// Gets or sets whether to track cache hit/miss rates (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tracks how often models are found in cache vs loaded from disk.
    /// Helps optimize cache size and eviction policies.
    /// </para>
    /// </remarks>
    public bool TrackCacheMetrics { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track model version usage (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records which model versions are being used.
    /// Helps track rollouts and A/B test results.
    /// </para>
    /// </remarks>
    public bool TrackVersionUsage { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track detailed timing breakdowns (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tracks pre-processing, inference, and post-processing times separately.
    /// More detailed but slight performance overhead. Use for debugging performance issues.
    /// </para>
    /// </remarks>
    public bool TrackDetailedTiming { get; set; } = false;

    /// <summary>
    /// Gets or sets the sampling rate for telemetry (default: 1.0 = 100%).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> What percentage of requests to track (0.0 to 1.0).
    /// 1.0 = track everything, 0.1 = track 10% of requests. Lower values reduce overhead.
    /// </para>
    /// </remarks>
    public double SamplingRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the telemetry export endpoint URL (optional).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> URL to send telemetry data for centralized monitoring.
    /// Leave null to only store locally. Set to your monitoring system's endpoint.
    /// </para>
    /// </remarks>
    public string? ExportEndpoint { get; set; }

    /// <summary>
    /// Gets or sets the telemetry flush interval in seconds (default: 60).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How often to send telemetry data to the export endpoint.
    /// Default is every minute. Lower values = more real-time, higher values = less network traffic.
    /// </para>
    /// </remarks>
    public int FlushIntervalSeconds { get; set; } = 60;

    /// <summary>
    /// Gets or sets custom tags to include with all telemetry events.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Add custom labels to telemetry data (e.g., environment, region).
    /// Useful for filtering and organizing metrics from different deployments.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> CustomTags { get; set; } = new();
}
