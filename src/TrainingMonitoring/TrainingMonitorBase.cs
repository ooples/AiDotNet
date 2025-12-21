using AiDotNet.Interfaces;
using AiDotNet.Models;
using Newtonsoft.Json;

namespace AiDotNet.TrainingMonitoring;

/// <summary>
/// Base class for training monitoring implementations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This abstract base class provides common functionality for training
/// monitoring systems. It handles session management, metric storage, and provides helper methods
/// for tracking training progress while leaving specific visualization to derived classes.
///
/// Key features:
/// - Thread-safe session and metric management
/// - Common metric aggregation utilities
/// - Resource usage tracking support
/// - Progress estimation helpers
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public abstract class TrainingMonitorBase<T> : ITrainingMonitor<T>
{
    /// <summary>
    /// Active monitoring sessions keyed by session ID.
    /// </summary>
    protected readonly Dictionary<string, MonitoringSession<T>> Sessions;

    /// <summary>
    /// Lock object for thread-safe operations.
    /// </summary>
    protected readonly object SyncLock = new();

    /// <summary>
    /// JSON serialization settings for consistent serialization.
    /// Uses a custom SerializationBinder for security against deserialization attacks.
    /// </summary>
    protected static readonly JsonSerializerSettings JsonSettings = new()
    {
        Formatting = Formatting.Indented,
        TypeNameHandling = TypeNameHandling.Auto,
        SerializationBinder = new SafeTypeSerializationBinder()
    };

    /// <summary>
    /// Custom serialization binder that restricts deserialization to safe types only.
    /// Prevents remote code execution vulnerabilities from malicious JSON payloads.
    /// </summary>
    private sealed class SafeTypeSerializationBinder : Newtonsoft.Json.Serialization.ISerializationBinder
    {
        private static readonly HashSet<string> AllowedNamespacePrefixes = new(StringComparer.OrdinalIgnoreCase)
        {
            "AiDotNet.",
            "System.Collections.Generic.",
            "System.Collections.Concurrent.",
            "System.Collections.ObjectModel."
            // Note: Removed broad "System." prefix to prevent deserialization of dangerous types
            // like System.Diagnostics.Process. Use AllowedSystemTypes for specific safe types.
        };

        private static readonly HashSet<string> AllowedSystemTypes = new(StringComparer.OrdinalIgnoreCase)
        {
            "System.String",
            "System.Int32",
            "System.Int64",
            "System.Double",
            "System.Single",
            "System.Decimal",
            "System.Boolean",
            "System.DateTime",
            "System.DateTimeOffset",
            "System.Guid",
            "System.TimeSpan",
            "System.Object",
            "System.Object[]"
        };

        public void BindToName(Type serializedType, out string? assemblyName, out string? typeName)
        {
            assemblyName = null;
            typeName = serializedType.AssemblyQualifiedName;
        }

        public Type BindToType(string? assemblyName, string typeName)
        {
            if (!IsTypeAllowed(typeName))
            {
                throw new JsonSerializationException(
                    $"Deserialization of type '{typeName}' is not allowed for security reasons.");
            }

            // Try to resolve the type from the type name
            var type = Type.GetType(typeName);
            if (type != null)
                return type;

            // If Type.GetType fails, search through loaded assemblies
            // This is needed for types in other assemblies (like test assemblies)
            var baseTypeName = ExtractBaseTypeName(typeName);
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                type = assembly.GetType(baseTypeName);
                if (type != null)
                    return type;
            }

            throw new JsonSerializationException($"Could not resolve type '{typeName}'.");
        }

        private static bool IsTypeAllowed(string typeName)
        {
            if (string.IsNullOrWhiteSpace(typeName))
                return false;

            var baseTypeName = ExtractBaseTypeName(typeName);

            if (AllowedSystemTypes.Contains(baseTypeName))
                return true;

            foreach (var prefix in AllowedNamespacePrefixes)
            {
                if (baseTypeName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    return true;
            }

            return false;
        }

        private static string ExtractBaseTypeName(string typeName)
        {
            var genericIndex = typeName.IndexOf('`');
            if (genericIndex > 0)
                return typeName.Substring(0, genericIndex);

            var commaIndex = typeName.IndexOf(',');
            if (commaIndex > 0)
                return typeName.Substring(0, commaIndex).Trim();

            return typeName;
        }
    }

    /// <summary>
    /// Initializes a new instance of the TrainingMonitorBase class.
    /// </summary>
    protected TrainingMonitorBase()
    {
        Sessions = new Dictionary<string, MonitoringSession<T>>();
    }

    /// <summary>
    /// Starts monitoring a training session.
    /// </summary>
    public abstract string StartSession(string sessionName, Dictionary<string, object>? metadata = null);

    /// <summary>
    /// Ends the current monitoring session.
    /// </summary>
    public abstract void EndSession(string sessionId);

    /// <summary>
    /// Records a metric value for the current training step.
    /// </summary>
    public abstract void LogMetric(string sessionId, string metricName, T value, int step, DateTime? timestamp = null);

    /// <summary>
    /// Records multiple metrics at once.
    /// </summary>
    public abstract void LogMetrics(string sessionId, Dictionary<string, T> metrics, int step);

    /// <summary>
    /// Records system resource usage.
    /// </summary>
    public abstract void LogResourceUsage(
        string sessionId,
        double cpuUsage,
        double memoryUsage,
        double? gpuUsage = null,
        double? gpuMemory = null);

    /// <summary>
    /// Updates the training progress information.
    /// </summary>
    public abstract void UpdateProgress(
        string sessionId,
        int currentStep,
        int totalSteps,
        int currentEpoch,
        int totalEpochs);

    /// <summary>
    /// Logs a text message or event during training.
    /// </summary>
    public abstract void LogMessage(string sessionId, LogLevel level, string message);

    /// <summary>
    /// Records the start of a new training epoch.
    /// </summary>
    public abstract void OnEpochStart(string sessionId, int epochNumber);

    /// <summary>
    /// Records the end of a training epoch with summary metrics.
    /// </summary>
    public abstract void OnEpochEnd(string sessionId, int epochNumber, Dictionary<string, T> metrics, TimeSpan duration);

    /// <summary>
    /// Gets the current metrics for a session.
    /// </summary>
    public abstract Dictionary<string, T> GetCurrentMetrics(string sessionId);

    /// <summary>
    /// Gets the history of a specific metric.
    /// </summary>
    public abstract List<(int Step, T Value, DateTime Timestamp)> GetMetricHistory(string sessionId, string metricName);

    /// <summary>
    /// Gets statistics about training speed.
    /// </summary>
    public abstract TrainingSpeedStats GetSpeedStats(string sessionId);

    /// <summary>
    /// Gets the current resource usage.
    /// </summary>
    public abstract ResourceUsageStats GetResourceUsage(string sessionId);

    /// <summary>
    /// Checks for potential training issues and returns warnings.
    /// </summary>
    public abstract List<string> CheckForIssues(string sessionId);

    /// <summary>
    /// Exports monitoring data to a file.
    /// </summary>
    public abstract void ExportData(string sessionId, string filePath, string format = "json");

    /// <summary>
    /// Creates a visualization of training metrics.
    /// </summary>
    public abstract void CreateVisualization(string sessionId, List<string> metricNames, string outputPath);

    #region Protected Helper Methods

    /// <summary>
    /// Validates that a session exists.
    /// </summary>
    /// <param name="sessionId">The session identifier to validate.</param>
    /// <exception cref="ArgumentException">Thrown when the session does not exist.</exception>
    protected virtual void ValidateSessionExists(string sessionId)
    {
        if (string.IsNullOrWhiteSpace(sessionId))
            throw new ArgumentException("Session ID cannot be null or empty.", nameof(sessionId));

        lock (SyncLock)
        {
            if (!Sessions.ContainsKey(sessionId))
                throw new ArgumentException($"Session with ID '{sessionId}' not found.", nameof(sessionId));
        }
    }

    /// <summary>
    /// Gets a session by ID.
    /// </summary>
    /// <param name="sessionId">The session identifier.</param>
    /// <returns>The monitoring session.</returns>
    /// <exception cref="ArgumentException">Thrown when the session does not exist.</exception>
    /// <remarks>
    /// This method uses a single lock to avoid TOCTOU race conditions.
    /// The validation and retrieval are performed atomically within the same lock.
    /// </remarks>
    protected virtual MonitoringSession<T> GetSession(string sessionId)
    {
        if (string.IsNullOrWhiteSpace(sessionId))
            throw new ArgumentException("Session ID cannot be null or empty.", nameof(sessionId));

        lock (SyncLock)
        {
            if (!Sessions.TryGetValue(sessionId, out var session))
                throw new ArgumentException($"Session with ID '{sessionId}' not found.", nameof(sessionId));
            return session;
        }
    }

    /// <summary>
    /// Generates a unique session ID.
    /// </summary>
    /// <returns>A unique session identifier.</returns>
    protected virtual string GenerateSessionId()
    {
        return Guid.NewGuid().ToString("N");
    }

    /// <summary>
    /// Serializes an object to JSON.
    /// </summary>
    protected virtual string SerializeToJson(object obj)
    {
        return JsonConvert.SerializeObject(obj, JsonSettings);
    }

    /// <summary>
    /// Validates and sanitizes a file path for export.
    /// </summary>
    protected virtual string ValidateExportPath(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        return Path.GetFullPath(filePath);
    }

    #endregion

    /// <summary>
    /// Represents a monitoring session.
    /// </summary>
    /// <typeparam name="TValue">The numeric data type.</typeparam>
    protected class MonitoringSession<TValue>
    {
        /// <summary>
        /// Unique identifier for the session.
        /// </summary>
        public string SessionId { get; set; } = string.Empty;

        /// <summary>
        /// Name of the session.
        /// </summary>
        public string SessionName { get; set; } = string.Empty;

        /// <summary>
        /// When the session started.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// When the session ended (if ended).
        /// </summary>
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// Session metadata.
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new();

        /// <summary>
        /// Current metric values.
        /// </summary>
        public Dictionary<string, TValue> CurrentMetrics { get; set; } = new();

        /// <summary>
        /// Metric history by metric name.
        /// </summary>
        public Dictionary<string, List<(int Step, TValue Value, DateTime Timestamp)>> MetricHistory { get; set; } = new();

        /// <summary>
        /// Training progress information.
        /// </summary>
        public ProgressInfo Progress { get; set; } = new();

        /// <summary>
        /// Resource usage history.
        /// </summary>
        public List<ResourceSnapshot> ResourceHistory { get; set; } = new();

        /// <summary>
        /// Log messages.
        /// </summary>
        public List<LogEntry> Messages { get; set; } = new();

        /// <summary>
        /// Epoch summary information.
        /// </summary>
        public List<EpochSummary<TValue>> EpochSummaries { get; set; } = new();
    }

    /// <summary>
    /// Represents training progress.
    /// </summary>
    protected class ProgressInfo
    {
        /// <summary>
        /// Current training step.
        /// </summary>
        public int CurrentStep { get; set; }

        /// <summary>
        /// Total planned steps.
        /// </summary>
        public int TotalSteps { get; set; }

        /// <summary>
        /// Current epoch.
        /// </summary>
        public int CurrentEpoch { get; set; }

        /// <summary>
        /// Total planned epochs.
        /// </summary>
        public int TotalEpochs { get; set; }

        /// <summary>
        /// Last update time.
        /// </summary>
        public DateTime LastUpdateTime { get; set; }
    }

    /// <summary>
    /// Represents a resource usage snapshot.
    /// </summary>
    protected class ResourceSnapshot
    {
        /// <summary>
        /// CPU usage percentage.
        /// </summary>
        public double CpuUsage { get; set; }

        /// <summary>
        /// Memory usage percentage (0-100).
        /// </summary>
        public double MemoryUsage { get; set; }

        /// <summary>
        /// GPU usage percentage (if available).
        /// </summary>
        public double? GpuUsage { get; set; }

        /// <summary>
        /// GPU memory usage percentage (0-100, if available).
        /// </summary>
        public double? GpuMemory { get; set; }

        /// <summary>
        /// Timestamp of the snapshot.
        /// </summary>
        public DateTime Timestamp { get; set; }
    }

    /// <summary>
    /// Represents a log entry.
    /// </summary>
    protected class LogEntry
    {
        /// <summary>
        /// Log level.
        /// </summary>
        public LogLevel Level { get; set; }

        /// <summary>
        /// Log message.
        /// </summary>
        public string Message { get; set; } = string.Empty;

        /// <summary>
        /// Timestamp.
        /// </summary>
        public DateTime Timestamp { get; set; }
    }

    /// <summary>
    /// Represents epoch summary information.
    /// </summary>
    protected class EpochSummary<TValue>
    {
        /// <summary>
        /// Epoch number.
        /// </summary>
        public int EpochNumber { get; set; }

        /// <summary>
        /// Start time.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// End time.
        /// </summary>
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// Duration of the epoch.
        /// </summary>
        public TimeSpan Duration { get; set; }

        /// <summary>
        /// Metrics at end of epoch.
        /// </summary>
        public Dictionary<string, TValue> Metrics { get; set; } = new();
    }
}
