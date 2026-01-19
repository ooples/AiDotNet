using System.Diagnostics;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Newtonsoft.Json;
using LogLevel = AiDotNet.Interfaces.LogLevel;

namespace AiDotNet.TrainingMonitoring;

/// <summary>
/// Implementation of training monitoring system for tracking model training progress.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This is a complete implementation of a training monitor that tracks
/// all aspects of your model training in real-time.
///
/// Features include:
/// - Real-time metric logging (loss, accuracy, etc.)
/// - Resource usage tracking (CPU, memory, GPU)
/// - Progress estimation and ETA calculation
/// - Automatic issue detection (NaN values, stalled training, etc.)
/// - Export to JSON/CSV formats
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class TrainingMonitor<T> : TrainingMonitorBase<T>
{
    /// <summary>
    /// Initializes a new instance of the TrainingMonitor class.
    /// </summary>
    public TrainingMonitor() : base()
    {
    }

    /// <summary>
    /// Starts monitoring a training session.
    /// </summary>
    public override string StartSession(string sessionName, Dictionary<string, object>? metadata = null)
    {
        if (string.IsNullOrWhiteSpace(sessionName))
            throw new ArgumentException("Session name cannot be null or empty.", nameof(sessionName));

        lock (SyncLock)
        {
            var sessionId = GenerateSessionId();
            var session = new MonitoringSession<T>
            {
                SessionId = sessionId,
                SessionName = sessionName,
                StartTime = DateTime.UtcNow,
                StartTimestamp = Stopwatch.GetTimestamp(),
                Metadata = metadata ?? new Dictionary<string, object>()
            };

            Sessions[sessionId] = session;
            return sessionId;
        }
    }

    /// <summary>
    /// Ends the current monitoring session.
    /// </summary>
    public override void EndSession(string sessionId)
    {
        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            session.EndTime = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Records a metric value for the current training step.
    /// </summary>
    public override void LogMetric(string sessionId, string metricName, T value, int step, DateTime? timestamp = null)
    {
        if (string.IsNullOrWhiteSpace(metricName))
            throw new ArgumentException("Metric name cannot be null or empty.", nameof(metricName));

        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            var time = timestamp ?? DateTime.UtcNow;

            // Update current metrics
            session.CurrentMetrics[metricName] = value;

            // Add to history
            if (!session.MetricHistory.ContainsKey(metricName))
            {
                session.MetricHistory[metricName] = new List<(int Step, T Value, DateTime Timestamp)>();
            }

            session.MetricHistory[metricName].Add((step, value, time));
        }
    }

    /// <summary>
    /// Records multiple metrics at once.
    /// </summary>
    public override void LogMetrics(string sessionId, Dictionary<string, T> metrics, int step)
    {
        if (metrics == null)
            throw new ArgumentNullException(nameof(metrics));

        var timestamp = DateTime.UtcNow;
        foreach (var kvp in metrics)
        {
            LogMetric(sessionId, kvp.Key, kvp.Value, step, timestamp);
        }
    }

    /// <summary>
    /// Records system resource usage.
    /// </summary>
    public override void LogResourceUsage(
        string sessionId,
        double cpuUsage,
        double memoryUsage,
        double? gpuUsage = null,
        double? gpuMemory = null)
    {
        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            var snapshot = new ResourceSnapshot
            {
                CpuUsage = cpuUsage,
                MemoryUsage = memoryUsage,
                GpuUsage = gpuUsage,
                GpuMemory = gpuMemory,
                Timestamp = DateTime.UtcNow
            };

            session.ResourceHistory.Add(snapshot);
        }
    }

    /// <summary>
    /// Updates the training progress information.
    /// </summary>
    public override void UpdateProgress(
        string sessionId,
        int currentStep,
        int totalSteps,
        int currentEpoch,
        int totalEpochs)
    {
        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            session.Progress = new ProgressInfo
            {
                CurrentStep = currentStep,
                TotalSteps = totalSteps,
                CurrentEpoch = currentEpoch,
                TotalEpochs = totalEpochs,
                LastUpdateTime = DateTime.UtcNow
            };
        }
    }

    /// <summary>
    /// Logs a text message or event during training.
    /// </summary>
    public override void LogMessage(string sessionId, LogLevel level, string message)
    {
        if (string.IsNullOrWhiteSpace(message))
            throw new ArgumentException("Message cannot be null or empty.", nameof(message));

        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            session.Messages.Add(new LogEntry
            {
                Level = level,
                Message = message,
                Timestamp = DateTime.UtcNow
            });
        }
    }

    /// <summary>
    /// Records the start of a new training epoch.
    /// </summary>
    public override void OnEpochStart(string sessionId, int epochNumber)
    {
        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            session.EpochSummaries.Add(new EpochSummary<T>
            {
                EpochNumber = epochNumber,
                StartTime = DateTime.UtcNow
            });

            LogMessage(sessionId, LogLevel.Info, $"Epoch {epochNumber} started");
        }
    }

    /// <summary>
    /// Records the end of a training epoch with summary metrics.
    /// </summary>
    public override void OnEpochEnd(string sessionId, int epochNumber, Dictionary<string, T> metrics, TimeSpan duration)
    {
        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            var epochSummary = session.EpochSummaries.FirstOrDefault(e => e.EpochNumber == epochNumber);

            if (epochSummary != null)
            {
                epochSummary.EndTime = DateTime.UtcNow;
                epochSummary.Duration = duration;
                epochSummary.Metrics = new Dictionary<string, T>(metrics);
            }

            LogMessage(sessionId, LogLevel.Info, $"Epoch {epochNumber} completed in {duration.TotalSeconds:F2}s");
        }
    }

    /// <summary>
    /// Gets the current metrics for a session.
    /// </summary>
    public override Dictionary<string, T> GetCurrentMetrics(string sessionId)
    {
        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            return new Dictionary<string, T>(session.CurrentMetrics);
        }
    }

    /// <summary>
    /// Gets the history of a specific metric.
    /// </summary>
    public override List<(int Step, T Value, DateTime Timestamp)> GetMetricHistory(string sessionId, string metricName)
    {
        if (string.IsNullOrWhiteSpace(metricName))
            throw new ArgumentException("Metric name cannot be null or empty.", nameof(metricName));

        lock (SyncLock)
        {
            var session = GetSession(sessionId);

            if (!session.MetricHistory.TryGetValue(metricName, out var history))
            {
                return new List<(int Step, T Value, DateTime Timestamp)>();
            }

            return new List<(int Step, T Value, DateTime Timestamp)>(history);
        }
    }

    /// <summary>
    /// Gets statistics about training speed.
    /// </summary>
    public override TrainingSpeedStats GetSpeedStats(string sessionId)
    {
        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            var progress = session.Progress;
            var elapsed = GetElapsedTime(session, progress);

            double iterationsPerSecond = 0;
            double secondsPerIteration = 0;
            TimeSpan estimatedRemaining = TimeSpan.Zero;
            double progressPercentage = 0;

            if (elapsed.TotalSeconds > 0 && progress.CurrentStep > 0)
            {
                iterationsPerSecond = progress.CurrentStep / elapsed.TotalSeconds;
                secondsPerIteration = elapsed.TotalSeconds / progress.CurrentStep;

                if (progress.TotalSteps > 0)
                {
                    progressPercentage = (double)progress.CurrentStep / progress.TotalSteps * 100;

                    if (iterationsPerSecond > 0 && progress.TotalSteps > progress.CurrentStep)
                    {
                        var remainingSteps = progress.TotalSteps - progress.CurrentStep;
                        estimatedRemaining = TimeSpan.FromSeconds(remainingSteps / iterationsPerSecond);
                    }
                }
            }

            return new TrainingSpeedStats
            {
                IterationsPerSecond = iterationsPerSecond,
                SecondsPerIteration = secondsPerIteration,
                ElapsedTime = elapsed,
                EstimatedTimeRemaining = estimatedRemaining,
                ProgressPercentage = progressPercentage,
                IterationsCompleted = progress.CurrentStep,
                TotalIterations = progress.TotalSteps
            };
        }
    }

    private static TimeSpan GetElapsedTime(MonitoringSession<T> session, ProgressInfo progress)
    {
        // Prefer high-resolution stopwatch time for stability across runtimes (notably .NET Framework).
        if (session.StartTimestamp > 0)
        {
            long deltaTicks = Stopwatch.GetTimestamp() - session.StartTimestamp;
            if (deltaTicks > 0 && Stopwatch.Frequency > 0)
            {
                return TimeSpan.FromSeconds(deltaTicks / (double)Stopwatch.Frequency);
            }
        }

        var elapsed = DateTime.UtcNow - session.StartTime;
        if (elapsed < TimeSpan.Zero)
        {
            elapsed = TimeSpan.Zero;
        }

        // If progress has started but timer resolution produced a 0 duration, clamp to a minimal value.
        if (elapsed == TimeSpan.Zero && progress.CurrentStep > 0)
        {
            elapsed = TimeSpan.FromMilliseconds(1);
        }

        return elapsed;
    }

    /// <summary>
    /// Gets the current resource usage.
    /// </summary>
    public override ResourceUsageStats GetResourceUsage(string sessionId)
    {
        lock (SyncLock)
        {
            var session = GetSession(sessionId);
            var latest = session.ResourceHistory.LastOrDefault();

            if (latest == null)
            {
                return new ResourceUsageStats
                {
                    Timestamp = DateTime.UtcNow
                };
            }

            return new ResourceUsageStats
            {
                CpuUsagePercent = latest.CpuUsage,
                MemoryUsageMB = latest.MemoryUsage,
                MemoryUsagePercent = 0, // Would need total memory to calculate
                GpuUsagePercent = latest.GpuUsage,
                GpuMemoryUsageMB = latest.GpuMemory,
                Timestamp = latest.Timestamp
            };
        }
    }

    /// <summary>
    /// Checks for potential training issues and returns warnings.
    /// </summary>
    public override List<string> CheckForIssues(string sessionId)
    {
        var issues = new List<string>();

        lock (SyncLock)
        {
            var session = GetSession(sessionId);

            // Check for stalled training (no progress in last 5 minutes)
            if (session.Progress.LastUpdateTime != default)
            {
                var timeSinceUpdate = DateTime.UtcNow - session.Progress.LastUpdateTime;
                if (timeSinceUpdate > TimeSpan.FromMinutes(5))
                {
                    issues.Add($"Training appears stalled - no progress update in {timeSinceUpdate.TotalMinutes:F1} minutes");
                }
            }

            // Check for high memory usage
            var latestResource = session.ResourceHistory.LastOrDefault();
            if (latestResource != null)
            {
                if (latestResource.MemoryUsage > 90)
                {
                    issues.Add($"High memory usage: {latestResource.MemoryUsage:F1}%");
                }

                if (latestResource.GpuMemory.HasValue && latestResource.GpuMemory > 95)
                {
                    issues.Add($"Critical GPU memory usage: {latestResource.GpuMemory:F1}%");
                }
            }

            // Check for error messages
            var recentErrors = session.Messages
                .Where(m => m.Level == LogLevel.Error && m.Timestamp > DateTime.UtcNow.AddMinutes(-10))
                .ToList();

            if (recentErrors.Count > 0)
            {
                issues.Add($"{recentErrors.Count} error(s) in the last 10 minutes");
            }

            // Check epoch durations for anomalies
            if (session.EpochSummaries.Count >= 3)
            {
                var completedEpochs = session.EpochSummaries.Where(e => e.EndTime.HasValue).ToList();
                if (completedEpochs.Count >= 3)
                {
                    var avgDuration = completedEpochs.Average(e => e.Duration.TotalSeconds);
                    var lastDuration = completedEpochs.Last().Duration.TotalSeconds;

                    if (lastDuration > avgDuration * 2)
                    {
                        issues.Add($"Last epoch took {lastDuration:F1}s, which is more than 2x the average ({avgDuration:F1}s)");
                    }
                }
            }
        }

        return issues;
    }

    /// <summary>
    /// Exports monitoring data to a file.
    /// </summary>
    public override void ExportData(string sessionId, string filePath, string format = "json")
    {
        var validatedPath = ValidateExportPath(filePath);

        lock (SyncLock)
        {
            var session = GetSession(sessionId);

            if (format.Equals("json", StringComparison.OrdinalIgnoreCase))
            {
                var exportData = new
                {
                    session.SessionId,
                    session.SessionName,
                    session.StartTime,
                    session.EndTime,
                    session.Metadata,
                    session.CurrentMetrics,
                    MetricHistory = session.MetricHistory.ToDictionary(
                        kvp => kvp.Key,
                        kvp => kvp.Value.Select(v => new { v.Step, v.Value, v.Timestamp }).ToList()
                    ),
                    Progress = session.Progress,
                    ResourceHistory = session.ResourceHistory,
                    Messages = session.Messages,
                    EpochSummaries = session.EpochSummaries
                };

                var json = SerializeToJson(exportData);
                File.WriteAllText(validatedPath, json);
            }
            else if (format.Equals("csv", StringComparison.OrdinalIgnoreCase))
            {
                // Export metrics as CSV
                using var writer = new StreamWriter(validatedPath);

                // Write header
                var metricNames = session.MetricHistory.Keys.ToList();
                writer.WriteLine("Step,Timestamp," + string.Join(",", metricNames));

                // Get all unique steps
                var allSteps = session.MetricHistory.Values
                    .SelectMany(h => h.Select(v => v.Step))
                    .Distinct()
                    .OrderBy(s => s)
                    .ToList();

                foreach (var step in allSteps)
                {
                    var values = new List<string> { step.ToString() };

                    // Find timestamp for this step
                    var timestamp = session.MetricHistory.Values
                        .SelectMany(h => h)
                        .FirstOrDefault(v => v.Step == step)
                        .Timestamp;
                    values.Add(timestamp.ToString("O"));

                    foreach (var metricName in metricNames)
                    {
                        var metricValue = session.MetricHistory[metricName]
                            .FirstOrDefault(v => v.Step == step);
                        values.Add(metricValue.Value?.ToString() ?? "");
                    }

                    writer.WriteLine(string.Join(",", values));
                }
            }
            else
            {
                throw new ArgumentException($"Unsupported export format: {format}. Supported formats: json, csv", nameof(format));
            }
        }
    }

    /// <summary>
    /// Creates a visualization of training metrics.
    /// </summary>
    /// <remarks>
    /// This implementation exports data in a format suitable for visualization tools.
    /// For actual plotting, integrate with a visualization library.
    /// </remarks>
    public override void CreateVisualization(string sessionId, List<string> metricNames, string outputPath)
    {
        if (metricNames == null || metricNames.Count == 0)
            throw new ArgumentException("At least one metric name must be specified.", nameof(metricNames));

        var validatedPath = ValidateExportPath(outputPath);

        lock (SyncLock)
        {
            var session = GetSession(sessionId);

            // Export visualization-ready data as JSON
            var visualizationData = new
            {
                Title = $"Training Metrics - {session.SessionName}",
                SessionId = session.SessionId,
                StartTime = session.StartTime,
                Metrics = metricNames
                    .Where(m => session.MetricHistory.ContainsKey(m))
                    .Select(metricName => new
                    {
                        Name = metricName,
                        Data = session.MetricHistory[metricName]
                            .Select(v => new { v.Step, v.Value, v.Timestamp })
                            .ToList()
                    })
                    .ToList()
            };

            var json = SerializeToJson(visualizationData);
            File.WriteAllText(validatedPath, json);
        }
    }
}
