#if !NET6_0_OR_GREATER
using AiDotNet.TrainingMonitoring;
#endif
using AiDotNet.Dashboard.Console;
using AiDotNet.Dashboard.Visualization;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Newtonsoft.Json;
using SystemConsole = System.Console;

namespace AiDotNet.Dashboard.Dashboard;

/// <summary>
/// A training monitor implementation with real-time console dashboard visualization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This class provides both training monitoring AND visual feedback.
/// Unlike a simple monitor that just collects data, this one displays live progress bars,
/// training curves, and alerts in the console as training happens.
///
/// Example usage:
/// <code>
/// using var monitor = new TrainingMonitorDashboard&lt;double&gt;();
/// var sessionId = monitor.StartSession("My Training");
///
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     monitor.OnEpochStart(sessionId, epoch + 1);
///
///     for (int batch = 0; batch &lt; batches.Length; batch++)
///     {
///         // Train...
///         monitor.UpdateProgress(sessionId, batch + epoch * batches.Length,
///                                totalSteps, epoch + 1, 100);
///         monitor.LogMetric(sessionId, "loss", loss, batch);
///     }
///
///     monitor.OnEpochEnd(sessionId, epoch + 1, metrics, duration);
/// }
///
/// monitor.EndSession(sessionId);
/// </code>
///
/// Features:
/// - Live progress bars for epochs and batches
/// - Real-time training curves in the console
/// - Resource utilization monitoring
/// - Alert notifications when thresholds are crossed
/// - Full ITrainingMonitor compliance for infrastructure integration
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class TrainingMonitorDashboard<T> : ITrainingMonitor<T>, IDisposable
{
    private readonly string _defaultName;
    private readonly object _lock = new();
    private readonly Dictionary<string, SessionData> _sessions = new();

    private ProgressBar? _epochProgress;
    private ProgressBar? _batchProgress;
    private TrainingCurves? _trainingCurves;
    private Timer? _refreshTimer;
    private bool _isDisposed;
    private string? _activeSessionId;

    private class SessionData
    {
        public string SessionId { get; set; } = string.Empty;
        public string SessionName { get; set; } = string.Empty;
        public Dictionary<string, object>? Metadata { get; set; }
        public DateTime StartTime { get; set; } = DateTime.UtcNow;
        public DateTime? EndTime { get; set; }
        public int CurrentStep { get; set; }
        public int TotalSteps { get; set; }
        public int CurrentEpoch { get; set; }
        public int TotalEpochs { get; set; }
        public Dictionary<string, List<(int Step, T Value, DateTime Timestamp)>> MetricHistory { get; } = new();
        public Dictionary<string, T> CurrentMetrics { get; } = new();
        public Dictionary<string, double> CurrentMetricsDouble { get; } = new();
        public List<(DateTime Time, LogLevel Level, string Message)> Messages { get; } = new();
        public ResourceUsageStats? LastResourceUsage { get; set; }
        public Dictionary<string, (double Threshold, bool TriggerAbove)> Alerts { get; } = new();
        public List<string> AlertMessages { get; } = new();
    }

    /// <summary>
    /// Gets or sets the refresh interval in milliseconds for the dashboard display.
    /// </summary>
    public int RefreshIntervalMs { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to show training curves.
    /// </summary>
    public bool ShowCurves { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to show resource utilization.
    /// </summary>
    public bool ShowResources { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use console colors.
    /// </summary>
    public bool UseColors { get; set; } = true;

    /// <summary>
    /// Event raised when an alert is triggered.
    /// </summary>
    public event EventHandler<AlertEventArgs>? AlertTriggered;

    /// <summary>
    /// Initializes a new instance of the TrainingMonitorDashboard class.
    /// </summary>
    /// <param name="defaultName">Default name for training sessions.</param>
    public TrainingMonitorDashboard(string defaultName = "Training")
    {
        _defaultName = defaultName;
    }

    /// <inheritdoc />
    public string StartSession(string sessionName, Dictionary<string, object>? metadata = null)
    {
        if (string.IsNullOrWhiteSpace(sessionName))
            sessionName = _defaultName;

        lock (_lock)
        {
            var sessionId = Guid.NewGuid().ToString();
            var session = new SessionData
            {
                SessionId = sessionId,
                SessionName = sessionName,
                Metadata = metadata,
                StartTime = DateTime.UtcNow
            };

            _sessions[sessionId] = session;
            _activeSessionId = sessionId;

            // Initialize dashboard components
            _trainingCurves = new TrainingCurves(80, 15, sessionName);

            // Start refresh timer
            _refreshTimer?.Dispose();
            _refreshTimer = new Timer(RefreshDisplay, null, 0, RefreshIntervalMs);

            RenderHeader(session);
            return sessionId;
        }
    }

    /// <inheritdoc />
    public void EndSession(string sessionId)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return;

            session.EndTime = DateTime.UtcNow;

            // Complete progress bars
            _epochProgress?.Complete();
            _batchProgress?.Complete();

            // Stop refresh timer
            _refreshTimer?.Dispose();
            _refreshTimer = null;

            RenderSummary(session);

            if (_activeSessionId == sessionId)
            {
                _activeSessionId = null;
            }
        }
    }

    /// <inheritdoc />
    public void LogMetric(string sessionId, string metricName, T value, int step, DateTime? timestamp = null)
    {
        if (string.IsNullOrWhiteSpace(sessionId) || string.IsNullOrWhiteSpace(metricName))
            return;

        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return;

            var ts = timestamp ?? DateTime.UtcNow;

            // Store in history
            if (!session.MetricHistory.ContainsKey(metricName))
            {
                session.MetricHistory[metricName] = new List<(int, T, DateTime)>();
            }
            session.MetricHistory[metricName].Add((step, value, ts));

            // Update current value
            session.CurrentMetrics[metricName] = value;

            // Convert to double for visualization
            double doubleValue;
            if (value is IConvertible convertible)
            {
                doubleValue = convertible.ToDouble(null);
                session.CurrentMetricsDouble[metricName] = doubleValue;
            }
            else
            {
                // Type doesn't implement IConvertible; skip visualization for this metric
                System.Diagnostics.Debug.WriteLine(
                    $"[TrainingMonitorDashboard] Warning: Metric '{metricName}' has type '{typeof(T).Name}' " +
                    $"which does not implement IConvertible. Visualization will be skipped for this metric.");
                return;
            }

            // Add to training curves
            _trainingCurves?.AddPoint(metricName, session.CurrentEpoch > 0 ? session.CurrentEpoch : step, doubleValue);

            // Check alerts
            CheckAlert(session, metricName, doubleValue);
        }
    }

    /// <inheritdoc />
    public void LogMetrics(string sessionId, Dictionary<string, T> metrics, int step)
    {
        if (metrics is null)
            return;

        foreach (var kvp in metrics)
        {
            LogMetric(sessionId, kvp.Key, kvp.Value, step);
        }
    }

    /// <inheritdoc />
    public void LogResourceUsage(string sessionId, double cpuUsage, double memoryUsage, double? gpuUsage = null, double? gpuMemory = null)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return;

            session.LastResourceUsage = new ResourceUsageStats
            {
                CpuUsagePercent = cpuUsage,
                MemoryUsageMB = memoryUsage,
                GpuUsagePercent = gpuUsage,
                GpuMemoryUsageMB = gpuMemory,
                Timestamp = DateTime.UtcNow
            };

            // Store as metrics for visualization
            session.CurrentMetricsDouble["cpu_percent"] = cpuUsage;
            session.CurrentMetricsDouble["memory_mb"] = memoryUsage;
            if (gpuUsage.HasValue)
                session.CurrentMetricsDouble["gpu_percent"] = gpuUsage.Value;
            if (gpuMemory.HasValue)
                session.CurrentMetricsDouble["gpu_memory_mb"] = gpuMemory.Value;
        }
    }

    /// <inheritdoc />
    public void UpdateProgress(string sessionId, int currentStep, int totalSteps, int currentEpoch, int totalEpochs)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return;

            session.CurrentStep = currentStep;
            session.TotalSteps = totalSteps;
            session.CurrentEpoch = currentEpoch;
            session.TotalEpochs = totalEpochs;

            // Update epoch progress bar
            if (totalEpochs > 0)
            {
                if (_epochProgress is null || _epochProgress.Total != totalEpochs)
                {
                    _epochProgress?.Dispose();
                    _epochProgress = new ProgressBar(totalEpochs, "Epoch", useColors: UseColors);
                }
                _epochProgress.Update(currentEpoch);
            }

            // Update batch progress bar
            // Handle edge cases: totalSteps < totalEpochs, steps not dividing evenly
            if (totalSteps > 0 && totalEpochs > 0)
            {
                // Calculate steps per epoch, with a minimum of 1 to prevent division by zero
                int stepsPerEpoch = Math.Max(1, totalSteps / totalEpochs);

                // Calculate which batch within the current epoch
                // Use currentEpoch-1 as base since epochs are 1-indexed
                int epochStartStep = (currentEpoch - 1) * stepsPerEpoch;
                int batchInEpoch = Math.Max(0, currentStep - epochStartStep);

                // Ensure batch stays within bounds
                batchInEpoch = Math.Min(batchInEpoch, stepsPerEpoch);

                if (_batchProgress is null || _batchProgress.Total != stepsPerEpoch)
                {
                    _batchProgress?.Dispose();
                    _batchProgress = new ProgressBar(stepsPerEpoch, "Batch", barWidth: 30, useColors: UseColors);
                }

                // Update with 1-indexed batch number (minimum 1, maximum stepsPerEpoch)
                _batchProgress.Update(Math.Max(1, Math.Min(batchInEpoch + 1, stepsPerEpoch)));
            }
        }
    }

    /// <inheritdoc />
    public void LogMessage(string sessionId, LogLevel level, string message)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return;

            session.Messages.Add((DateTime.UtcNow, level, message));

            // Display message with appropriate formatting
            string prefix = level switch
            {
                LogLevel.Warning => "[WARN] ",
                LogLevel.Error => "[ERROR] ",
                LogLevel.Debug => "[DEBUG] ",
                _ => "[INFO] "
            };

            _epochProgress?.SetStatus(prefix + message);
        }
    }

    /// <inheritdoc />
    public void OnEpochStart(string sessionId, int epochNumber)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return;

            session.CurrentEpoch = epochNumber;

            if (_epochProgress is not null && session.TotalEpochs > 0)
            {
                _epochProgress.Update(epochNumber);
            }

            _epochProgress?.SetStatus($"Epoch {epochNumber} started");
        }
    }

    /// <inheritdoc />
    public void OnEpochEnd(string sessionId, int epochNumber, Dictionary<string, T> metrics, TimeSpan duration)
    {
        if (metrics is null)
            return;

        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return;

            // Log all epoch metrics at this step
            foreach (var kvp in metrics)
            {
                LogMetric(sessionId, kvp.Key, kvp.Value, epochNumber);
            }

            _epochProgress?.SetStatus($"Epoch {epochNumber} completed in {duration.TotalSeconds:F2}s");
        }
    }

    /// <inheritdoc />
    public Dictionary<string, T> GetCurrentMetrics(string sessionId)
    {
        lock (_lock)
        {
            if (_sessions.TryGetValue(sessionId, out var session))
            {
                return new Dictionary<string, T>(session.CurrentMetrics);
            }
            return new Dictionary<string, T>();
        }
    }

    /// <inheritdoc />
    public List<(int Step, T Value, DateTime Timestamp)> GetMetricHistory(string sessionId, string metricName)
    {
        lock (_lock)
        {
            if (_sessions.TryGetValue(sessionId, out var session) &&
                session.MetricHistory.TryGetValue(metricName, out var history))
            {
                return new List<(int, T, DateTime)>(history);
            }
            return new List<(int, T, DateTime)>();
        }
    }

    /// <inheritdoc />
    public TrainingSpeedStats GetSpeedStats(string sessionId)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return new TrainingSpeedStats();

            var elapsed = (session.EndTime ?? DateTime.UtcNow) - session.StartTime;
            var iterationsCompleted = session.CurrentStep;
            var totalIterations = session.TotalSteps;

            double iterationsPerSecond = elapsed.TotalSeconds > 0
                ? iterationsCompleted / elapsed.TotalSeconds
                : 0;

            double secondsPerIteration = iterationsCompleted > 0
                ? elapsed.TotalSeconds / iterationsCompleted
                : 0;

            int remaining = totalIterations - iterationsCompleted;
            var estimatedRemaining = TimeSpan.FromSeconds(remaining * secondsPerIteration);

            double progressPercentage = totalIterations > 0
                ? (double)iterationsCompleted / totalIterations * 100
                : 0;

            return new TrainingSpeedStats
            {
                IterationsPerSecond = iterationsPerSecond,
                SecondsPerIteration = secondsPerIteration,
                EstimatedTimeRemaining = estimatedRemaining,
                ElapsedTime = elapsed,
                ProgressPercentage = progressPercentage,
                IterationsCompleted = iterationsCompleted,
                TotalIterations = totalIterations
            };
        }
    }

    /// <inheritdoc />
    public ResourceUsageStats GetResourceUsage(string sessionId)
    {
        lock (_lock)
        {
            if (_sessions.TryGetValue(sessionId, out var session) && session.LastResourceUsage is not null)
            {
                return session.LastResourceUsage;
            }
            return new ResourceUsageStats { Timestamp = DateTime.UtcNow };
        }
    }

    /// <inheritdoc />
    public List<string> CheckForIssues(string sessionId)
    {
        var issues = new List<string>();

        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return issues;

            // Check for NaN/Infinity metrics
            foreach (var kvp in session.CurrentMetricsDouble)
            {
                if (double.IsNaN(kvp.Value) || double.IsInfinity(kvp.Value))
                {
                    issues.Add($"Metric '{kvp.Key}' has invalid value: {kvp.Value}");
                }
            }

            // Check for stagnant loss
            if (session.MetricHistory.TryGetValue("loss", out var lossHistory) ||
                session.MetricHistory.TryGetValue("train_loss", out lossHistory))
            {
                if (lossHistory.Count >= 10)
                {
                    // Use Skip/Take for net471 compatibility (TakeLast not available)
                    var recentLosses = lossHistory.Skip(Math.Max(0, lossHistory.Count - 10)).Take(10).ToList();
                    var firstValue = recentLosses.First().Value;
                    var lastValue = recentLosses.Last().Value;

                    if (firstValue is IConvertible firstConv && lastValue is IConvertible lastConv)
                    {
                        double first = firstConv.ToDouble(null);
                        double last = lastConv.ToDouble(null);

                        if (Math.Abs(first - last) < 0.0001 && first > 0)
                        {
                            issues.Add("Loss appears stagnant - training may not be learning");
                        }

                        if (last > first * 1.5)
                        {
                            issues.Add("Loss is increasing - learning rate may be too high");
                        }
                    }
                }
            }

            // Check resource usage
            if (session.LastResourceUsage is not null)
            {
                if (session.LastResourceUsage.MemoryUsagePercent > 90)
                {
                    issues.Add("Memory usage is above 90% - risk of out-of-memory error");
                }

                if (session.LastResourceUsage.GpuMemoryUsagePercent > 95)
                {
                    issues.Add("GPU memory usage is above 95% - risk of out-of-memory error");
                }
            }

            // Include recent errors (use Skip/Take for net471 compatibility)
            var errorMessages = session.Messages.Where(m => m.Level == LogLevel.Error).ToList();
            var recentErrors = errorMessages.Skip(Math.Max(0, errorMessages.Count - 5)).Take(5).ToList();

            foreach (var error in recentErrors)
            {
                issues.Add($"Error: {error.Message}");
            }
        }

        return issues;
    }

    /// <inheritdoc />
    public void ExportData(string sessionId, string filePath, string format = "json")
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        string content;

        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                throw new ArgumentException($"Session '{sessionId}' not found.", nameof(sessionId));

            var exportData = new
            {
                session.SessionId,
                session.SessionName,
                session.Metadata,
                session.StartTime,
                session.EndTime,
                Duration = (session.EndTime ?? DateTime.UtcNow) - session.StartTime,
                session.CurrentEpoch,
                session.TotalEpochs,
                session.CurrentStep,
                session.TotalSteps,
                CurrentMetrics = session.CurrentMetricsDouble,
                MetricHistory = session.MetricHistory.ToDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value.Select(v => new
                    {
                        v.Step,
                        Value = v.Value is IConvertible conv ? conv.ToDouble(null) : 0.0,
                        v.Timestamp
                    }).ToList()),
                Messages = session.Messages.Select(m => new
                {
                    m.Time,
                    Level = m.Level.ToString(),
                    m.Message
                }).ToList(),
                session.LastResourceUsage,
                SpeedStats = GetSpeedStats(sessionId),
                Issues = CheckForIssues(sessionId)
            };

            if (format.Equals("json", StringComparison.OrdinalIgnoreCase))
            {
                content = JsonConvert.SerializeObject(exportData, Formatting.Indented);
            }
            else if (format.Equals("csv", StringComparison.OrdinalIgnoreCase))
            {
                var lines = new List<string> { "Timestamp,Metric,Step,Value" };
                foreach (var metricKvp in session.MetricHistory)
                {
                    foreach (var entry in metricKvp.Value)
                    {
                        double value = entry.Value is IConvertible conv ? conv.ToDouble(null) : 0.0;
                        lines.Add($"{entry.Timestamp:O},{metricKvp.Key},{entry.Step},{value}");
                    }
                }
                content = string.Join(Environment.NewLine, lines);
            }
            else
            {
                throw new ArgumentException($"Unsupported format: {format}", nameof(format));
            }
        }

        // File I/O outside the lock to reduce thread contention
        File.WriteAllText(filePath, content);
    }

    /// <inheritdoc />
    public void CreateVisualization(string sessionId, List<string> metricNames, string outputPath)
    {
        lock (_lock)
        {
            // Show curves in console
            _trainingCurves?.Render();
        }

        // Export data for external visualization
        ExportData(sessionId, outputPath, "json");
    }

    /// <summary>
    /// Sets an alert threshold for a metric.
    /// </summary>
    /// <param name="sessionId">The session ID.</param>
    /// <param name="metricName">The metric to monitor.</param>
    /// <param name="threshold">The threshold value.</param>
    /// <param name="triggerAbove">True to trigger when above threshold, false for below.</param>
    public void SetAlert(string sessionId, string metricName, double threshold, bool triggerAbove = true)
    {
        lock (_lock)
        {
            if (_sessions.TryGetValue(sessionId, out var session))
            {
                session.Alerts[metricName] = (threshold, triggerAbove);
            }
        }
    }

    /// <summary>
    /// Shows the training curves visualization.
    /// </summary>
    public void ShowTrainingCurves()
    {
        lock (_lock)
        {
            _trainingCurves?.Render();
        }
    }

    private void CheckAlert(SessionData session, string metricName, double value)
    {
        if (!session.Alerts.TryGetValue(metricName, out var alert))
            return;

        bool triggered = alert.TriggerAbove ? value > alert.Threshold : value < alert.Threshold;

        if (triggered)
        {
            var message = $"[{DateTime.UtcNow:HH:mm:ss}] ALERT: {metricName} = {value:F4} " +
                         $"({(alert.TriggerAbove ? "above" : "below")} threshold {alert.Threshold:F4})";

            session.AlertMessages.Add(message);

            AlertTriggered?.Invoke(this, new AlertEventArgs
            {
                MetricName = metricName,
                Value = value,
                Threshold = alert.Threshold,
                TriggerAbove = alert.TriggerAbove,
                Message = message
            });
        }
    }

    private void RefreshDisplay(object? state)
    {
        // Progress bars auto-render on update
        // Additional display logic can be added here
    }

    private void RenderHeader(SessionData session)
    {
        SystemConsole.WriteLine();
        SystemConsole.WriteLine($"{'=',-60}");
        SystemConsole.WriteLine($"  {session.SessionName}");
        SystemConsole.WriteLine($"  Started: {session.StartTime:yyyy-MM-dd HH:mm:ss} UTC");
        SystemConsole.WriteLine($"{'=',-60}");
        SystemConsole.WriteLine();
    }

    private void RenderSummary(SessionData session)
    {
        var duration = (session.EndTime ?? DateTime.UtcNow) - session.StartTime;

        SystemConsole.WriteLine();
        SystemConsole.WriteLine($"{'=',-60}");
        SystemConsole.WriteLine("  Training Complete!");
        SystemConsole.WriteLine($"{'=',-60}");
        SystemConsole.WriteLine();
        SystemConsole.WriteLine($"Duration: {duration:hh\\:mm\\:ss}");
        SystemConsole.WriteLine($"Epochs: {session.CurrentEpoch}/{session.TotalEpochs}");
        SystemConsole.WriteLine($"Steps: {session.CurrentStep}/{session.TotalSteps}");
        SystemConsole.WriteLine();

        SystemConsole.WriteLine("Final Metrics:");
        foreach (var kvp in session.CurrentMetricsDouble.OrderBy(k => k.Key))
        {
            SystemConsole.WriteLine($"  {kvp.Key}: {kvp.Value:F4}");
        }

        var issues = CheckForIssues(session.SessionId);
        if (issues.Count > 0)
        {
            SystemConsole.WriteLine();
            SystemConsole.WriteLine("Issues Detected:");
            foreach (var issue in issues)
            {
                SystemConsole.WriteLine($"  - {issue}");
            }
        }

        if (session.AlertMessages.Count > 0)
        {
            SystemConsole.WriteLine();
            SystemConsole.WriteLine("Alerts:");
            // Use Skip/Take for net471 compatibility (TakeLast not available)
            var recentAlerts = session.AlertMessages.Skip(Math.Max(0, session.AlertMessages.Count - 5)).Take(5);
            foreach (var alert in recentAlerts)
            {
                SystemConsole.WriteLine($"  {alert}");
            }
        }

        SystemConsole.WriteLine();

        if (ShowCurves && session.MetricHistory.Any(m => m.Value.Count > 1))
        {
            ShowTrainingCurves();
        }
    }

    /// <summary>
    /// Disposes the dashboard and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;

        _refreshTimer?.Dispose();
        _epochProgress?.Dispose();
        _batchProgress?.Dispose();
    }
}
