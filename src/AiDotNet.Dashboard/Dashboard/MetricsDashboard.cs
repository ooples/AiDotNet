#if !NET6_0_OR_GREATER
using AiDotNet.TrainingMonitoring;
#endif
using AiDotNet.Dashboard.Console;
using AiDotNet.Dashboard.Visualization;
using SystemConsole = System.Console;

namespace AiDotNet.Dashboard.Dashboard;

/// <summary>
/// A real-time metrics dashboard for monitoring training progress.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> The MetricsDashboard provides a unified view of your training:
/// - Live progress bars for epoch and batch progress
/// - Real-time training curves (loss, accuracy)
/// - Resource utilization (GPU, CPU, memory)
/// - Alert notifications when thresholds are crossed
///
/// Example usage:
/// <code>
/// var dashboard = new MetricsDashboard("My Training Run");
/// dashboard.Start();
///
/// for (int epoch = 0; epoch &lt; maxEpochs; epoch++)
/// {
///     dashboard.UpdateEpoch(epoch + 1, maxEpochs);
///
///     foreach (var batch in batches)
///     {
///         dashboard.UpdateBatch(batchIndex + 1, totalBatches);
///         // Train...
///     }
///
///     dashboard.LogMetric("train_loss", trainLoss);
///     dashboard.LogMetric("val_loss", valLoss);
/// }
///
/// dashboard.Stop();
/// </code>
/// </remarks>
public class MetricsDashboard : IDisposable
{
    private readonly string _runName;
    private readonly object _lock = new();
    private readonly Dictionary<string, List<(DateTime time, double value)>> _metricHistory;
    private readonly Dictionary<string, double> _currentMetrics;
    private readonly Dictionary<string, (double threshold, bool above)> _alerts;
    private readonly List<string> _alertMessages;

    private ProgressBar? _epochProgress;
    private ProgressBar? _batchProgress;
    private TrainingCurves? _trainingCurves;
    private Timer? _refreshTimer;
    private bool _isRunning;
    private bool _isDisposed;
    private DateTime _startTime;
    private int _currentEpoch;
    private int _totalEpochs;
    private int _currentBatch;
    private int _totalBatches;

    /// <summary>
    /// Gets or sets the refresh interval in milliseconds.
    /// </summary>
    public int RefreshIntervalMs { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to show the training curves chart.
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
    /// Gets whether the dashboard is currently running.
    /// </summary>
    public bool IsRunning => _isRunning;

    /// <summary>
    /// Event raised when an alert is triggered.
    /// </summary>
    public event EventHandler<AlertEventArgs>? AlertTriggered;

    /// <summary>
    /// Initializes a new instance of the MetricsDashboard class.
    /// </summary>
    /// <param name="runName">Name of the training run.</param>
    public MetricsDashboard(string runName = "Training")
    {
        _runName = runName;
        _metricHistory = new Dictionary<string, List<(DateTime, double)>>();
        _currentMetrics = new Dictionary<string, double>();
        _alerts = new Dictionary<string, (double, bool)>();
        _alertMessages = new List<string>();
    }

    /// <summary>
    /// Creates an ITrainingMonitor that automatically updates a dashboard display.
    /// </summary>
    /// <typeparam name="T">The numeric data type used for calculations.</typeparam>
    /// <param name="runName">Name for the training run displayed in the dashboard.</param>
    /// <returns>A TrainingMonitorDashboard that implements ITrainingMonitor and displays metrics.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This factory method creates a training monitor that shows a live dashboard.
    ///
    /// Use this with AiModelBuilder to get real-time training visualization:
    /// <code>
    /// // Create a dashboard-enabled training monitor
    /// using var monitor = MetricsDashboard.CreateTrainingMonitor&lt;double&gt;("My Training");
    ///
    /// // Configure the builder with the monitor
    /// var result = new AiModelBuilder&lt;double, Matrix, Vector&gt;()
    ///     .ConfigureTrainingMonitor(monitor)
    ///     .ConfigureModel(myModel)
    ///     .Build(trainingData, validationData);
    ///
    /// // The dashboard automatically updates during training!
    /// </code>
    ///
    /// The returned monitor:
    /// - Implements ITrainingMonitor for infrastructure integration
    /// - Displays real-time progress bars in the console
    /// - Shows training curves (loss, accuracy)
    /// - Tracks resource utilization
    /// - Triggers alerts when thresholds are crossed
    /// </remarks>
    public static TrainingMonitorDashboard<T> CreateTrainingMonitor<T>(string runName = "Training")
    {
        return new TrainingMonitorDashboard<T>(runName);
    }

    /// <summary>
    /// Starts the dashboard display.
    /// </summary>
    public void Start()
    {
        if (_isRunning)
            return;

        _isRunning = true;
        _startTime = DateTime.UtcNow;
        _trainingCurves = new TrainingCurves(80, 15, _runName);

        // Start refresh timer
        _refreshTimer = new Timer(RefreshDisplay, null, 0, RefreshIntervalMs);

        RenderHeader();
    }

    /// <summary>
    /// Stops the dashboard display.
    /// </summary>
    public void Stop()
    {
        if (!_isRunning)
            return;

        _isRunning = false;
        _refreshTimer?.Dispose();
        _refreshTimer = null;

        // Complete progress bars
        _epochProgress?.Complete();
        _batchProgress?.Complete();

        RenderSummary();
    }

    /// <summary>
    /// Updates the epoch progress.
    /// </summary>
    /// <param name="current">Current epoch (1-based).</param>
    /// <param name="total">Total epochs.</param>
    public void UpdateEpoch(int current, int total)
    {
        lock (_lock)
        {
            _currentEpoch = current;
            _totalEpochs = total;

            if (_epochProgress == null)
            {
                _epochProgress = new ProgressBar(total, "Epoch", useColors: UseColors);
            }
            else if (_epochProgress.Total != total)
            {
                _epochProgress.Dispose();
                _epochProgress = new ProgressBar(total, "Epoch", useColors: UseColors);
            }

            _epochProgress.Update(current);
        }
    }

    /// <summary>
    /// Updates the batch progress within an epoch.
    /// </summary>
    /// <param name="current">Current batch (1-based).</param>
    /// <param name="total">Total batches in epoch.</param>
    public void UpdateBatch(int current, int total)
    {
        lock (_lock)
        {
            _currentBatch = current;
            _totalBatches = total;

            if (_batchProgress == null)
            {
                _batchProgress = new ProgressBar(total, "Batch", barWidth: 30, useColors: UseColors);
            }
            else if (_batchProgress.Total != total)
            {
                _batchProgress.Dispose();
                _batchProgress = new ProgressBar(total, "Batch", barWidth: 30, useColors: UseColors);
            }

            _batchProgress.Update(current);
        }
    }

    /// <summary>
    /// Logs a metric value.
    /// </summary>
    /// <param name="name">The metric name.</param>
    /// <param name="value">The metric value.</param>
    public void LogMetric(string name, double value)
    {
        if (string.IsNullOrWhiteSpace(name))
            return;

        AlertEventArgs? pendingAlert = null;

        lock (_lock)
        {
            _currentMetrics[name] = value;

            if (!_metricHistory.ContainsKey(name))
            {
                _metricHistory[name] = new List<(DateTime, double)>();
            }
            _metricHistory[name].Add((DateTime.UtcNow, value));

            // Add to training curves
            _trainingCurves?.AddPoint(name, _currentEpoch, value);

            // Check alerts - capture alert data to raise event outside lock
            pendingAlert = CheckAlertAndCapture(name, value);
        }

        // Raise alert event outside the lock to prevent potential deadlocks
        if (pendingAlert != null)
        {
            AlertTriggered?.Invoke(this, pendingAlert);
        }
    }

    /// <summary>
    /// Logs multiple metrics at once.
    /// </summary>
    /// <param name="metrics">Dictionary of metric names and values.</param>
    public void LogMetrics(Dictionary<string, double> metrics)
    {
        if (metrics == null)
            return;

        foreach (var kvp in metrics)
        {
            LogMetric(kvp.Key, kvp.Value);
        }
    }

    /// <summary>
    /// Sets an alert threshold for a metric.
    /// </summary>
    /// <param name="metricName">The metric to monitor.</param>
    /// <param name="threshold">The threshold value.</param>
    /// <param name="triggerAbove">True to trigger when value goes above threshold, false for below.</param>
    public void SetAlert(string metricName, double threshold, bool triggerAbove = true)
    {
        lock (_lock)
        {
            _alerts[metricName] = (threshold, triggerAbove);
        }
    }

    /// <summary>
    /// Removes an alert for a metric.
    /// </summary>
    public void RemoveAlert(string metricName)
    {
        lock (_lock)
        {
            _alerts.Remove(metricName);
        }
    }

    /// <summary>
    /// Logs a status message.
    /// </summary>
    public void LogStatus(string message)
    {
        lock (_lock)
        {
            _epochProgress?.SetStatus(message);
        }
    }

    /// <summary>
    /// Logs resource utilization.
    /// </summary>
    public void LogResourceUsage(double cpuPercent, double memoryMB, double? gpuPercent = null, double? gpuMemoryMB = null)
    {
        lock (_lock)
        {
            _currentMetrics["cpu_percent"] = cpuPercent;
            _currentMetrics["memory_mb"] = memoryMB;

            if (gpuPercent.HasValue)
                _currentMetrics["gpu_percent"] = gpuPercent.Value;
            if (gpuMemoryMB.HasValue)
                _currentMetrics["gpu_memory_mb"] = gpuMemoryMB.Value;
        }
    }

    /// <summary>
    /// Gets the current value of a metric.
    /// </summary>
    public double? GetMetric(string name)
    {
        lock (_lock)
        {
            return _currentMetrics.TryGetValue(name, out var value) ? value : null;
        }
    }

    /// <summary>
    /// Gets the history of a metric.
    /// </summary>
    public List<(DateTime time, double value)> GetMetricHistory(string name)
    {
        lock (_lock)
        {
            if (_metricHistory.TryGetValue(name, out var history))
            {
                return new List<(DateTime, double)>(history);
            }
            return new List<(DateTime, double)>();
        }
    }

    /// <summary>
    /// Shows the training curves chart.
    /// </summary>
    public void ShowTrainingCurves()
    {
        lock (_lock)
        {
            _trainingCurves?.Render();
        }
    }

    /// <summary>
    /// Gets a summary of all metrics.
    /// </summary>
    public string GetSummary()
    {
        lock (_lock)
        {
            var lines = new List<string>
            {
                $"=== {_runName} Summary ===",
                $"Duration: {DateTime.UtcNow - _startTime:hh\\:mm\\:ss}",
                $"Epochs: {_currentEpoch}/{_totalEpochs}",
                ""
            };

            lines.Add("Metrics:");
            foreach (var kvp in _currentMetrics.OrderBy(k => k.Key))
            {
                lines.Add($"  {kvp.Key}: {kvp.Value:F4}");
            }

            if (_alertMessages.Count > 0)
            {
                lines.Add("");
                lines.Add("Alerts:");
                // Use Skip/Take for net471 compatibility (TakeLast not available)
                var recentAlerts = _alertMessages.Skip(Math.Max(0, _alertMessages.Count - 5)).Take(5);
                foreach (var alert in recentAlerts)
                {
                    lines.Add($"  {alert}");
                }
            }

            return string.Join(Environment.NewLine, lines);
        }
    }

    /// <summary>
    /// Checks if an alert should be triggered and returns the event args if so.
    /// Does not raise the event - caller must do that outside the lock.
    /// </summary>
    private AlertEventArgs? CheckAlertAndCapture(string metricName, double value)
    {
        if (!_alerts.TryGetValue(metricName, out var alert))
            return null;

        bool triggered = alert.above ? value > alert.threshold : value < alert.threshold;

        if (triggered)
        {
            var message = $"[{DateTime.UtcNow:HH:mm:ss}] ALERT: {metricName} = {value:F4} " +
                         $"({(alert.above ? "above" : "below")} threshold {alert.threshold:F4})";

            _alertMessages.Add(message);

            return new AlertEventArgs
            {
                MetricName = metricName,
                Value = value,
                Threshold = alert.threshold,
                TriggerAbove = alert.above,
                Message = message
            };
        }

        return null;
    }

    private void RefreshDisplay(object? state)
    {
        if (!_isRunning)
            return;

        // The progress bars auto-render on update
        // Additional display logic can be added here
    }

    private void RenderHeader()
    {
        SystemConsole.WriteLine();
        SystemConsole.WriteLine($"{'=',-60}");
        SystemConsole.WriteLine($"  {_runName}");
        SystemConsole.WriteLine($"  Started: {_startTime:yyyy-MM-dd HH:mm:ss} UTC");
        SystemConsole.WriteLine($"{'=',-60}");
        SystemConsole.WriteLine();
    }

    private void RenderSummary()
    {
        SystemConsole.WriteLine();
        SystemConsole.WriteLine($"{'=',-60}");
        SystemConsole.WriteLine("  Training Complete!");
        SystemConsole.WriteLine($"{'=',-60}");
        SystemConsole.WriteLine();
        SystemConsole.WriteLine(GetSummary());
        SystemConsole.WriteLine();

        if (ShowCurves && _metricHistory.Any(m => m.Value.Count > 1))
        {
            ShowTrainingCurves();
        }
    }

    /// <summary>
    /// Disposes the dashboard.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;
        Stop();

        _epochProgress?.Dispose();
        _batchProgress?.Dispose();
    }
}

/// <summary>
/// Event arguments for alert events.
/// </summary>
public class AlertEventArgs : EventArgs
{
    /// <summary>
    /// The name of the metric that triggered the alert.
    /// </summary>
    public string MetricName { get; set; } = string.Empty;

    /// <summary>
    /// The value that triggered the alert.
    /// </summary>
    public double Value { get; set; }

    /// <summary>
    /// The threshold that was crossed.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Whether the alert was triggered by going above the threshold.
    /// </summary>
    public bool TriggerAbove { get; set; }

    /// <summary>
    /// The alert message.
    /// </summary>
    public string Message { get; set; } = string.Empty;
}
