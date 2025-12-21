namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for training monitoring systems that track and visualize model training progress.
/// </summary>
/// <remarks>
/// A training monitor provides real-time visibility into the training process, tracking metrics,
/// system resources, and training state to help identify issues and optimize performance.
///
/// <b>For Beginners:</b> Think of a training monitor as a dashboard for your model training.
/// Just like a car dashboard shows speed, fuel, and engine temperature, a training monitor shows:
/// - Training metrics (loss, accuracy)
/// - System resources (CPU, GPU, memory usage)
/// - Training speed (iterations per second)
/// - Progress and estimated time remaining
///
/// Why training monitoring matters:
/// - Catch problems early (model not learning, overfitting, resource issues)
/// - Understand training dynamics and patterns
/// - Optimize resource usage
/// - Track progress on long-running training jobs
/// - Enable remote monitoring of training
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface ITrainingMonitor<T>
{
    /// <summary>
    /// Starts monitoring a training session.
    /// </summary>
    /// <param name="sessionName">Name for this training session.</param>
    /// <param name="metadata">Optional metadata about the training.</param>
    /// <returns>The unique identifier for the monitoring session.</returns>
    string StartSession(string sessionName, Dictionary<string, object>? metadata = null);

    /// <summary>
    /// Ends the current monitoring session.
    /// </summary>
    /// <param name="sessionId">The ID of the session to end.</param>
    void EndSession(string sessionId);

    /// <summary>
    /// Records a metric value for the current training step.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This logs a measurement from your training, like loss or accuracy.
    /// These values are tracked over time so you can see how training progresses.
    /// </remarks>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="metricName">Name of the metric (e.g., "train_loss", "val_accuracy").</param>
    /// <param name="value">The metric value.</param>
    /// <param name="step">The training step/iteration.</param>
    /// <param name="timestamp">Optional timestamp for the metric.</param>
    void LogMetric(string sessionId, string metricName, T value, int step, DateTime? timestamp = null);

    /// <summary>
    /// Records multiple metrics at once.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="metrics">Dictionary of metric names and values.</param>
    /// <param name="step">The training step/iteration.</param>
    void LogMetrics(string sessionId, Dictionary<string, T> metrics, int step);

    /// <summary>
    /// Records system resource usage.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tracks how much of your computer's resources
    /// (CPU, memory, GPU) are being used during training.
    /// </remarks>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="cpuUsage">CPU usage percentage (0-100).</param>
    /// <param name="memoryUsage">Memory usage in MB.</param>
    /// <param name="gpuUsage">GPU usage percentage (0-100), if applicable.</param>
    /// <param name="gpuMemory">GPU memory usage in MB, if applicable.</param>
    void LogResourceUsage(
        string sessionId,
        double cpuUsage,
        double memoryUsage,
        double? gpuUsage = null,
        double? gpuMemory = null);

    /// <summary>
    /// Updates the training progress information.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="currentStep">Current training step.</param>
    /// <param name="totalSteps">Total number of steps planned.</param>
    /// <param name="currentEpoch">Current epoch number.</param>
    /// <param name="totalEpochs">Total number of epochs planned.</param>
    void UpdateProgress(
        string sessionId,
        int currentStep,
        int totalSteps,
        int currentEpoch,
        int totalEpochs);

    /// <summary>
    /// Logs a text message or event during training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This lets you add notes or warnings during training,
    /// like "Started learning rate decay" or "Warning: High memory usage".
    /// </remarks>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="level">Severity level (Info, Warning, Error).</param>
    /// <param name="message">The message to log.</param>
    void LogMessage(string sessionId, LogLevel level, string message);

    /// <summary>
    /// Records the start of a new training epoch.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="epochNumber">The epoch number starting.</param>
    void OnEpochStart(string sessionId, int epochNumber);

    /// <summary>
    /// Records the end of a training epoch with summary metrics.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="epochNumber">The epoch number ending.</param>
    /// <param name="metrics">Summary metrics for the epoch.</param>
    /// <param name="duration">Time taken for the epoch.</param>
    void OnEpochEnd(string sessionId, int epochNumber, Dictionary<string, T> metrics, TimeSpan duration);

    /// <summary>
    /// Gets the current metrics for a session.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <returns>Dictionary of current metric values.</returns>
    Dictionary<string, T> GetCurrentMetrics(string sessionId);

    /// <summary>
    /// Gets the history of a specific metric.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="metricName">Name of the metric.</param>
    /// <returns>List of (step, value, timestamp) tuples for the metric.</returns>
    List<(int Step, T Value, DateTime Timestamp)> GetMetricHistory(string sessionId, string metricName);

    /// <summary>
    /// Gets statistics about training speed.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <returns>Statistics including steps/second, estimated time remaining.</returns>
    TrainingSpeedStats GetSpeedStats(string sessionId);

    /// <summary>
    /// Gets the current resource usage.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <returns>Current resource usage statistics.</returns>
    ResourceUsageStats GetResourceUsage(string sessionId);

    /// <summary>
    /// Checks for potential training issues and returns warnings.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This automatically detects common problems like:
    /// - Training loss not decreasing
    /// - Metrics showing NaN (not a number) values
    /// - Very high or low learning rates
    /// - Memory leaks
    /// </remarks>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <returns>List of detected issues and warnings.</returns>
    List<string> CheckForIssues(string sessionId);

    /// <summary>
    /// Exports monitoring data to a file.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="filePath">Path to save the export.</param>
    /// <param name="format">Export format (CSV, JSON, etc.).</param>
    void ExportData(string sessionId, string filePath, string format = "json");

    /// <summary>
    /// Creates a visualization of training metrics.
    /// </summary>
    /// <param name="sessionId">The ID of the monitoring session.</param>
    /// <param name="metricNames">Names of metrics to visualize.</param>
    /// <param name="outputPath">Path to save the visualization.</param>
    void CreateVisualization(string sessionId, List<string> metricNames, string outputPath);
}

/// <summary>
/// Log levels for training messages.
/// </summary>
public enum LogLevel
{
    /// <summary>Informational message.</summary>
    Info,
    /// <summary>Warning message.</summary>
    Warning,
    /// <summary>Error message.</summary>
    Error,
    /// <summary>Debug message.</summary>
    Debug
}
