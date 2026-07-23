namespace AiDotNet.TrainingMonitoring.ExperimentTracking;

/// <summary>
/// Status of an experiment run.
/// </summary>
public enum RunStatus
{
    /// <summary>Run is in progress.</summary>
    Running,
    /// <summary>Run completed successfully.</summary>
    Completed,
    /// <summary>Run failed with an error.</summary>
    Failed,
    /// <summary>Run was killed/stopped.</summary>
    Killed,
    /// <summary>Run is scheduled but not started.</summary>
    Scheduled
}

/// <summary>
/// Interface for experiment tracking systems.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Experiment tracking is essential for machine learning
/// workflows. It helps you:
/// - Keep track of different experiments and their parameters
/// - Compare results across runs
/// - Reproduce experiments by logging all relevant information
/// - Organize and search through past experiments
///
/// Think of it like MLflow or Weights &amp; Biases - it's a central place
/// to log everything about your training runs.
///
/// Key concepts:
/// - Experiment: A named project (e.g., "image-classification")
/// - Run: A single execution of training (e.g., "run_20241220_143052")
/// - Parameters: Configuration values (learning_rate, batch_size, etc.)
/// - Metrics: Measured values (loss, accuracy, etc.)
/// - Artifacts: Files produced (model weights, plots, etc.)
/// - Tags: Labels for organization (dev, production, etc.)
/// </remarks>
public interface IExperimentTracker : IDisposable
{
    /// <summary>
    /// Gets the tracking URI.
    /// </summary>
    string TrackingUri { get; }

    /// <summary>
    /// Gets the active experiment name, if any.
    /// </summary>
    string? ActiveExperiment { get; }

    /// <summary>
    /// Gets the active run ID, if any.
    /// </summary>
    string? ActiveRunId { get; }

    /// <summary>
    /// Creates or gets an experiment by name.
    /// </summary>
    /// <param name="name">Experiment name.</param>
    /// <param name="description">Optional description.</param>
    /// <param name="tags">Optional tags.</param>
    /// <returns>Experiment information.</returns>
    ExperimentInfo CreateExperiment(string name, string? description = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Gets an experiment by name.
    /// </summary>
    /// <param name="name">Experiment name.</param>
    /// <returns>Experiment info or null if not found.</returns>
    ExperimentInfo? GetExperiment(string name);

    /// <summary>
    /// Lists all experiments.
    /// </summary>
    /// <returns>List of experiments.</returns>
    List<ExperimentInfo> ListExperiments();

    /// <summary>
    /// Sets the active experiment for subsequent runs.
    /// </summary>
    /// <param name="name">Experiment name.</param>
    void SetExperiment(string name);

    /// <summary>
    /// Starts a new run within the active experiment.
    /// </summary>
    /// <param name="runName">Optional run name.</param>
    /// <param name="tags">Optional run tags.</param>
    /// <param name="description">Optional run description.</param>
    /// <returns>Run information.</returns>
    RunInfo StartRun(string? runName = null, Dictionary<string, string>? tags = null, string? description = null);

    /// <summary>
    /// Ends the active run.
    /// </summary>
    /// <param name="status">Final status of the run.</param>
    void EndRun(RunStatus status = RunStatus.Completed);

    /// <summary>
    /// Gets a run by ID.
    /// </summary>
    /// <param name="runId">Run ID.</param>
    /// <returns>Run info or null if not found.</returns>
    RunInfo? GetRun(string runId);

    /// <summary>
    /// Lists runs in an experiment.
    /// </summary>
    /// <param name="experimentName">Experiment name.</param>
    /// <param name="filter">Optional filter query.</param>
    /// <param name="orderBy">Optional ordering.</param>
    /// <param name="maxResults">Maximum results to return.</param>
    /// <returns>List of runs.</returns>
    List<RunInfo> ListRuns(string? experimentName = null, string? filter = null, string? orderBy = null, int maxResults = 100);

    /// <summary>
    /// Logs a parameter (configuration value).
    /// </summary>
    /// <param name="key">Parameter name.</param>
    /// <param name="value">Parameter value.</param>
    void LogParameter(string key, string value);

    /// <summary>
    /// Logs multiple parameters.
    /// </summary>
    /// <param name="parameters">Dictionary of parameters.</param>
    void LogParameters(Dictionary<string, string> parameters);

    /// <summary>
    /// Logs a metric at a specific step.
    /// </summary>
    /// <param name="key">Metric name.</param>
    /// <param name="value">Metric value.</param>
    /// <param name="step">Training step (optional).</param>
    void LogMetric(string key, double value, long? step = null);

    /// <summary>
    /// Logs multiple metrics at a specific step.
    /// </summary>
    /// <param name="metrics">Dictionary of metrics.</param>
    /// <param name="step">Training step (optional).</param>
    void LogMetrics(Dictionary<string, double> metrics, long? step = null);

    /// <summary>
    /// Sets a tag on the active run.
    /// </summary>
    /// <param name="key">Tag key.</param>
    /// <param name="value">Tag value.</param>
    void SetTag(string key, string value);

    /// <summary>
    /// Sets multiple tags on the active run.
    /// </summary>
    /// <param name="tags">Dictionary of tags.</param>
    void SetTags(Dictionary<string, string> tags);

    /// <summary>
    /// Logs an artifact file.
    /// </summary>
    /// <param name="localPath">Path to the local file.</param>
    /// <param name="artifactPath">Optional path within the artifact directory.</param>
    void LogArtifact(string localPath, string? artifactPath = null);

    /// <summary>
    /// Logs all files in a directory as artifacts.
    /// </summary>
    /// <param name="localDir">Path to the local directory.</param>
    /// <param name="artifactPath">Optional path within the artifact directory.</param>
    void LogArtifacts(string localDir, string? artifactPath = null);

    /// <summary>
    /// Logs a model artifact.
    /// </summary>
    /// <param name="modelPath">Path to the model file or directory.</param>
    /// <param name="modelName">Name for the model.</param>
    /// <param name="metadata">Optional model metadata.</param>
    void LogModel(string modelPath, string modelName, Dictionary<string, object>? metadata = null);

    /// <summary>
    /// Gets metric history for a run.
    /// </summary>
    /// <param name="runId">Run ID.</param>
    /// <param name="metricKey">Metric name.</param>
    /// <returns>List of metric values with steps.</returns>
    List<MetricValue> GetMetricHistory(string runId, string metricKey);

    /// <summary>
    /// Compares multiple runs.
    /// </summary>
    /// <param name="runIds">Run IDs to compare.</param>
    /// <returns>Comparison data.</returns>
    RunComparison CompareRuns(params string[] runIds);

    /// <summary>
    /// Searches for runs matching criteria.
    /// </summary>
    /// <param name="experimentNames">Experiment names to search.</param>
    /// <param name="filter">Filter expression (e.g., "metrics.accuracy > 0.9").</param>
    /// <param name="orderBy">Order by expression (e.g., "metrics.loss ASC").</param>
    /// <param name="maxResults">Maximum results.</param>
    /// <returns>Matching runs.</returns>
    List<RunInfo> SearchRuns(
        IEnumerable<string>? experimentNames = null,
        string? filter = null,
        string? orderBy = null,
        int maxResults = 100);

    /// <summary>
    /// Deletes a run.
    /// </summary>
    /// <param name="runId">Run ID to delete.</param>
    void DeleteRun(string runId);

    /// <summary>
    /// Restores a deleted run.
    /// </summary>
    /// <param name="runId">Run ID to restore.</param>
    void RestoreRun(string runId);

    /// <summary>
    /// Deletes an experiment and all its runs.
    /// </summary>
    /// <param name="experimentName">Experiment name.</param>
    void DeleteExperiment(string experimentName);
}

/// <summary>
/// Information about an experiment.
/// </summary>
public class ExperimentInfo
{
    /// <summary>
    /// Gets or sets the experiment ID.
    /// </summary>
    public string ExperimentId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the experiment name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets when the experiment was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets when the experiment was last updated.
    /// </summary>
    public DateTime LastUpdatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the number of runs.
    /// </summary>
    public int RunCount { get; set; }

    /// <summary>
    /// Gets or sets the artifact location.
    /// </summary>
    public string? ArtifactLocation { get; set; }

    /// <summary>
    /// Gets or sets tags.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets whether the experiment is deleted.
    /// </summary>
    public bool IsDeleted { get; set; }
}

/// <summary>
/// Information about a run.
/// </summary>
public class RunInfo
{
    /// <summary>
    /// Gets or sets the run ID.
    /// </summary>
    public string RunId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the run name.
    /// </summary>
    public string? RunName { get; set; }

    /// <summary>
    /// Gets or sets the experiment ID.
    /// </summary>
    public string ExperimentId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the experiment name.
    /// </summary>
    public string ExperimentName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the run status.
    /// </summary>
    public RunStatus Status { get; set; } = RunStatus.Running;

    /// <summary>
    /// Gets or sets when the run started.
    /// </summary>
    public DateTime StartTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets when the run ended.
    /// </summary>
    public DateTime? EndTime { get; set; }

    /// <summary>
    /// Gets the run duration.
    /// </summary>
    public TimeSpan? Duration => EndTime.HasValue ? EndTime.Value - StartTime : null;

    /// <summary>
    /// Gets or sets the run description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the user who started the run.
    /// </summary>
    public string? User { get; set; }

    /// <summary>
    /// Gets or sets source information (script, git, etc.).
    /// </summary>
    public SourceInfo? Source { get; set; }

    /// <summary>
    /// Gets or sets parameters.
    /// </summary>
    public Dictionary<string, string> Parameters { get; set; } = new();

    /// <summary>
    /// Gets or sets the latest metric values.
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>
    /// Gets or sets tags.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets artifact paths.
    /// </summary>
    public List<string> Artifacts { get; set; } = new();

    /// <summary>
    /// Gets or sets whether the run is deleted.
    /// </summary>
    public bool IsDeleted { get; set; }
}

/// <summary>
/// Source information for a run.
/// </summary>
public class SourceInfo
{
    /// <summary>
    /// Gets or sets the source type.
    /// </summary>
    public string SourceType { get; set; } = "LOCAL";

    /// <summary>
    /// Gets or sets the source name (e.g., script path).
    /// </summary>
    public string? SourceName { get; set; }

    /// <summary>
    /// Gets or sets the git commit hash.
    /// </summary>
    public string? GitCommit { get; set; }

    /// <summary>
    /// Gets or sets the git branch.
    /// </summary>
    public string? GitBranch { get; set; }

    /// <summary>
    /// Gets or sets the git repository URL.
    /// </summary>
    public string? GitRepoUrl { get; set; }

    /// <summary>
    /// Gets or sets the entry point.
    /// </summary>
    public string? EntryPoint { get; set; }
}

/// <summary>
/// A metric value with step information.
/// </summary>
public class MetricValue
{
    /// <summary>
    /// Gets or sets the metric key.
    /// </summary>
    public string Key { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the metric value.
    /// </summary>
    public double Value { get; set; }

    /// <summary>
    /// Gets or sets the step number.
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Gets or sets the timestamp.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Comparison of multiple runs.
/// </summary>
public class RunComparison
{
    /// <summary>
    /// Gets or sets the runs being compared.
    /// </summary>
    public List<RunInfo> Runs { get; set; } = new();

    /// <summary>
    /// Gets or sets parameter differences.
    /// </summary>
    public Dictionary<string, Dictionary<string, string>> ParameterComparison { get; set; } = new();

    /// <summary>
    /// Gets or sets metric comparisons.
    /// </summary>
    public Dictionary<string, Dictionary<string, double>> MetricComparison { get; set; } = new();

    /// <summary>
    /// Gets the best run for a given metric.
    /// </summary>
    /// <param name="metricKey">Metric name.</param>
    /// <param name="minimize">Whether to minimize (true) or maximize (false).</param>
    /// <returns>Run ID of the best run.</returns>
    public string? GetBestRun(string metricKey, bool minimize = true)
    {
        if (!MetricComparison.TryGetValue(metricKey, out var values) || values.Count == 0)
            return null;

        return minimize
            ? values.MinBy(kvp => kvp.Value).Key
            : values.MaxBy(kvp => kvp.Value).Key;
    }

    /// <summary>
    /// Gets a summary of the comparison.
    /// </summary>
    public string GetSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Comparing {Runs.Count} runs:");

        foreach (var run in Runs)
        {
            sb.AppendLine($"  - {run.RunId} ({run.RunName ?? "unnamed"})");
        }

        sb.AppendLine("\nMetric Comparison:");
        foreach (var kvp in MetricComparison)
        {
            sb.AppendLine($"  {kvp.Key}:");
            foreach (var runMetric in kvp.Value.OrderBy(v => v.Value))
            {
                sb.AppendLine($"    {runMetric.Key}: {runMetric.Value:F6}");
            }
        }

        return sb.ToString();
    }
}
