using AiDotNet.Interfaces;
using AiDotNet.Serialization;
using System.Text.Json;

namespace AiDotNet.Models;

/// <summary>
/// Represents a single training run within an experiment.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A run captures all the details of one training session,
/// including parameters, metrics, and artifacts.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ExperimentRun<T> : IExperimentRun<T>
{
    /// <summary>
    /// Gets the unique identifier for this run.
    /// </summary>
    public string RunId { get; private set; }

    /// <summary>
    /// Gets the experiment ID this run belongs to.
    /// </summary>
    public string ExperimentId { get; private set; }

    /// <summary>
    /// Gets or sets the name of this run.
    /// </summary>
    public string? RunName { get; set; }

    /// <summary>
    /// Gets the timestamp when the run was started.
    /// </summary>
    public DateTime StartTime { get; private set; }

    /// <summary>
    /// Gets the timestamp when the run ended.
    /// </summary>
    public DateTime? EndTime { get; private set; }

    /// <summary>
    /// Gets the current status of the run.
    /// </summary>
    public string Status { get; private set; }

    /// <summary>
    /// Gets or sets tags associated with this run.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; }

    private readonly Dictionary<string, object> _parameters;
    private readonly Dictionary<string, List<(int Step, T Value, DateTime Timestamp)>> _metrics;
    private readonly List<string> _artifacts;
    private readonly List<(DateTime Timestamp, string Note)> _notes;
    private string? _errorMessage;

    /// <summary>
    /// Initializes a new instance of the ExperimentRun class.
    /// </summary>
    /// <param name="experimentId">The experiment ID this run belongs to.</param>
    /// <param name="runName">Optional name for the run.</param>
    /// <param name="tags">Optional tags.</param>
    public ExperimentRun(string experimentId, string? runName = null, Dictionary<string, string>? tags = null)
    {
        RunId = Guid.NewGuid().ToString();
        ExperimentId = experimentId ?? throw new ArgumentNullException(nameof(experimentId));
        RunName = runName;
        StartTime = DateTime.UtcNow;
        Status = "Running";
        Tags = tags ?? new Dictionary<string, string>();
        _parameters = new Dictionary<string, object>();
        _metrics = new Dictionary<string, List<(int, T, DateTime)>>();
        _artifacts = new List<string>();
        _notes = new List<(DateTime, string)>();
    }

    /// <summary>
    /// Logs a single parameter value for this run.
    /// </summary>
    public void LogParameter(string key, object value)
    {
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("Parameter key cannot be null or empty.", nameof(key));

        _parameters[key] = value;
    }

    /// <summary>
    /// Logs multiple parameters at once.
    /// </summary>
    public void LogParameters(Dictionary<string, object> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        foreach (var kvp in parameters)
        {
            LogParameter(kvp.Key, kvp.Value);
        }
    }

    /// <summary>
    /// Logs a metric value at a specific step/iteration.
    /// </summary>
    public void LogMetric(string key, T value, int step = 0, DateTime? timestamp = null)
    {
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("Metric key cannot be null or empty.", nameof(key));

        if (!_metrics.ContainsKey(key))
        {
            _metrics[key] = new List<(int, T, DateTime)>();
        }

        _metrics[key].Add((step, value, timestamp ?? DateTime.UtcNow));
    }

    /// <summary>
    /// Logs multiple metrics at once for a specific step.
    /// </summary>
    public void LogMetrics(Dictionary<string, T> metrics, int step = 0, DateTime? timestamp = null)
    {
        if (metrics == null)
            throw new ArgumentNullException(nameof(metrics));

        var ts = timestamp ?? DateTime.UtcNow;
        foreach (var kvp in metrics)
        {
            LogMetric(kvp.Key, kvp.Value, step, ts);
        }
    }

    /// <summary>
    /// Logs an artifact (file) associated with this run.
    /// </summary>
    public void LogArtifact(string localPath, string? artifactPath = null)
    {
        if (string.IsNullOrWhiteSpace(localPath))
            throw new ArgumentException("Local path cannot be null or empty.", nameof(localPath));

        var path = artifactPath ?? Path.GetFileName(localPath);
        _artifacts.Add(path);
    }

    /// <summary>
    /// Logs a directory of artifacts.
    /// </summary>
    public void LogArtifacts(string localDir, string? artifactPath = null)
    {
        if (string.IsNullOrWhiteSpace(localDir))
            throw new ArgumentException("Local directory cannot be null or empty.", nameof(localDir));

        if (!Directory.Exists(localDir))
            throw new DirectoryNotFoundException($"Directory not found: {localDir}");

        var files = Directory.GetFiles(localDir, "*", SearchOption.AllDirectories);
        foreach (var file in files)
        {
            var relativePath = Path.GetRelativePath(localDir, file);
            var fullPath = artifactPath != null ? Path.Combine(artifactPath, relativePath) : relativePath;
            _artifacts.Add(fullPath);
        }
    }

    /// <summary>
    /// Logs a trained model as an artifact.
    /// </summary>
    public void LogModel(IModel<TInput, TOutput, TMetadata> model, string? artifactPath = null)
        where TInput : class
        where TOutput : class
        where TMetadata : class
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        var path = artifactPath ?? "model";
        _artifacts.Add(path);

        // Add model metadata as parameters
        var metadata = model.GetModelMetadata();
        if (metadata != null)
        {
            LogParameter("model_type", metadata.GetType().Name);
        }
    }

    /// <summary>
    /// Gets all parameters logged for this run.
    /// </summary>
    public Dictionary<string, object> GetParameters()
    {
        return new Dictionary<string, object>(_parameters);
    }

    /// <summary>
    /// Gets all metrics logged for this run.
    /// </summary>
    public Dictionary<string, List<(int Step, T Value, DateTime Timestamp)>> GetMetrics()
    {
        return new Dictionary<string, List<(int, T, DateTime)>>(_metrics);
    }

    /// <summary>
    /// Gets the latest value for a specific metric.
    /// </summary>
    public T? GetLatestMetric(string metricName)
    {
        if (!_metrics.ContainsKey(metricName) || _metrics[metricName].Count == 0)
            return default;

        var latest = _metrics[metricName].OrderByDescending(m => m.Step).First();
        return latest.Value;
    }

    /// <summary>
    /// Gets all artifacts logged for this run.
    /// </summary>
    public List<string> GetArtifacts()
    {
        return new List<string>(_artifacts);
    }

    /// <summary>
    /// Marks the run as completed successfully.
    /// </summary>
    public void Complete()
    {
        Status = "Completed";
        EndTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Marks the run as failed with an optional error message.
    /// </summary>
    public void Fail(string? errorMessage = null)
    {
        Status = "Failed";
        EndTime = DateTime.UtcNow;
        _errorMessage = errorMessage;
        if (errorMessage != null)
        {
            AddNote($"Run failed: {errorMessage}");
        }
    }

    /// <summary>
    /// Adds a note or comment to the run.
    /// </summary>
    public void AddNote(string note)
    {
        if (string.IsNullOrWhiteSpace(note))
            throw new ArgumentException("Note cannot be null or empty.", nameof(note));

        _notes.Add((DateTime.UtcNow, note));
    }

    /// <summary>
    /// Gets all notes added to this run.
    /// </summary>
    public List<(DateTime Timestamp, string Note)> GetNotes()
    {
        return new List<(DateTime, string)>(_notes);
    }

    /// <summary>
    /// Gets the error message if the run failed.
    /// </summary>
    public string? GetErrorMessage() => _errorMessage;

    /// <summary>
    /// Gets the duration of the run.
    /// </summary>
    public TimeSpan? GetDuration()
    {
        if (EndTime.HasValue)
            return EndTime.Value - StartTime;

        if (Status == "Running")
            return DateTime.UtcNow - StartTime;

        return null;
    }
}
