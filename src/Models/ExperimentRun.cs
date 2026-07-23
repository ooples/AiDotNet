using System.Collections.Concurrent;
using AiDotNet.Interfaces;
using AiDotNet.Serialization;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.Models;

/// <summary>
/// Represents a single training run within an experiment.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A run captures all the details of one training session,
/// including parameters, metrics, and artifacts.
/// <para>
/// This class is thread-safe and can be safely accessed from multiple threads concurrently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ExperimentRun<T> : IExperimentRun<T>
{
    /// <summary>
    /// Gets the unique identifier for this run.
    /// </summary>
    [JsonProperty]
    public string RunId { get; private set; }

    /// <summary>
    /// Gets the experiment ID this run belongs to.
    /// </summary>
    [JsonProperty]
    public string ExperimentId { get; private set; }

    /// <summary>
    /// Gets or sets the name of this run.
    /// </summary>
    public string? RunName { get; set; }

    /// <summary>
    /// Gets the timestamp when the run was started.
    /// </summary>
    [JsonProperty]
    public DateTime StartTime { get; private set; }

    /// <summary>
    /// Gets the timestamp when the run ended.
    /// </summary>
    [JsonProperty]
    public DateTime? EndTime
    {
        get { lock (_statusLock) { return _endTime; } }
        private set { lock (_statusLock) { _endTime = value; } }
    }

    /// <summary>
    /// Gets the current status of the run.
    /// </summary>
    [JsonProperty]
    public string Status
    {
        get { lock (_statusLock) { return _status; } }
        private set { lock (_statusLock) { _status = value; } }
    }

    /// <summary>
    /// Gets the tags associated with this run in a thread-safe manner.
    /// </summary>
    [JsonProperty]
    public ConcurrentDictionary<string, string> Tags { get; private set; }

    // Backing fields for synchronized Status and EndTime
    private readonly object _statusLock = new();
    private string _status = "Running";
    private DateTime? _endTime;

    // Thread-safe collections for concurrent access
    private readonly ConcurrentDictionary<string, object> _parameters;
    private readonly ConcurrentDictionary<string, List<(int Step, T Value, DateTime Timestamp)>> _metrics;
    private readonly ConcurrentBag<string> _artifacts;
    private readonly ConcurrentBag<(DateTime Timestamp, string Note)> _notes;
    private volatile string? _errorMessage;

    /// <summary>
    /// Private constructor for JSON deserialization.
    /// </summary>
    [JsonConstructor]
    private ExperimentRun()
    {
        RunId = string.Empty;
        ExperimentId = string.Empty;
        _status = "Running";
        Tags = new ConcurrentDictionary<string, string>();
        _parameters = new ConcurrentDictionary<string, object>();
        _metrics = new ConcurrentDictionary<string, List<(int, T, DateTime)>>();
        _artifacts = new ConcurrentBag<string>();
        _notes = new ConcurrentBag<(DateTime, string)>();
    }

    /// <summary>
    /// Initializes a new instance of the ExperimentRun class.
    /// </summary>
    /// <param name="experimentId">The experiment ID this run belongs to.</param>
    /// <param name="runName">Optional name for the run.</param>
    /// <param name="tags">Optional tags.</param>
    public ExperimentRun(string experimentId, string? runName = null, Dictionary<string, string>? tags = null)
    {
        RunId = Guid.NewGuid().ToString();
        Guard.NotNull(experimentId);
        ExperimentId = experimentId;
        RunName = runName;
        StartTime = DateTime.UtcNow;
        _status = "Running";
        Tags = tags != null
            ? new ConcurrentDictionary<string, string>(tags)
            : new ConcurrentDictionary<string, string>();
        _parameters = new ConcurrentDictionary<string, object>();
        _metrics = new ConcurrentDictionary<string, List<(int, T, DateTime)>>();
        _artifacts = new ConcurrentBag<string>();
        _notes = new ConcurrentBag<(DateTime, string)>();
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
            if (string.IsNullOrWhiteSpace(kvp.Key))
                throw new ArgumentException("Parameter key cannot be null or empty.", nameof(parameters));
            _parameters[kvp.Key] = kvp.Value;
        }
    }

    /// <summary>
    /// Logs a metric value at a specific step/iteration.
    /// </summary>
    public void LogMetric(string key, T value, int step = 0, DateTime? timestamp = null)
    {
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("Metric key cannot be null or empty.", nameof(key));

        var metricList = _metrics.GetOrAdd(key, _ => new List<(int, T, DateTime)>());
        lock (metricList)
        {
            metricList.Add((step, value, timestamp ?? DateTime.UtcNow));
        }
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
            if (string.IsNullOrWhiteSpace(kvp.Key))
                throw new ArgumentException("Metric key cannot be null or empty.", nameof(metrics));

            var metricList = _metrics.GetOrAdd(kvp.Key, _ => new List<(int, T, DateTime)>());
            lock (metricList)
            {
                metricList.Add((step, kvp.Value, ts));
            }
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
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Could not extract file name from path. Please provide an explicit artifact path.", nameof(localPath));

        _artifacts.Add(path);
    }

    /// <summary>
    /// Logs a directory of artifacts.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when localDir is null or empty.</exception>
    /// <exception cref="DirectoryNotFoundException">Thrown when localDir does not exist.</exception>
    /// <exception cref="UnauthorizedAccessException">Thrown when access to the directory or subdirectories is denied.</exception>
    public void LogArtifacts(string localDir, string? artifactPath = null)
    {
        if (string.IsNullOrWhiteSpace(localDir))
            throw new ArgumentException("Local directory cannot be null or empty.", nameof(localDir));

        if (!Directory.Exists(localDir))
            throw new DirectoryNotFoundException($"Directory not found: {localDir}");

        // Sanitize artifact path to prevent path traversal attacks
        var sanitizedArtifactPath = artifactPath != null ? GetSanitizedArtifactPath(artifactPath) : null;

        // Get files - may throw UnauthorizedAccessException for inaccessible subdirectories
        string[] files;
        try
        {
            files = Directory.GetFiles(localDir, "*", SearchOption.AllDirectories);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new UnauthorizedAccessException(
                $"Access denied to directory '{localDir}' or one of its subdirectories. " +
                "Ensure you have read permissions for all subdirectories.", ex);
        }

        var baseDirUri = new Uri(localDir.EndsWith(Path.DirectorySeparatorChar.ToString())
            ? localDir
            : localDir + Path.DirectorySeparatorChar);

        foreach (var file in files)
        {
            var fileUri = new Uri(file);
            var relativePath = Uri.UnescapeDataString(baseDirUri.MakeRelativeUri(fileUri).ToString())
                .Replace('/', Path.DirectorySeparatorChar);
            var computedPath = sanitizedArtifactPath != null ? Path.Combine(sanitizedArtifactPath, relativePath) : relativePath;
            _artifacts.Add(computedPath);
        }
    }

    /// <summary>
    /// Logs a trained model as an artifact.
    /// </summary>
    public void LogModel<TInput, TOutput, TMetadata>(IModel<TInput, TOutput, TMetadata> model, string? artifactPath = null)
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
        var modelTypeName = metadata?.GetType().Name;
        if (modelTypeName != null)
        {
            _parameters["model_type"] = modelTypeName;
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
        // Deep copy to prevent external modification of internal lists
        var result = new Dictionary<string, List<(int Step, T Value, DateTime Timestamp)>>();
        foreach (var kvp in _metrics)
        {
            lock (kvp.Value)
            {
                result[kvp.Key] = new List<(int Step, T Value, DateTime Timestamp)>(kvp.Value);
            }
        }
        return result;
    }

    /// <summary>
    /// Gets the latest value for a specific metric.
    /// </summary>
    public T? GetLatestMetric(string metricName)
    {
        if (metricName == null)
            throw new ArgumentNullException(nameof(metricName));

        if (!_metrics.TryGetValue(metricName, out var metricList))
            return default;

        lock (metricList)
        {
            if (metricList.Count == 0)
                return default;

            var latest = metricList.OrderByDescending(m => m.Step).First();
            return latest.Value;
        }
    }

    /// <summary>
    /// Gets all artifacts logged for this run.
    /// </summary>
    public List<string> GetArtifacts()
    {
        return _artifacts.ToList();
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
        // Return notes ordered by timestamp since ConcurrentBag doesn't preserve order
        return _notes.OrderBy(n => n.Timestamp).ToList();
    }

    /// <summary>
    /// Gets the error message if the run failed.
    /// </summary>
    public string? GetErrorMessage() => _errorMessage;

    /// <summary>
    /// Sanitizes an artifact path to prevent path traversal attacks.
    /// </summary>
    /// <param name="path">The path to sanitize.</param>
    /// <returns>The sanitized path without parent directory references.</returns>
    private static string GetSanitizedArtifactPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return string.Empty;

        // Remove any parent directory references to prevent path traversal
        var sanitized = path.Replace("..", string.Empty)
            .Replace("~", string.Empty);

        // Normalize path separators
        sanitized = sanitized.Replace('/', Path.DirectorySeparatorChar)
            .Replace('\\', Path.DirectorySeparatorChar);

        // Remove any leading or double separators
        while (sanitized.Contains(string.Concat(Path.DirectorySeparatorChar, Path.DirectorySeparatorChar)))
        {
            sanitized = sanitized.Replace(
                string.Concat(Path.DirectorySeparatorChar, Path.DirectorySeparatorChar),
                Path.DirectorySeparatorChar.ToString());
        }

        return sanitized.Trim(Path.DirectorySeparatorChar);
    }

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
