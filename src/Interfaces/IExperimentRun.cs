using System.Collections.Concurrent;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a single training run within an experiment.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A run represents one attempt at training a model with specific settings.
/// Each run records everything about that training session: what settings you used (parameters),
/// how well it performed (metrics), and what it created (artifacts like the trained model).
///
/// Think of a run like a single entry in a lab notebook - it captures all the details of one
/// specific experiment attempt.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ExperimentRun")]
public interface IExperimentRun<T>
{
    /// <summary>
    /// Gets the unique identifier for this run.
    /// </summary>
    string RunId { get; }

    /// <summary>
    /// Gets the experiment ID this run belongs to.
    /// </summary>
    string ExperimentId { get; }

    /// <summary>
    /// Gets or sets the name of this run.
    /// </summary>
    string? RunName { get; set; }

    /// <summary>
    /// Gets the timestamp when the run was started.
    /// </summary>
    DateTime StartTime { get; }

    /// <summary>
    /// Gets the timestamp when the run ended.
    /// </summary>
    DateTime? EndTime { get; }

    /// <summary>
    /// Gets the current status of the run (Running, Completed, Failed, etc.).
    /// </summary>
    string Status { get; }

    /// <summary>
    /// Gets the tags associated with this run.
    /// </summary>
    /// <remarks>
    /// This property returns a thread-safe concurrent dictionary for safe multi-threaded access.
    /// </remarks>
    ConcurrentDictionary<string, string> Tags { get; }

    /// <summary>
    /// Logs a single parameter value for this run.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Parameters are the settings you chose for this training run,
    /// like learning rate, batch size, or number of layers. These are typically set at the
    /// start of training and don't change during the run.
    /// </remarks>
    /// <param name="key">The parameter name.</param>
    /// <param name="value">The parameter value.</param>
    void LogParameter(string key, object value);

    /// <summary>
    /// Logs multiple parameters at once.
    /// </summary>
    /// <param name="parameters">Dictionary of parameter names and values.</param>
    void LogParameters(Dictionary<string, object> parameters);

    /// <summary>
    /// Logs a metric value at a specific step/iteration.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Metrics are measurements of how well your model is performing,
    /// like accuracy or loss. Unlike parameters, metrics change over time as the model trains.
    /// The step parameter lets you track how the metric changes over iterations.
    /// </remarks>
    /// <param name="key">The metric name (e.g., "loss", "accuracy").</param>
    /// <param name="value">The metric value.</param>
    /// <param name="step">The training step/iteration number.</param>
    /// <param name="timestamp">Optional timestamp for the metric.</param>
    void LogMetric(string key, T value, int step = 0, DateTime? timestamp = null);

    /// <summary>
    /// Logs multiple metrics at once for a specific step.
    /// </summary>
    /// <param name="metrics">Dictionary of metric names and values.</param>
    /// <param name="step">The training step/iteration number.</param>
    /// <param name="timestamp">Optional timestamp for the metrics.</param>
    void LogMetrics(Dictionary<string, T> metrics, int step = 0, DateTime? timestamp = null);

    /// <summary>
    /// Logs an artifact (file) associated with this run.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Artifacts are files produced during training, like:
    /// - Saved models
    /// - Training plots/visualizations
    /// - Configuration files
    /// - Log files
    /// </remarks>
    /// <param name="localPath">Path to the file to log.</param>
    /// <param name="artifactPath">Optional path within the artifact store.</param>
    void LogArtifact(string localPath, string? artifactPath = null);

    /// <summary>
    /// Logs a directory of artifacts.
    /// </summary>
    /// <param name="localDir">Path to the directory to log.</param>
    /// <param name="artifactPath">Optional path within the artifact store.</param>
    void LogArtifacts(string localDir, string? artifactPath = null);

    /// <summary>
    /// Logs a trained model as an artifact.
    /// </summary>
    /// <param name="model">The model to log.</param>
    /// <param name="artifactPath">Optional path within the artifact store.</param>
    void LogModel<TInput, TOutput, TMetadata>(IModel<TInput, TOutput, TMetadata> model, string? artifactPath = null) where TInput : class where TOutput : class where TMetadata : class;

    /// <summary>
    /// Gets all parameters logged for this run.
    /// </summary>
    /// <returns>Dictionary of parameter names and values.</returns>
    Dictionary<string, object> GetParameters();

    /// <summary>
    /// Gets all metrics logged for this run.
    /// </summary>
    /// <returns>Dictionary where keys are metric names and values are lists of (step, value, timestamp) tuples.</returns>
    Dictionary<string, List<(int Step, T Value, DateTime Timestamp)>> GetMetrics();

    /// <summary>
    /// Gets the latest value for a specific metric.
    /// </summary>
    /// <param name="metricName">The name of the metric.</param>
    /// <returns>The latest metric value, or null if not found.</returns>
    T? GetLatestMetric(string metricName);

    /// <summary>
    /// Gets all artifacts logged for this run.
    /// </summary>
    /// <returns>List of artifact paths.</returns>
    List<string> GetArtifacts();

    /// <summary>
    /// Marks the run as completed successfully.
    /// </summary>
    void Complete();

    /// <summary>
    /// Marks the run as failed with an optional error message.
    /// </summary>
    /// <param name="errorMessage">Description of the failure.</param>
    void Fail(string? errorMessage = null);

    /// <summary>
    /// Adds a note or comment to the run.
    /// </summary>
    /// <param name="note">The note content.</param>
    void AddNote(string note);

    /// <summary>
    /// Gets all notes added to this run.
    /// </summary>
    /// <returns>List of notes with timestamps.</returns>
    List<(DateTime Timestamp, string Note)> GetNotes();
}
