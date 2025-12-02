using AiDotNet.Interfaces;

namespace AiDotNet.Models;

/// <summary>
/// Represents a saved checkpoint of model training state.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A checkpoint is like a save point in a video game - it captures
/// everything needed to resume training from that exact point.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class Checkpoint<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the unique identifier for this checkpoint.
    /// </summary>
    public string CheckpointId { get; private set; }

    /// <summary>
    /// Gets or sets the model at this checkpoint.
    /// </summary>
    public IModel<TInput, TOutput, TMetadata>? Model { get; set; }

    /// <summary>
    /// Gets or sets the optimizer state.
    /// </summary>
    public IOptimizer<T, TInput, TOutput>? Optimizer { get; set; }

    /// <summary>
    /// Gets or sets the training epoch number.
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// Gets or sets the training step number.
    /// </summary>
    public int Step { get; set; }

    /// <summary>
    /// Gets or sets the performance metrics at this checkpoint.
    /// </summary>
    public Dictionary<string, T> Metrics { get; set; }

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; }

    /// <summary>
    /// Gets the timestamp when this checkpoint was created.
    /// </summary>
    public DateTime CreatedAt { get; private set; }

    /// <summary>
    /// Gets or sets the file path where the checkpoint is stored.
    /// </summary>
    public string? FilePath { get; set; }

    /// <summary>
    /// Initializes a new instance of the Checkpoint class.
    /// </summary>
    public Checkpoint()
    {
        CheckpointId = Guid.NewGuid().ToString();
        Metrics = new Dictionary<string, T>();
        Metadata = new Dictionary<string, object>();
        CreatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Initializes a new instance of the Checkpoint class with specified values.
    /// </summary>
    public Checkpoint(
        IModel<TInput, TOutput, TMetadata> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        Dictionary<string, object>? metadata = null)
        : this()
    {
        Model = model;
        Optimizer = optimizer;
        Epoch = epoch;
        Step = step;
        Metrics = metrics ?? new Dictionary<string, T>();
        Metadata = metadata ?? new Dictionary<string, object>();
    }
}

/// <summary>
/// Contains metadata about a checkpoint without loading the full checkpoint data.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public class CheckpointMetadata<T>
{
    /// <summary>
    /// Gets or sets the checkpoint ID.
    /// </summary>
    public string CheckpointId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the epoch number.
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// Gets or sets the step number.
    /// </summary>
    public int Step { get; set; }

    /// <summary>
    /// Gets or sets the metrics.
    /// </summary>
    public Dictionary<string, T> Metrics { get; set; } = new();

    /// <summary>
    /// Gets or sets the creation timestamp.
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Gets or sets the file path.
    /// </summary>
    public string? FilePath { get; set; }

    /// <summary>
    /// Gets or sets the file size in bytes.
    /// </summary>
    public long FileSizeBytes { get; set; }
}
