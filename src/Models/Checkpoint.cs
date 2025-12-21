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
    /// Note: The model is stored as object for serialization compatibility.
    /// </summary>
    public object? Model { get; set; }

    /// <summary>
    /// Gets or sets the optimizer state as a serializable dictionary.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Instead of storing the optimizer object directly (which can
    /// cause serialization issues with interfaces), we store its state as a dictionary.
    /// The optimizer can be reconstructed from this state when loading the checkpoint.
    ///
    /// Key values typically include:
    /// - "LearningRate": The current learning rate
    /// - "OptimizerType": The type name of the optimizer
    /// - "Parameters": Any additional optimizer-specific parameters
    /// </remarks>
    public Dictionary<string, object> OptimizerState { get; set; } = new();

    /// <summary>
    /// Gets or sets the optimizer type name for reconstruction.
    /// </summary>
    public string? OptimizerTypeName { get; set; }

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
    /// <param name="model">The model to save.</param>
    /// <param name="optimizerState">The optimizer state dictionary.</param>
    /// <param name="optimizerTypeName">The optimizer type name for reconstruction.</param>
    /// <param name="epoch">The current epoch.</param>
    /// <param name="step">The current step.</param>
    /// <param name="metrics">Performance metrics at this checkpoint.</param>
    /// <param name="metadata">Additional metadata.</param>
    public Checkpoint(
        object model,
        Dictionary<string, object> optimizerState,
        string? optimizerTypeName,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        Dictionary<string, object>? metadata = null)
        : this()
    {
        Model = model;
        OptimizerState = optimizerState ?? new Dictionary<string, object>();
        OptimizerTypeName = optimizerTypeName;
        Epoch = epoch;
        Step = step;
        Metrics = metrics ?? new Dictionary<string, T>();
        Metadata = metadata ?? new Dictionary<string, object>();
    }

    /// <summary>
    /// Initializes a new instance of the Checkpoint class with an optimizer object.
    /// </summary>
    /// <param name="model">The model to save.</param>
    /// <param name="optimizer">The optimizer to extract state from.</param>
    /// <param name="epoch">The current epoch.</param>
    /// <param name="step">The current step.</param>
    /// <param name="metrics">Performance metrics at this checkpoint.</param>
    /// <param name="metadata">Additional metadata.</param>
    public Checkpoint(
        object model,
        IOptimizer<T, TInput, TOutput> optimizer,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        Dictionary<string, object>? metadata = null)
        : this()
    {
        Model = model;
        OptimizerState = ExtractOptimizerState(optimizer);
        OptimizerTypeName = optimizer?.GetType().AssemblyQualifiedName;
        Epoch = epoch;
        Step = step;
        Metrics = metrics ?? new Dictionary<string, T>();
        Metadata = metadata ?? new Dictionary<string, object>();
    }

    /// <summary>
    /// Extracts serializable state from an optimizer.
    /// </summary>
    private static Dictionary<string, object> ExtractOptimizerState(IOptimizer<T, TInput, TOutput>? optimizer)
    {
        if (optimizer == null)
            return new Dictionary<string, object>();

        var state = new Dictionary<string, object>
        {
            ["TypeName"] = optimizer.GetType().FullName ?? "Unknown"
        };

        // Extract common optimizer properties using reflection
        var type = optimizer.GetType();

        // Try to get learning rate
        var lrProp = type.GetProperty("LearningRate") ?? type.GetProperty("CurrentLearningRate");
        if (lrProp != null)
        {
            var value = lrProp.GetValue(optimizer);
            if (value != null)
                state["LearningRate"] = value;
        }

        // Try to get momentum
        var momentumProp = type.GetProperty("Momentum");
        if (momentumProp != null)
        {
            var value = momentumProp.GetValue(optimizer);
            if (value != null)
                state["Momentum"] = value;
        }

        // Try to get weight decay
        var wdProp = type.GetProperty("WeightDecay") ?? type.GetProperty("L2Regularization");
        if (wdProp != null)
        {
            var value = wdProp.GetValue(optimizer);
            if (value != null)
                state["WeightDecay"] = value;
        }

        return state;
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
