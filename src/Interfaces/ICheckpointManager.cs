namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for checkpoint management systems that save and restore training state.
/// </summary>
/// <remarks>
/// A checkpoint manager handles saving and restoring the complete state of model training,
/// allowing you to pause and resume training, recover from failures, and track model evolution.
///
/// <b>For Beginners:</b> Think of checkpoints like save points in a video game. They let you:
/// - Save your progress so you can come back later
/// - Go back to an earlier point if something goes wrong
/// - Keep the best version you've found so far
///
/// Checkpoints typically save:
/// - Model parameters (weights and biases)
/// - Optimizer state (momentum, learning rate schedule, etc.)
/// - Training metadata (epoch number, step count)
/// - Performance metrics
///
/// Why checkpoint management matters:
/// - Training can be interrupted (crashes, time limits)
/// - You want to keep the best model even if later training makes it worse
/// - Long training runs need progress saved periodically
/// - Enables experimentation with different training strategies from same point
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface ICheckpointManager<T, TInput, TOutput>
{
    /// <summary>
    /// Saves a checkpoint of the current training state.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This saves everything about the current state of training
    /// so you can restore it later.
    /// </remarks>
    /// <param name="model">The model to checkpoint.</param>
    /// <param name="optimizer">The optimizer state to save.</param>
    /// <param name="epoch">The current training epoch.</param>
    /// <param name="step">The current training step.</param>
    /// <param name="metrics">Current performance metrics.</param>
    /// <param name="metadata">Additional metadata to save with the checkpoint.</param>
    /// <returns>The unique identifier for the saved checkpoint.</returns>
    string SaveCheckpoint(
        IModel<TInput, TOutput, TMetadata> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        Dictionary<string, object>? metadata = null) where TMetadata : class;

    /// <summary>
    /// Loads a checkpoint and restores the training state.
    /// </summary>
    /// <param name="checkpointId">The ID of the checkpoint to load.</param>
    /// <returns>A checkpoint object containing the restored state.</returns>
    Checkpoint<T, TInput, TOutput> LoadCheckpoint(string checkpointId);

    /// <summary>
    /// Loads the most recent checkpoint.
    /// </summary>
    /// <returns>The latest checkpoint, or null if none exist.</returns>
    Checkpoint<T, TInput, TOutput>? LoadLatestCheckpoint();

    /// <summary>
    /// Loads the checkpoint with the best metric value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This finds and loads the checkpoint where your model performed best
    /// according to a specific metric (like lowest loss or highest accuracy).
    /// </remarks>
    /// <param name="metricName">The name of the metric to optimize.</param>
    /// <param name="mode">Whether to minimize or maximize the metric.</param>
    /// <returns>The best checkpoint, or null if none exist.</returns>
    Checkpoint<T, TInput, TOutput>? LoadBestCheckpoint(string metricName, OptimizationMode mode);

    /// <summary>
    /// Lists all available checkpoints.
    /// </summary>
    /// <param name="sortBy">Optional metric to sort by.</param>
    /// <param name="descending">Whether to sort in descending order.</param>
    /// <returns>List of checkpoint metadata.</returns>
    List<CheckpointMetadata<T>> ListCheckpoints(string? sortBy = null, bool descending = true);

    /// <summary>
    /// Deletes a specific checkpoint.
    /// </summary>
    /// <param name="checkpointId">The ID of the checkpoint to delete.</param>
    void DeleteCheckpoint(string checkpointId);

    /// <summary>
    /// Deletes old checkpoints, keeping only a specified number of the most recent ones.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Checkpoints take up disk space, so this helps clean up old ones
    /// while keeping your most recent saves. It's like deleting old game saves to free up space.
    /// </remarks>
    /// <param name="keepLast">Number of recent checkpoints to keep.</param>
    /// <returns>Number of checkpoints deleted.</returns>
    int CleanupOldCheckpoints(int keepLast = 5);

    /// <summary>
    /// Deletes checkpoints except the best N according to a metric.
    /// </summary>
    /// <param name="metricName">The metric to use for determining best checkpoints.</param>
    /// <param name="keepBest">Number of best checkpoints to keep.</param>
    /// <param name="mode">Whether to minimize or maximize the metric.</param>
    /// <returns>Number of checkpoints deleted.</returns>
    int CleanupKeepBest(string metricName, int keepBest = 3, OptimizationMode mode = OptimizationMode.Both);

    /// <summary>
    /// Gets the storage path for checkpoints.
    /// </summary>
    string GetCheckpointDirectory();

    /// <summary>
    /// Sets up automatic checkpointing during training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This configures automatic saving, so checkpoints are created
    /// periodically without you having to manually save them.
    /// </remarks>
    /// <param name="saveFrequency">Save every N steps.</param>
    /// <param name="keepLast">Number of recent checkpoints to keep.</param>
    /// <param name="saveOnImprovement">Whether to save when metric improves.</param>
    /// <param name="metricName">Metric to track for improvement-based saving.</param>
    void ConfigureAutoCheckpointing(
        int saveFrequency,
        int keepLast = 5,
        bool saveOnImprovement = true,
        string? metricName = null);
}
