using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Enums;
using Newtonsoft.Json;

namespace AiDotNet.CheckpointManagement;

/// <summary>
/// Implementation of checkpoint management for saving and restoring training state.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This manages checkpoints (save points) of your model training,
/// allowing you to save progress, resume training, and keep track of the best models.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public class CheckpointManager<T, TInput, TOutput> : CheckpointManagerBase<T, TInput, TOutput>
{
    private readonly Dictionary<string, CheckpointMetadata<T>> _checkpoints;

    /// <summary>
    /// Initializes a new instance of the CheckpointManager class.
    /// </summary>
    /// <param name="checkpointDirectory">Directory to store checkpoints. Defaults to "./checkpoints".</param>
    public CheckpointManager(string? checkpointDirectory = null) : base(checkpointDirectory)
    {
        _checkpoints = new Dictionary<string, CheckpointMetadata<T>>();

        // Load existing checkpoints
        LoadExistingCheckpoints();
    }

    /// <summary>
    /// Saves a checkpoint of the current training state.
    /// </summary>
    public override string SaveCheckpoint<TMetadata>(
        IModel<TInput, TOutput, TMetadata> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        Dictionary<string, object>? metadata = null)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        lock (SyncLock)
        {
            var checkpoint = new Checkpoint<T, TInput, TOutput>(model, optimizer, epoch, step, metrics, metadata);
            var checkpointPath = GetCheckpointFilePath(checkpoint.CheckpointId);

            // Serialize checkpoint using Newtonsoft.Json
            var json = SerializeToJson(checkpoint);
            File.WriteAllText(checkpointPath, json);

            checkpoint.FilePath = checkpointPath;

            // Store metadata
            var checkpointMetadata = new CheckpointMetadata<T>
            {
                CheckpointId = checkpoint.CheckpointId,
                Epoch = epoch,
                Step = step,
                Metrics = new Dictionary<string, T>(metrics),
                CreatedAt = checkpoint.CreatedAt,
                FilePath = checkpointPath,
                FileSizeBytes = new FileInfo(checkpointPath).Length
            };

            _checkpoints[checkpoint.CheckpointId] = checkpointMetadata;

            // Auto-cleanup if configured
            if (AutoConfig != null && AutoConfig.KeepLast > 0)
            {
                CleanupOldCheckpoints(AutoConfig.KeepLast);
            }

            return checkpoint.CheckpointId;
        }
    }

    /// <summary>
    /// Loads a checkpoint and restores the training state.
    /// </summary>
    public override Checkpoint<T, TInput, TOutput> LoadCheckpoint(string checkpointId)
    {
        lock (SyncLock)
        {
            if (!_checkpoints.TryGetValue(checkpointId, out var metadata))
                throw new ArgumentException($"Checkpoint with ID '{checkpointId}' not found.", nameof(checkpointId));

            if (metadata.FilePath == null || !File.Exists(metadata.FilePath))
                throw new FileNotFoundException($"Checkpoint file not found: {metadata.FilePath}");

            // Validate the path is within the checkpoint directory
            ValidatePathWithinDirectory(metadata.FilePath, CheckpointDirectory);

            var json = File.ReadAllText(metadata.FilePath);
            var checkpoint = DeserializeFromJson<Checkpoint<T, TInput, TOutput>>(json);

            if (checkpoint == null)
                throw new InvalidOperationException($"Failed to deserialize checkpoint: {checkpointId}");

            return checkpoint;
        }
    }

    /// <summary>
    /// Loads the most recent checkpoint.
    /// </summary>
    public override Checkpoint<T, TInput, TOutput>? LoadLatestCheckpoint()
    {
        lock (SyncLock)
        {
            var latest = _checkpoints.Values
                .OrderByDescending(c => c.CreatedAt)
                .FirstOrDefault();

            return latest != null ? LoadCheckpoint(latest.CheckpointId) : null;
        }
    }

    /// <summary>
    /// Loads the checkpoint with the best metric value.
    /// </summary>
    public override Checkpoint<T, TInput, TOutput>? LoadBestCheckpoint(string metricName, MetricOptimizationDirection direction)
    {
        lock (SyncLock)
        {
            var checkpointsWithMetric = _checkpoints.Values
                .Where(c => c.Metrics.ContainsKey(metricName))
                .ToList();

            if (checkpointsWithMetric.Count == 0)
                return null;

            var best = direction == MetricOptimizationDirection.Minimize
                ? checkpointsWithMetric.OrderBy(c => c.Metrics[metricName]).First()
                : checkpointsWithMetric.OrderByDescending(c => c.Metrics[metricName]).First();

            return LoadCheckpoint(best.CheckpointId);
        }
    }

    /// <summary>
    /// Lists all available checkpoints.
    /// </summary>
    public override List<CheckpointMetadata<T>> ListCheckpoints(string? sortBy = null, bool descending = true)
    {
        lock (SyncLock)
        {
            IEnumerable<CheckpointMetadata<T>> checkpoints = _checkpoints.Values;

            if (sortBy != null && !string.IsNullOrWhiteSpace(sortBy))
            {
                var sortKey = sortBy;
                if (sortKey.Equals("created", StringComparison.OrdinalIgnoreCase))
                {
                    checkpoints = descending
                        ? checkpoints.OrderByDescending(c => c.CreatedAt)
                        : checkpoints.OrderBy(c => c.CreatedAt);
                }
                else if (sortKey.Equals("step", StringComparison.OrdinalIgnoreCase))
                {
                    checkpoints = descending
                        ? checkpoints.OrderByDescending(c => c.Step)
                        : checkpoints.OrderBy(c => c.Step);
                }
                else if (_checkpoints.Values.Any(c => c.Metrics.ContainsKey(sortKey)))
                {
                    checkpoints = descending
                        ? checkpoints.OrderByDescending(c => { c.Metrics.TryGetValue(sortKey, out var value); return value; })
                        : checkpoints.OrderBy(c => { c.Metrics.TryGetValue(sortKey, out var value); return value; });
                }
            }

            return checkpoints.ToList();
        }
    }

    /// <summary>
    /// Deletes a specific checkpoint.
    /// </summary>
    public override void DeleteCheckpoint(string checkpointId)
    {
        lock (SyncLock)
        {
            if (!_checkpoints.TryGetValue(checkpointId, out var metadata))
                return;

            // Delete file
            if (metadata.FilePath != null && File.Exists(metadata.FilePath))
            {
                File.Delete(metadata.FilePath);
            }

            // Remove from tracking
            _checkpoints.Remove(checkpointId);
        }
    }

    /// <summary>
    /// Deletes old checkpoints, keeping only the most recent ones.
    /// </summary>
    public override int CleanupOldCheckpoints(int keepLast = 5)
    {
        lock (SyncLock)
        {
            var checkpointsToDelete = _checkpoints.Values
                .OrderByDescending(c => c.CreatedAt)
                .Skip(keepLast)
                .ToList();

            foreach (var checkpoint in checkpointsToDelete)
            {
                DeleteCheckpoint(checkpoint.CheckpointId);
            }

            return checkpointsToDelete.Count;
        }
    }

    /// <summary>
    /// Deletes checkpoints except the best N according to a metric.
    /// </summary>
    public override int CleanupKeepBest(string metricName, int keepBest = 3, MetricOptimizationDirection direction = MetricOptimizationDirection.Minimize)
    {
        lock (SyncLock)
        {
            var checkpointsWithMetric = _checkpoints.Values
                .Where(c => c.Metrics.ContainsKey(metricName))
                .ToList();

            var checkpointsToKeep = direction == MetricOptimizationDirection.Minimize
                ? checkpointsWithMetric.OrderBy(c => c.Metrics[metricName]).Take(keepBest).ToList()
                : checkpointsWithMetric.OrderByDescending(c => c.Metrics[metricName]).Take(keepBest).ToList();

            var keepIds = checkpointsToKeep.Select(c => c.CheckpointId).ToHashSet();
            // Only delete from checkpoints that have the metric (preserve those without it)
            var checkpointsToDelete = checkpointsWithMetric
                .Where(c => !keepIds.Contains(c.CheckpointId))
                .ToList();

            foreach (var checkpoint in checkpointsToDelete)
            {
                DeleteCheckpoint(checkpoint.CheckpointId);
            }

            return checkpointsToDelete.Count;
        }
    }

    /// <summary>
    /// Attempts to save a checkpoint automatically based on configured auto-checkpoint settings.
    /// This method is called internally by training facades - users don't need to call it directly.
    /// </summary>
    public override string? TryAutoSaveCheckpoint<TMetadata>(
        IModel<TInput, TOutput, TMetadata> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        double? metricValue = null,
        bool shouldMinimize = true,
        Dictionary<string, object>? metadata = null)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        // Check if we should save based on configured criteria (SaveFrequency, SaveOnImprovement)
        if (!ShouldAutoSaveCheckpoint(step, metricValue, shouldMinimize))
        {
            return null;
        }

        // Save the checkpoint
        var checkpointId = SaveCheckpoint(model, optimizer, epoch, step, metrics, metadata);

        // Update auto-save state for future checks
        UpdateAutoSaveState(step, metricValue, shouldMinimize);

        return checkpointId;
    }

    #region Private Helper Methods

    private void LoadExistingCheckpoints()
    {
        if (!Directory.Exists(CheckpointDirectory))
            return;

        var checkpointFiles = Directory.GetFiles(CheckpointDirectory, "checkpoint_*.json");

        foreach (var file in checkpointFiles)
        {
            try
            {
                // Validate the path is within the checkpoint directory
                ValidatePathWithinDirectory(file, CheckpointDirectory);

                var json = File.ReadAllText(file);
                var checkpoint = DeserializeFromJson<Checkpoint<T, TInput, TOutput>>(json);

                if (checkpoint != null)
                {
                    var metadata = new CheckpointMetadata<T>
                    {
                        CheckpointId = checkpoint.CheckpointId,
                        Epoch = checkpoint.Epoch,
                        Step = checkpoint.Step,
                        Metrics = checkpoint.Metrics,
                        CreatedAt = checkpoint.CreatedAt,
                        FilePath = file,
                        FileSizeBytes = new FileInfo(file).Length
                    };

                    _checkpoints[checkpoint.CheckpointId] = metadata;
                }
            }
            catch (IOException ex)
            {
                Console.WriteLine($"[CheckpointManager] Failed to read checkpoint file '{file}': {ex.Message}");
            }
            catch (JsonException ex)
            {
                Console.WriteLine($"[CheckpointManager] Failed to deserialize checkpoint file '{file}': {ex.Message}");
            }
        }
    }

    #endregion
}
