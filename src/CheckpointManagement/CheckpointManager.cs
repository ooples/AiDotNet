using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Enums;
using System.Text.Json;

namespace AiDotNet.CheckpointManagement;

/// <summary>
/// Implementation of checkpoint management for saving and restoring training state.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This manages checkpoints (save points) of your model training,
/// allowing you to save progress, resume training, and keep track of the best models.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class CheckpointManager<T, TInput, TOutput> : ICheckpointManager<T, TInput, TOutput>
{
    private readonly string _checkpointDirectory;
    private readonly Dictionary<string, CheckpointMetadata<T>> _checkpoints;
    private readonly object _lock = new();
    private AutoCheckpointConfig? _autoConfig;

    /// <summary>
    /// Initializes a new instance of the CheckpointManager class.
    /// </summary>
    /// <param name="checkpointDirectory">Directory to store checkpoints. Defaults to "./checkpoints".</param>
    public CheckpointManager(string? checkpointDirectory = null)
    {
        _checkpointDirectory = checkpointDirectory ?? Path.Combine(Directory.GetCurrentDirectory(), "checkpoints");
        _checkpoints = new Dictionary<string, CheckpointMetadata<T>>();

        // Create checkpoint directory if it doesn't exist
        if (!Directory.Exists(_checkpointDirectory))
        {
            Directory.CreateDirectory(_checkpointDirectory);
        }

        // Load existing checkpoints
        LoadExistingCheckpoints();
    }

    /// <summary>
    /// Saves a checkpoint of the current training state.
    /// </summary>
    public string SaveCheckpoint(
        IModel<TInput, TOutput, TMetadata> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        Dictionary<string, object>? metadata = null) where TMetadata : class
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        lock (_lock)
        {
            var checkpoint = new Checkpoint<T, TInput, TOutput>(model, optimizer, epoch, step, metrics, metadata);
            var checkpointPath = Path.Combine(_checkpointDirectory, $"checkpoint_{checkpoint.CheckpointId}.json");

            // Serialize checkpoint
            var json = JsonSerializer.Serialize(checkpoint, new JsonSerializerOptions { WriteIndented = true });
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
            if (_autoConfig != null && _autoConfig.KeepLast > 0)
            {
                CleanupOldCheckpoints(_autoConfig.KeepLast);
            }

            return checkpoint.CheckpointId;
        }
    }

    /// <summary>
    /// Loads a checkpoint and restores the training state.
    /// </summary>
    public Checkpoint<T, TInput, TOutput> LoadCheckpoint(string checkpointId)
    {
        lock (_lock)
        {
            if (!_checkpoints.TryGetValue(checkpointId, out var metadata))
                throw new ArgumentException($"Checkpoint with ID '{checkpointId}' not found.", nameof(checkpointId));

            if (metadata.FilePath == null || !File.Exists(metadata.FilePath))
                throw new FileNotFoundException($"Checkpoint file not found: {metadata.FilePath}");

            var json = File.ReadAllText(metadata.FilePath);
            var checkpoint = JsonSerializer.Deserialize<Checkpoint<T, TInput, TOutput>>(json);

            if (checkpoint == null)
                throw new InvalidOperationException($"Failed to deserialize checkpoint: {checkpointId}");

            return checkpoint;
        }
    }

    /// <summary>
    /// Loads the most recent checkpoint.
    /// </summary>
    public Checkpoint<T, TInput, TOutput>? LoadLatestCheckpoint()
    {
        lock (_lock)
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
    public Checkpoint<T, TInput, TOutput>? LoadBestCheckpoint(string metricName, OptimizationMode mode)
    {
        lock (_lock)
        {
            var checkpointsWithMetric = _checkpoints.Values
                .Where(c => c.Metrics.ContainsKey(metricName))
                .ToList();

            if (checkpointsWithMetric.Count == 0)
                return null;

            var best = mode == OptimizationMode.ParametersOnly // Using as minimize
                ? checkpointsWithMetric.OrderBy(c => c.Metrics[metricName]).First()
                : checkpointsWithMetric.OrderByDescending(c => c.Metrics[metricName]).First();

            return LoadCheckpoint(best.CheckpointId);
        }
    }

    /// <summary>
    /// Lists all available checkpoints.
    /// </summary>
    public List<CheckpointMetadata<T>> ListCheckpoints(string? sortBy = null, bool descending = true)
    {
        lock (_lock)
        {
            IEnumerable<CheckpointMetadata<T>> checkpoints = _checkpoints.Values;

            if (!string.IsNullOrWhiteSpace(sortBy))
            {
                if (sortBy.Equals("created", StringComparison.OrdinalIgnoreCase))
                {
                    checkpoints = descending
                        ? checkpoints.OrderByDescending(c => c.CreatedAt)
                        : checkpoints.OrderBy(c => c.CreatedAt);
                }
                else if (sortBy.Equals("step", StringComparison.OrdinalIgnoreCase))
                {
                    checkpoints = descending
                        ? checkpoints.OrderByDescending(c => c.Step)
                        : checkpoints.OrderBy(c => c.Step);
                }
                else if (_checkpoints.Values.Any(c => c.Metrics.ContainsKey(sortBy)))
                {
                    checkpoints = descending
                        ? checkpoints.OrderByDescending(c => c.Metrics.ContainsKey(sortBy) ? c.Metrics[sortBy] : default)
                        : checkpoints.OrderBy(c => c.Metrics.ContainsKey(sortBy) ? c.Metrics[sortBy] : default);
                }
            }

            return checkpoints.ToList();
        }
    }

    /// <summary>
    /// Deletes a specific checkpoint.
    /// </summary>
    public void DeleteCheckpoint(string checkpointId)
    {
        lock (_lock)
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
    public int CleanupOldCheckpoints(int keepLast = 5)
    {
        lock (_lock)
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
    public int CleanupKeepBest(string metricName, int keepBest = 3, OptimizationMode mode = OptimizationMode.Both)
    {
        lock (_lock)
        {
            var checkpointsWithMetric = _checkpoints.Values
                .Where(c => c.Metrics.ContainsKey(metricName))
                .ToList();

            var checkpointsToKeep = mode == OptimizationMode.ParametersOnly // Using as minimize
                ? checkpointsWithMetric.OrderBy(c => c.Metrics[metricName]).Take(keepBest).ToList()
                : checkpointsWithMetric.OrderByDescending(c => c.Metrics[metricName]).Take(keepBest).ToList();

            var keepIds = checkpointsToKeep.Select(c => c.CheckpointId).ToHashSet();
            var checkpointsToDelete = _checkpoints.Values
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
    /// Gets the storage path for checkpoints.
    /// </summary>
    public string GetCheckpointDirectory()
    {
        return _checkpointDirectory;
    }

    /// <summary>
    /// Sets up automatic checkpointing during training.
    /// </summary>
    public void ConfigureAutoCheckpointing(
        int saveFrequency,
        int keepLast = 5,
        bool saveOnImprovement = true,
        string? metricName = null)
    {
        _autoConfig = new AutoCheckpointConfig
        {
            SaveFrequency = saveFrequency,
            KeepLast = keepLast,
            SaveOnImprovement = saveOnImprovement,
            MetricName = metricName
        };
    }

    #region Private Helper Methods

    private void LoadExistingCheckpoints()
    {
        if (!Directory.Exists(_checkpointDirectory))
            return;

        var checkpointFiles = Directory.GetFiles(_checkpointDirectory, "checkpoint_*.json");

        foreach (var file in checkpointFiles)
        {
            try
            {
                var json = File.ReadAllText(file);
                var checkpoint = JsonSerializer.Deserialize<Checkpoint<T, TInput, TOutput>>(json);

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
            catch
            {
                // Skip corrupted checkpoint files
            }
        }
    }

    #endregion

    private class AutoCheckpointConfig
    {
        public int SaveFrequency { get; set; }
        public int KeepLast { get; set; }
        public bool SaveOnImprovement { get; set; }
        public string? MetricName { get; set; }
    }
}
