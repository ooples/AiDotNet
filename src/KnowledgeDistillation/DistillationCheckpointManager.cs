using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation.Strategies;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Manages checkpointing during knowledge distillation training.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class handles saving and loading model states during
/// distillation training. It's like the "save game" manager in a video game - it decides
/// when to save, what to save, and how to load progress later.</para>
///
/// <para><b>Key Features:</b>
/// - Automatic checkpointing at specified intervals
/// - Keep only best N checkpoints based on validation metrics
/// - Save/restore curriculum learning progress
/// - Support for multi-stage distillation (student → teacher)
/// - Resume interrupted training</para>
///
/// <para><b>Example Usage:</b>
/// <code>
/// var config = new DistillationCheckpointConfig
/// {
///     CheckpointDirectory = "./checkpoints",
///     SaveEveryEpochs = 5,
///     KeepBestN = 3,
///     SaveStudent = true
/// };
///
/// var manager = new DistillationCheckpointManager&lt;double&gt;(config);
///
/// // During training
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     // ... training code ...
///
///     double validationLoss = EvaluateStudent();
///     manager.SaveCheckpointIfNeeded(
///         epoch: epoch,
///         student: studentModel,
///         metrics: new Dictionary&lt;string, double&gt; { { "validation_loss", validationLoss } }
///     );
/// }
///
/// // Load best checkpoint
/// manager.LoadBestCheckpoint(studentModel);
/// </code>
/// </para>
/// </remarks>
public class DistillationCheckpointManager<T>
{
    private readonly DistillationCheckpointConfig _config;
    private readonly List<CheckpointMetadata> _savedCheckpoints;
    private int _batchCounter;

    /// <summary>
    /// Initializes a new instance of the DistillationCheckpointManager class.
    /// </summary>
    /// <param name="config">Configuration for checkpoint management.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> This constructor requires explicit configuration.
    /// All parameters have recommended defaults in <see cref="DistillationCheckpointConfig"/>.</para>
    /// </remarks>
    public DistillationCheckpointManager(DistillationCheckpointConfig config)
    {
        Guard.NotNull(config);
        _config = config;
        _savedCheckpoints = new List<CheckpointMetadata>();
        _batchCounter = 0;

        // Create checkpoint directory if it doesn't exist
        if (!Directory.Exists(_config.CheckpointDirectory))
        {
            Directory.CreateDirectory(_config.CheckpointDirectory);
        }

        // Load existing checkpoint metadata
        LoadExistingCheckpointMetadata();
    }

    /// <summary>
    /// Determines if a checkpoint should be saved based on current progress (internal logic).
    /// </summary>
    private bool ShouldSaveCheckpoint(int? epoch = null, int? batch = null)
    {
        if (epoch.HasValue && _config.SaveEveryEpochs > 0)
        {
            return (epoch.Value + 1) % _config.SaveEveryEpochs == 0;
        }

        if (batch.HasValue && _config.SaveEveryBatches > 0)
        {
            _batchCounter++;
            return _batchCounter % _config.SaveEveryBatches == 0;
        }

        return false;
    }

    /// <summary>
    /// Saves a checkpoint if conditions are met.
    /// </summary>
    /// <param name="epoch">Current epoch number.</param>
    /// <param name="student">Student model to checkpoint (if SaveStudent = true).</param>
    /// <param name="teacher">Teacher model to checkpoint (if SaveTeacher = true).</param>
    /// <param name="strategy">Distillation strategy to checkpoint.</param>
    /// <param name="metrics">Training/validation metrics for this checkpoint.</param>
    /// <param name="batch">Optional batch number.</param>
    /// <param name="force">Force save regardless of schedule.</param>
    /// <returns>True if checkpoint was saved.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this method periodically during training.
    /// It will automatically decide whether to save based on your configuration.</para>
    /// </remarks>
    public bool SaveCheckpointIfNeeded(
        int epoch,
        ICheckpointableModel? student = null,
        ICheckpointableModel? teacher = null,
        object? strategy = null,
        Dictionary<string, double>? metrics = null,
        int? batch = null,
        bool force = false)
    {
        if (!force && !ShouldSaveCheckpoint(epoch, batch))
        {
            return false;
        }

        string checkpointPath = GenerateCheckpointPath(epoch, batch);
        var metadata = new CheckpointMetadata
        {
            FilePath = checkpointPath,
            Epoch = epoch,
            Batch = batch,
            Timestamp = DateTime.UtcNow,
            Metrics = metrics ?? new Dictionary<string, double>()
        };

        SaveCheckpoint(checkpointPath, student, teacher, strategy, metadata);

        _savedCheckpoints.Add(metadata);
        PruneOldCheckpoints();
        SaveCheckpointMetadata();

        return true;
    }

    /// <summary>
    /// Saves a checkpoint to the specified path.
    /// </summary>
    private void SaveCheckpoint(
        string basePath,
        ICheckpointableModel? student,
        ICheckpointableModel? teacher,
        object? strategy,
        CheckpointMetadata metadata)
    {
        // Persist batch counter for resume continuity
        metadata.BatchCounter = _batchCounter;

        // Save student
        if (_config.SaveStudent && student != null)
        {
            string studentPath = basePath + ".student.bin";
            using var stream = File.Create(studentPath);
            student.SaveState(stream);
            metadata.StudentCheckpointPath = studentPath;
        }

        // Save teacher
        if (_config.SaveTeacher && teacher != null)
        {
            string teacherPath = basePath + ".teacher.bin";
            using var stream = File.Create(teacherPath);
            teacher.SaveState(stream);
            metadata.TeacherCheckpointPath = teacherPath;
        }

        // Save strategy state (if curriculum)
        if (_config.SaveCurriculumState && strategy != null)
        {
            string strategyPath = basePath + ".strategy.json";
            SaveStrategyState(strategyPath, strategy);
            metadata.StrategyCheckpointPath = strategyPath;
        }

        // Save metadata
        if (_config.SaveMetadata)
        {
            string metadataPath = basePath + ".metadata.json";
            var json = JsonConvert.SerializeObject(metadata, Formatting.Indented);
            File.WriteAllText(metadataPath, json);
        }
    }

    /// <summary>
    /// Loads the best checkpoint based on the configured metric.
    /// </summary>
    /// <param name="student">Student model to load into.</param>
    /// <param name="teacher">Optional teacher model to load into.</param>
    /// <returns>Metadata of the loaded checkpoint, or null if no checkpoints exist.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this after training to load the checkpoint
    /// with the best validation performance.</para>
    /// </remarks>
    public CheckpointMetadata? LoadBestCheckpoint(
        ICheckpointableModel? student = null,
        ICheckpointableModel? teacher = null)
    {
        var bestCheckpoint = GetBestCheckpoint();
        if (bestCheckpoint == null)
        {
            return null;
        }

        LoadCheckpoint(bestCheckpoint, student, teacher);
        return bestCheckpoint;
    }

    /// <summary>
    /// Loads a specific checkpoint.
    /// </summary>
    /// <param name="metadata">Metadata of the checkpoint to load.</param>
    /// <param name="student">Student model to load into.</param>
    /// <param name="teacher">Optional teacher model to load into.</param>
    public void LoadCheckpoint(
        CheckpointMetadata metadata,
        ICheckpointableModel? student = null,
        ICheckpointableModel? teacher = null)
    {
        // Restore batch counter for resume continuity
        _batchCounter = metadata.BatchCounter;

        // Load student
        if (student != null && metadata.StudentCheckpointPath != null)
        {
            using var stream = File.OpenRead(metadata.StudentCheckpointPath);
            student.LoadState(stream);
        }

        // Load teacher
        if (teacher != null && metadata.TeacherCheckpointPath != null)
        {
            using var stream = File.OpenRead(metadata.TeacherCheckpointPath);
            teacher.LoadState(stream);
        }

        // Strategy state would be loaded separately based on type
    }

    /// <summary>
    /// Gets the checkpoint with the best metric value.
    /// </summary>
    /// <returns>Metadata of the best checkpoint, or null if no checkpoints exist.</returns>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Returns the checkpoint with the best validation metric
    /// based on the configuration (e.g., lowest validation loss or highest accuracy).</para>
    /// </remarks>
    public CheckpointMetadata? GetBestCheckpoint()
    {
        if (_savedCheckpoints.Count == 0)
        {
            return null;
        }

        var checkpointsWithMetric = _savedCheckpoints
            .Where(c => c.Metrics.ContainsKey(_config.BestMetric))
            .ToList();

        if (checkpointsWithMetric.Count == 0)
        {
            return _savedCheckpoints.Last(); // Return most recent if no metrics
        }

        return _config.LowerIsBetter
            ? checkpointsWithMetric.OrderBy(c => c.Metrics[_config.BestMetric]).First()
            : checkpointsWithMetric.OrderByDescending(c => c.Metrics[_config.BestMetric]).First();
    }

    /// <summary>
    /// Gets the most recently saved checkpoint.
    /// </summary>
    /// <returns>Metadata of the most recent checkpoint, or null if no checkpoints exist.</returns>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Useful for resuming interrupted training from the last saved state.</para>
    /// </remarks>
    public CheckpointMetadata? GetMostRecentCheckpoint()
    {
        return _savedCheckpoints.OrderByDescending(c => c.Epoch).ThenByDescending(c => c.Batch).FirstOrDefault();
    }

    /// <summary>
    /// Gets a checkpoint for a specific epoch.
    /// </summary>
    /// <param name="epoch">The epoch number to find.</param>
    /// <returns>Metadata of the checkpoint at the specified epoch, or null if not found.</returns>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Returns the checkpoint saved at a specific epoch number.</para>
    /// </remarks>
    public CheckpointMetadata? GetCheckpointByEpoch(int epoch)
    {
        return _savedCheckpoints.FirstOrDefault(c => c.Epoch == epoch);
    }

    /// <summary>
    /// Gets all saved checkpoint metadata as a readonly collection.
    /// </summary>
    /// <returns>Readonly list of all checkpoint metadata.</returns>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Provides read-only access to all saved checkpoints for custom queries.</para>
    /// </remarks>
    public IReadOnlyList<CheckpointMetadata> GetAllCheckpoints()
    {
        return _savedCheckpoints.AsReadOnly();
    }

    /// <summary>
    /// Deletes old checkpoints, keeping only the best N.
    /// </summary>
    private void PruneOldCheckpoints()
    {
        if (_config.KeepBestN <= 0 || _savedCheckpoints.Count <= _config.KeepBestN)
        {
            return;
        }

        // Sort by metric (best first)
        var sorted = _savedCheckpoints
            .Where(c => c.Metrics.ContainsKey(_config.BestMetric))
            .OrderBy(c => _config.LowerIsBetter ? c.Metrics[_config.BestMetric] : -c.Metrics[_config.BestMetric])
            .ToList();

        // Get checkpoints without metrics
        var withoutMetric = _savedCheckpoints
            .Where(c => !c.Metrics.ContainsKey(_config.BestMetric))
            .ToList();

        // Include both sorted and without-metric checkpoints in pruning to match KeepBestN exactly
        // Preserve best N sorted checkpoints, then fill remaining slots with without-metric checkpoints
        var toKeep = sorted.Take(_config.KeepBestN)
            .Concat(withoutMetric.Take(Math.Max(0, _config.KeepBestN - sorted.Count)))
            .ToList();

        var toDelete = _savedCheckpoints.Where(c => !toKeep.Contains(c)).ToList();

        foreach (var checkpoint in toDelete)
        {
            DeleteCheckpointFiles(checkpoint);
            _savedCheckpoints.Remove(checkpoint);
        }
    }

    /// <summary>
    /// Deletes all files associated with a checkpoint.
    /// </summary>
    private void DeleteCheckpointFiles(CheckpointMetadata checkpoint)
    {
        if (checkpoint.StudentCheckpointPath != null && File.Exists(checkpoint.StudentCheckpointPath))
        {
            File.Delete(checkpoint.StudentCheckpointPath);
        }

        if (checkpoint.TeacherCheckpointPath != null && File.Exists(checkpoint.TeacherCheckpointPath))
        {
            File.Delete(checkpoint.TeacherCheckpointPath);
        }

        if (checkpoint.StrategyCheckpointPath != null && File.Exists(checkpoint.StrategyCheckpointPath))
        {
            File.Delete(checkpoint.StrategyCheckpointPath);
        }

        string metadataPath = checkpoint.FilePath + ".metadata.json";
        if (File.Exists(metadataPath))
        {
            File.Delete(metadataPath);
        }
    }

    private string GenerateCheckpointPath(int epoch, int? batch)
    {
        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        string filename = batch.HasValue
            ? $"{_config.CheckpointPrefix}_epoch{epoch}_batch{batch.Value}_{timestamp}"
            : $"{_config.CheckpointPrefix}_epoch{epoch}_{timestamp}";

        return Path.Combine(_config.CheckpointDirectory, filename);
    }

    private void SaveStrategyState(string path, object strategy)
    {
        // Serialize curriculum progress if applicable
        if (strategy is ICurriculumDistillationStrategy<T> curriculum)
        {
            var state = new
            {
                Type = strategy.GetType().FullName,
                CurriculumProgress = curriculum.CurriculumProgress,
                TotalSteps = curriculum.TotalSteps
            };

            var json = JsonConvert.SerializeObject(state, Formatting.Indented);
            File.WriteAllText(path, json);
        }
    }

    private void LoadExistingCheckpointMetadata()
    {
        string metadataIndexPath = Path.Combine(_config.CheckpointDirectory, "checkpoint_index.json");
        if (File.Exists(metadataIndexPath))
        {
            var json = File.ReadAllText(metadataIndexPath);
            var checkpoints = JsonConvert.DeserializeObject<List<CheckpointMetadata>>(json);
            if (checkpoints != null)
            {
                _savedCheckpoints.AddRange(checkpoints);
            }
        }
    }

    private void SaveCheckpointMetadata()
    {
        string metadataIndexPath = Path.Combine(_config.CheckpointDirectory, "checkpoint_index.json");
        var json = JsonConvert.SerializeObject(_savedCheckpoints, Formatting.Indented);
        File.WriteAllText(metadataIndexPath, json);
    }
}

/// <summary>
/// Metadata about a saved checkpoint.
/// </summary>
public class CheckpointMetadata
{
    /// <summary>
    /// Base file path for this checkpoint.
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Epoch number when checkpoint was saved.
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// Batch number when checkpoint was saved (if applicable).
    /// </summary>
    public int? Batch { get; set; }

    /// <summary>
    /// Batch counter value when checkpoint was saved (for resume continuity).
    /// </summary>
    public int BatchCounter { get; set; }

    /// <summary>
    /// Timestamp when checkpoint was created.
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Training/validation metrics at checkpoint time.
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>
    /// Path to student model checkpoint file.
    /// </summary>
    public string? StudentCheckpointPath { get; set; }

    /// <summary>
    /// Path to teacher model checkpoint file.
    /// </summary>
    public string? TeacherCheckpointPath { get; set; }

    /// <summary>
    /// Path to strategy state file.
    /// </summary>
    public string? StrategyCheckpointPath { get; set; }
}
