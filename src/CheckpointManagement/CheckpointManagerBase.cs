using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Enums;
using Newtonsoft.Json;

namespace AiDotNet.CheckpointManagement;

/// <summary>
/// Base class for checkpoint management implementations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This abstract base class provides common functionality for checkpoint
/// management systems. It handles storage path management, security validation, and JSON serialization
/// while leaving the specific storage implementation to derived classes.
///
/// Key features:
/// - Path security validation to prevent traversal attacks
/// - Consistent JSON serialization settings
/// - Thread-safe checkpoint tracking
/// - Auto-checkpointing configuration support
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public abstract class CheckpointManagerBase<T, TInput, TOutput> : ICheckpointManager<T, TInput, TOutput>
{
    /// <summary>
    /// The directory where checkpoints are stored.
    /// </summary>
    protected readonly string CheckpointDirectory;

    /// <summary>
    /// Lock object for thread-safe operations.
    /// </summary>
    protected readonly object SyncLock = new();

    /// <summary>
    /// JSON serialization settings for consistent serialization.
    /// Uses a custom SerializationBinder for security against deserialization attacks.
    /// </summary>
    protected static readonly JsonSerializerSettings JsonSettings = new()
    {
        Formatting = Formatting.Indented,
        TypeNameHandling = TypeNameHandling.Auto,
        SerializationBinder = new SafeTypeSerializationBinder()
    };

    /// <summary>
    /// Custom serialization binder that restricts deserialization to safe types only.
    /// Prevents remote code execution vulnerabilities from malicious JSON payloads.
    /// </summary>
    private sealed class SafeTypeSerializationBinder : Newtonsoft.Json.Serialization.ISerializationBinder
    {
        private static readonly HashSet<string> AllowedNamespacePrefixes = new(StringComparer.OrdinalIgnoreCase)
        {
            "AiDotNet.",
            "System.Collections.Generic.",
            "System.Collections.Concurrent.",
            "System.Collections.ObjectModel."
            // Note: Removed broad "System." prefix to prevent deserialization of dangerous types
            // like System.Diagnostics.Process. Use AllowedSystemTypes for specific safe types.
        };

        private static readonly HashSet<string> AllowedSystemTypes = new(StringComparer.OrdinalIgnoreCase)
        {
            "System.String",
            "System.Int32",
            "System.Int64",
            "System.Double",
            "System.Single",
            "System.Decimal",
            "System.Boolean",
            "System.DateTime",
            "System.DateTimeOffset",
            "System.Guid",
            "System.TimeSpan",
            "System.Object",
            "System.Object[]"
        };

        public void BindToName(Type serializedType, out string? assemblyName, out string? typeName)
        {
            assemblyName = null;
            typeName = serializedType.AssemblyQualifiedName;
        }

        public Type BindToType(string? assemblyName, string typeName)
        {
            // Validate the type name against allowed patterns
            if (!IsTypeAllowed(typeName))
            {
                throw new JsonSerializationException(
                    $"Deserialization of type '{typeName}' is not allowed for security reasons.");
            }

            // Try to resolve the type from the type name
            var type = Type.GetType(typeName);
            if (type != null)
                return type;

            // If Type.GetType fails, search through loaded assemblies
            // This is needed for types in other assemblies (like test assemblies)
            var baseTypeName = ExtractBaseTypeName(typeName);
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                type = assembly.GetType(baseTypeName);
                if (type != null)
                    return type;
            }

            throw new JsonSerializationException($"Could not resolve type '{typeName}'.");
        }

        private static bool IsTypeAllowed(string typeName)
        {
            if (string.IsNullOrWhiteSpace(typeName))
                return false;

            // Extract the base type name (before generic arguments and assembly info)
            var baseTypeName = ExtractBaseTypeName(typeName);

            // Check if it's a basic allowed System type
            if (AllowedSystemTypes.Contains(baseTypeName))
                return true;

            // Check if it starts with an allowed namespace prefix
            foreach (var prefix in AllowedNamespacePrefixes)
            {
                if (baseTypeName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    return true;
            }

            return false;
        }

        private static string ExtractBaseTypeName(string typeName)
        {
            // Handle generic types like "System.Collections.Generic.Dictionary`2[[...]]"
            var genericIndex = typeName.IndexOf('`');
            if (genericIndex > 0)
            {
                return typeName.Substring(0, genericIndex);
            }

            // Handle assembly-qualified names like "Namespace.Type, Assembly"
            var commaIndex = typeName.IndexOf(',');
            if (commaIndex > 0)
            {
                return typeName.Substring(0, commaIndex).Trim();
            }

            return typeName;
        }
    }

    /// <summary>
    /// Configuration for auto-checkpointing.
    /// </summary>
    protected AutoCheckpointConfiguration? AutoConfig;

    /// <summary>
    /// The last step at which an auto-checkpoint was saved.
    /// </summary>
    protected int LastAutoSaveStep;

    /// <summary>
    /// The best metric value seen for improvement-based checkpointing.
    /// </summary>
    protected double? BestMetricValue;

    /// <summary>
    /// Initializes a new instance of the CheckpointManagerBase class.
    /// </summary>
    /// <param name="checkpointDirectory">Directory to store checkpoints.</param>
    /// <param name="baseDirectory">Base directory for path validation. Defaults to current directory.</param>
    /// <remarks>
    /// When a custom checkpointDirectory is provided without a baseDirectory, the checkpoint directory
    /// itself becomes the base for path validation. This allows users to store checkpoints in
    /// any location they choose (like temp directories for tests) while still preventing
    /// path traversal attacks within that chosen directory.
    /// </remarks>
    protected CheckpointManagerBase(string? checkpointDirectory = null, string? baseDirectory = null)
    {
        if (checkpointDirectory != null)
        {
            // User provided a custom checkpoint directory
            // Use the provided base directory, or default to the checkpoint directory itself
            var fullCheckpointPath = Path.GetFullPath(checkpointDirectory);
            var baseDir = baseDirectory != null ? Path.GetFullPath(baseDirectory) : fullCheckpointPath;
            CheckpointDirectory = GetSanitizedPath(fullCheckpointPath, baseDir);
        }
        else
        {
            // Using default checkpoint directory - validate against base
            var baseDir = baseDirectory ?? Directory.GetCurrentDirectory();
            var defaultStorage = Path.Combine(baseDir, "checkpoints");
            CheckpointDirectory = GetSanitizedPath(defaultStorage, baseDir);
        }

        EnsureCheckpointDirectoryExists();
    }

    /// <summary>
    /// Saves a checkpoint of the current training state.
    /// </summary>
    public abstract string SaveCheckpoint<TMetadata>(
        IModel<TInput, TOutput, TMetadata> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        Dictionary<string, object>? metadata = null) where TMetadata : class;

    /// <summary>
    /// Loads a checkpoint and restores the training state.
    /// </summary>
    public abstract Checkpoint<T, TInput, TOutput> LoadCheckpoint(string checkpointId);

    /// <summary>
    /// Loads the most recent checkpoint.
    /// </summary>
    public abstract Checkpoint<T, TInput, TOutput>? LoadLatestCheckpoint();

    /// <summary>
    /// Loads the checkpoint with the best metric value.
    /// </summary>
    public abstract Checkpoint<T, TInput, TOutput>? LoadBestCheckpoint(string metricName, MetricOptimizationDirection direction);

    /// <summary>
    /// Lists all available checkpoints.
    /// </summary>
    public abstract List<CheckpointMetadata<T>> ListCheckpoints(string? sortBy = null, bool descending = true);

    /// <summary>
    /// Deletes a specific checkpoint.
    /// </summary>
    public abstract void DeleteCheckpoint(string checkpointId);

    /// <summary>
    /// Deletes old checkpoints, keeping only the most recent ones.
    /// </summary>
    public abstract int CleanupOldCheckpoints(int keepLast = 5);

    /// <summary>
    /// Deletes checkpoints except the best N according to a metric.
    /// </summary>
    public abstract int CleanupKeepBest(string metricName, int keepBest = 3, MetricOptimizationDirection direction = MetricOptimizationDirection.Minimize);

    /// <summary>
    /// Attempts to save a checkpoint automatically based on configured auto-checkpoint settings.
    /// This method is called internally by training facades - users don't need to call it directly.
    /// The training loop calls this every step and it automatically decides when to save.
    /// </summary>
    /// <typeparam name="TMetadata">The type of model metadata.</typeparam>
    /// <param name="model">The model to checkpoint.</param>
    /// <param name="optimizer">The optimizer state to checkpoint.</param>
    /// <param name="epoch">The current epoch.</param>
    /// <param name="step">The current training step.</param>
    /// <param name="metrics">Training metrics to store with the checkpoint.</param>
    /// <param name="metricValue">Optional metric value for improvement-based checkpointing.</param>
    /// <param name="shouldMinimize">Whether the metric should be minimized (true) or maximized (false).</param>
    /// <param name="metadata">Optional additional metadata.</param>
    /// <returns>The checkpoint ID if saved, or null if no checkpoint was saved.</returns>
    public abstract string? TryAutoSaveCheckpoint<TMetadata>(
        IModel<TInput, TOutput, TMetadata> model,
        IOptimizer<T, TInput, TOutput> optimizer,
        int epoch,
        int step,
        Dictionary<string, T> metrics,
        double? metricValue = null,
        bool shouldMinimize = true,
        Dictionary<string, object>? metadata = null) where TMetadata : class;

    /// <summary>
    /// Gets the storage path for checkpoints.
    /// </summary>
    public virtual string GetCheckpointDirectory()
    {
        return CheckpointDirectory;
    }

    /// <summary>
    /// Sets up automatic checkpointing during training.
    /// </summary>
    public virtual void ConfigureAutoCheckpointing(
        int saveFrequency,
        int keepLast = 5,
        bool saveOnImprovement = true,
        string? metricName = null)
    {
        lock (SyncLock)
        {
            AutoConfig = new AutoCheckpointConfiguration
            {
                SaveFrequency = saveFrequency,
                KeepLast = keepLast,
                SaveOnImprovement = saveOnImprovement,
                MetricName = metricName
            };

            // Reset state
            LastAutoSaveStep = 0;
            BestMetricValue = null;
        }
    }

    /// <summary>
    /// Determines whether an automatic checkpoint should be saved based on current configuration.
    /// </summary>
    /// <param name="currentStep">The current training step.</param>
    /// <param name="metricValue">Optional metric value for improvement-based checkpointing.</param>
    /// <param name="shouldMinimize">Whether the metric should be minimized (true) or maximized (false).</param>
    /// <returns>True if a checkpoint should be saved.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Call this method during training to check if you should save:
    /// <code>
    /// for (int epoch = 0; epoch &lt; maxEpochs; epoch++)
    /// {
    ///     double loss = TrainEpoch();
    ///     if (checkpointManager.ShouldAutoSaveCheckpoint(epoch, loss, shouldMinimize: true))
    ///     {
    ///         checkpointManager.SaveCheckpoint(model);
    ///     }
    /// }
    /// </code>
    /// </remarks>
    public virtual bool ShouldAutoSaveCheckpoint(int currentStep, double? metricValue = null, bool shouldMinimize = true)
    {
        lock (SyncLock)
        {
            if (AutoConfig == null)
                return false;

            // Check frequency-based saving
            if (AutoConfig.SaveFrequency > 0)
            {
                if ((currentStep - LastAutoSaveStep) >= AutoConfig.SaveFrequency)
                {
                    return true;
                }
            }

            // Check improvement-based saving
            if (AutoConfig.SaveOnImprovement && metricValue.HasValue)
            {
                if (!BestMetricValue.HasValue)
                {
                    return true; // First value is always an improvement
                }

                bool isImprovement = shouldMinimize
                    ? metricValue.Value < BestMetricValue.Value
                    : metricValue.Value > BestMetricValue.Value;

                if (isImprovement)
                {
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>
    /// Updates the auto-save state after a checkpoint is saved.
    /// </summary>
    /// <param name="step">The step at which the checkpoint was saved.</param>
    /// <param name="metricValue">Optional metric value for improvement tracking.</param>
    /// <param name="shouldMinimize">Whether the metric should be minimized.</param>
    /// <remarks>
    /// Call this after saving a checkpoint to update internal tracking state.
    /// </remarks>
    public virtual void UpdateAutoSaveState(int step, double? metricValue = null, bool shouldMinimize = true)
    {
        lock (SyncLock)
        {
            LastAutoSaveStep = step;

            if (metricValue.HasValue)
            {
                if (!BestMetricValue.HasValue)
                {
                    BestMetricValue = metricValue.Value;
                }
                else
                {
                    bool isImprovement = shouldMinimize
                        ? metricValue.Value < BestMetricValue.Value
                        : metricValue.Value > BestMetricValue.Value;

                    if (isImprovement)
                    {
                        BestMetricValue = metricValue.Value;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Gets the current auto-checkpointing state.
    /// </summary>
    public virtual AutoCheckpointState GetAutoCheckpointState()
    {
        lock (SyncLock)
        {
            return new AutoCheckpointState(
                AutoConfig != null,
                AutoConfig?.SaveFrequency ?? 0,
                AutoConfig?.SaveOnImprovement ?? false,
                AutoConfig?.KeepLast ?? 0,
                AutoConfig?.MetricName,
                LastAutoSaveStep,
                BestMetricValue
            );
        }
    }

    #region Protected Helper Methods

    /// <summary>
    /// Ensures the checkpoint directory exists.
    /// </summary>
    protected virtual void EnsureCheckpointDirectoryExists()
    {
        if (!Directory.Exists(CheckpointDirectory))
        {
            Directory.CreateDirectory(CheckpointDirectory);
        }
    }

    /// <summary>
    /// Generates a checkpoint file path.
    /// </summary>
    /// <param name="checkpointId">The checkpoint identifier.</param>
    /// <returns>The full path to the checkpoint file.</returns>
    protected virtual string GetCheckpointFilePath(string checkpointId)
    {
        var fileName = GetSanitizedFileName($"checkpoint_{checkpointId}.json");
        var path = Path.Combine(CheckpointDirectory, fileName);
        ValidatePathWithinDirectory(path, CheckpointDirectory);
        return path;
    }

    /// <summary>
    /// Serializes an object to JSON.
    /// </summary>
    /// <param name="obj">The object to serialize.</param>
    /// <returns>The JSON string representation.</returns>
    protected virtual string SerializeToJson(object obj)
    {
        return JsonConvert.SerializeObject(obj, JsonSettings);
    }

    /// <summary>
    /// Deserializes a JSON string to an object.
    /// </summary>
    /// <typeparam name="TResult">The type to deserialize to.</typeparam>
    /// <param name="json">The JSON string.</param>
    /// <returns>The deserialized object.</returns>
    protected virtual TResult? DeserializeFromJson<TResult>(string json) where TResult : class
    {
        return JsonConvert.DeserializeObject<TResult>(json, JsonSettings);
    }

    /// <summary>
    /// Sanitizes a file name to prevent path traversal attacks.
    /// </summary>
    protected static string GetSanitizedFileName(string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            throw new ArgumentException("File name cannot be null or empty.", nameof(fileName));

        var sanitized = Path.GetFileName(fileName);
        if (string.IsNullOrWhiteSpace(sanitized))
            throw new ArgumentException("Invalid file name after sanitization.", nameof(fileName));

        return sanitized;
    }

    /// <summary>
    /// Gets a sanitized path, ensuring it doesn't escape the base directory.
    /// </summary>
    /// <remarks>
    /// This method enforces strict path containment. All paths must resolve to within
    /// the base directory or match it exactly. Rooted paths that point outside the
    /// base directory are rejected to prevent directory traversal attacks.
    /// </remarks>
    protected static string GetSanitizedPath(string path, string baseDirectory)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));

        var fullPath = Path.GetFullPath(path);
        var fullBaseDir = Path.GetFullPath(baseDirectory);

        // Normalize paths for comparison (ensure they end consistently)
        if (!fullBaseDir.EndsWith(Path.DirectorySeparatorChar.ToString()))
        {
            fullBaseDir += Path.DirectorySeparatorChar;
        }

        // Check if path is within base directory or equals it
        bool isWithinBase = fullPath.StartsWith(fullBaseDir, StringComparison.OrdinalIgnoreCase) ||
                            fullPath.Equals(fullBaseDir.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase);

        if (!isWithinBase)
        {
            throw new ArgumentException($"Path '{path}' is outside the allowed directory '{baseDirectory}'.", nameof(path));
        }

        return fullPath;
    }

    /// <summary>
    /// Validates that a path is within the specified directory.
    /// </summary>
    protected static void ValidatePathWithinDirectory(string path, string directory)
    {
        var fullPath = Path.GetFullPath(path);
        var fullDir = Path.GetFullPath(directory);

        if (!fullPath.StartsWith(fullDir, StringComparison.OrdinalIgnoreCase))
        {
            throw new UnauthorizedAccessException($"Access to path '{path}' is denied. Path must be within '{directory}'.");
        }
    }

    #endregion

    /// <summary>
    /// Configuration for automatic checkpointing.
    /// </summary>
    protected class AutoCheckpointConfiguration
    {
        /// <summary>
        /// Save every N steps.
        /// </summary>
        public int SaveFrequency { get; set; }

        /// <summary>
        /// Number of recent checkpoints to keep.
        /// </summary>
        public int KeepLast { get; set; }

        /// <summary>
        /// Whether to save when metric improves.
        /// </summary>
        public bool SaveOnImprovement { get; set; }

        /// <summary>
        /// Metric to track for improvement-based saving.
        /// </summary>
        public string? MetricName { get; set; }
    }
}

/// <summary>
/// Represents the current state of auto-checkpointing.
/// </summary>
public class AutoCheckpointState
{
    /// <summary>
    /// Whether auto-checkpointing is enabled.
    /// </summary>
    public bool IsEnabled { get; }

    /// <summary>
    /// Save frequency in steps.
    /// </summary>
    public int SaveFrequency { get; }

    /// <summary>
    /// Whether saving on improvement is enabled.
    /// </summary>
    public bool SaveOnImprovement { get; }

    /// <summary>
    /// Number of recent checkpoints to keep.
    /// </summary>
    public int KeepLast { get; }

    /// <summary>
    /// Metric name for improvement tracking.
    /// </summary>
    public string? MetricName { get; }

    /// <summary>
    /// Last step at which a checkpoint was saved.
    /// </summary>
    public int LastSaveStep { get; }

    /// <summary>
    /// Best metric value seen so far.
    /// </summary>
    public double? BestMetricValue { get; }

    /// <summary>
    /// Initializes a new AutoCheckpointState.
    /// </summary>
    public AutoCheckpointState(
        bool isEnabled,
        int saveFrequency,
        bool saveOnImprovement,
        int keepLast,
        string? metricName,
        int lastSaveStep,
        double? bestMetricValue)
    {
        IsEnabled = isEnabled;
        SaveFrequency = saveFrequency;
        SaveOnImprovement = saveOnImprovement;
        KeepLast = keepLast;
        MetricName = metricName;
        LastSaveStep = lastSaveStep;
        BestMetricValue = bestMetricValue;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        if (!IsEnabled)
            return "Auto-checkpointing disabled";

        return $"Enabled: freq={SaveFrequency}, improvement={SaveOnImprovement}, keep={KeepLast}, last={LastSaveStep}, best={BestMetricValue:F4}";
    }
}
