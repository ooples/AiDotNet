using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.ExperimentTracking;

/// <summary>
/// Base class for experiment tracking implementations that manage ML experiments and runs.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This abstract base class provides common functionality for experiment
/// tracking systems. It handles storage path management, security validation, and JSON serialization
/// while leaving the specific storage implementation to derived classes.
///
/// Benefits of this architecture:
/// - Consistent path security across all experiment tracker implementations
/// - Shared JSON serialization settings
/// - Common helper methods for file name sanitization
/// - Extensible design for different storage backends
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public abstract class ExperimentTrackerBase<T> : IExperimentTracker<T>
{
    /// <summary>
    /// The directory where experiment data is stored.
    /// </summary>
    protected readonly string StorageDirectory;

    /// <summary>
    /// Lock object for thread-safe operations.
    /// </summary>
    protected readonly object SyncLock = new();

    /// <summary>
    /// JSON serialization settings for consistent serialization across all trackers.
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

            var baseTypeName = ExtractBaseTypeName(typeName);

            if (AllowedSystemTypes.Contains(baseTypeName))
                return true;

            foreach (var prefix in AllowedNamespacePrefixes)
            {
                if (baseTypeName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    return true;
            }

            return false;
        }

        private static string ExtractBaseTypeName(string typeName)
        {
            var genericIndex = typeName.IndexOf('`');
            if (genericIndex > 0)
                return typeName.Substring(0, genericIndex);

            var commaIndex = typeName.IndexOf(',');
            if (commaIndex > 0)
                return typeName.Substring(0, commaIndex).Trim();

            return typeName;
        }
    }

    /// <summary>
    /// Initializes a new instance of the ExperimentTrackerBase class.
    /// </summary>
    /// <param name="storageDirectory">Directory to store experiment data.</param>
    /// <param name="baseDirectory">Base directory for path validation. Defaults to current directory.</param>
    /// <remarks>
    /// When a custom storageDirectory is provided without a baseDirectory, the storage directory
    /// itself becomes the base for path validation. This allows users to store experiments in
    /// any location they choose (like temp directories for tests) while still preventing
    /// path traversal attacks within that chosen directory.
    /// </remarks>
    protected ExperimentTrackerBase(string? storageDirectory = null, string? baseDirectory = null)
    {
        if (storageDirectory != null)
        {
            // User provided a custom storage directory
            // Use the provided base directory, or default to the storage directory itself
            var fullStoragePath = Path.GetFullPath(storageDirectory);
            var baseDir = baseDirectory != null ? Path.GetFullPath(baseDirectory) : fullStoragePath;
            StorageDirectory = GetSanitizedPath(fullStoragePath, baseDir);
        }
        else
        {
            // Using default storage directory - validate against base
            var baseDir = baseDirectory ?? Directory.GetCurrentDirectory();
            var defaultStorage = Path.Combine(baseDir, "mlruns");
            StorageDirectory = GetSanitizedPath(defaultStorage, baseDir);
        }

        EnsureStorageDirectoryExists();
    }

    /// <summary>
    /// Creates a new experiment to organize related training runs.
    /// </summary>
    public abstract string CreateExperiment(string name, string? description = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Starts a new training run within an experiment.
    /// </summary>
    public abstract IExperimentRun<T> StartRun(string experimentId, string? runName = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Gets an existing experiment by its ID.
    /// </summary>
    public abstract IExperiment GetExperiment(string experimentId);

    /// <summary>
    /// Gets an existing run by its ID.
    /// </summary>
    public abstract IExperimentRun<T> GetRun(string runId);

    /// <summary>
    /// Lists all experiments, optionally filtered by criteria.
    /// </summary>
    public abstract IEnumerable<IExperiment> ListExperiments(string? filter = null);

    /// <summary>
    /// Lists all runs in an experiment, optionally filtered by criteria.
    /// </summary>
    public abstract IEnumerable<IExperimentRun<T>> ListRuns(string experimentId, string? filter = null);

    /// <summary>
    /// Deletes an experiment and all its associated runs.
    /// </summary>
    public abstract void DeleteExperiment(string experimentId);

    /// <summary>
    /// Deletes a specific run.
    /// </summary>
    public abstract void DeleteRun(string runId);

    /// <summary>
    /// Searches for runs across all experiments based on criteria.
    /// </summary>
    public abstract IEnumerable<IExperimentRun<T>> SearchRuns(string filter, int maxResults = 100);

    #region Protected Helper Methods

    /// <summary>
    /// Ensures the storage directory exists.
    /// </summary>
    protected virtual void EnsureStorageDirectoryExists()
    {
        if (!Directory.Exists(StorageDirectory))
        {
            Directory.CreateDirectory(StorageDirectory);
        }
    }

    /// <summary>
    /// Gets the directory path for an experiment.
    /// </summary>
    /// <param name="experimentId">The experiment identifier.</param>
    /// <returns>The sanitized directory path for the experiment.</returns>
    protected virtual string GetExperimentDirectoryPath(string experimentId)
    {
        var sanitizedId = GetSanitizedFileName(experimentId);
        var path = Path.Combine(StorageDirectory, sanitizedId);
        ValidatePathWithinDirectory(path, StorageDirectory);
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
    /// <param name="fileName">The file name to sanitize.</param>
    /// <returns>The sanitized file name.</returns>
    /// <exception cref="ArgumentException">Thrown when the file name is invalid.</exception>
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
    /// <param name="path">The path to sanitize.</param>
    /// <param name="baseDirectory">The base directory to validate against.</param>
    /// <returns>The sanitized full path.</returns>
    /// <exception cref="ArgumentException">Thrown when the path is invalid or escapes the base directory.</exception>
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
    /// <param name="path">The path to validate.</param>
    /// <param name="directory">The directory the path must be within.</param>
    /// <exception cref="UnauthorizedAccessException">Thrown when the path is outside the allowed directory.</exception>
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
}
