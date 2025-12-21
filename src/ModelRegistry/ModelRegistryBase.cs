using AiDotNet.Interfaces;
using AiDotNet.Models;
using Newtonsoft.Json;

namespace AiDotNet.ModelRegistry;

/// <summary>
/// Base class for model registry implementations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This abstract base class provides common functionality for model
/// registry systems. It handles storage path management, versioning logic, and stage transitions
/// while leaving specific storage implementation to derived classes.
///
/// Key features:
/// - Path security validation
/// - Model versioning support
/// - Stage transition management (Development, Staging, Production, Archived)
/// - Thread-safe model tracking
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public abstract class ModelRegistryBase<T, TInput, TOutput> : IModelRegistry<T, TInput, TOutput>
{
    /// <summary>
    /// The directory where models are stored.
    /// </summary>
    protected readonly string RegistryDirectory;

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
            "System."
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

            return Type.GetType(typeName) ?? throw new JsonSerializationException(
                $"Could not resolve type '{typeName}'.");
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
    /// Initializes a new instance of the ModelRegistryBase class.
    /// </summary>
    /// <param name="registryDirectory">Directory to store models.</param>
    /// <param name="baseDirectory">Base directory for path validation. Defaults to current directory.</param>
    protected ModelRegistryBase(string? registryDirectory = null, string? baseDirectory = null)
    {
        var baseDir = baseDirectory ?? Directory.GetCurrentDirectory();
        var defaultStorage = Path.Combine(baseDir, "model_registry");
        RegistryDirectory = GetSanitizedPath(registryDirectory ?? defaultStorage, baseDir);

        EnsureRegistryDirectoryExists();
    }

    /// <summary>
    /// Registers a new model in the registry.
    /// </summary>
    public abstract string RegisterModel<TMetadata>(
        string name,
        IModel<TInput, TOutput, TMetadata> model,
        ModelMetadata<T> metadata,
        Dictionary<string, string>? tags = null) where TMetadata : class;

    /// <summary>
    /// Creates a new version of an existing model.
    /// </summary>
    public abstract int CreateModelVersion<TMetadata>(
        string modelName,
        IModel<TInput, TOutput, TMetadata> model,
        ModelMetadata<T> metadata,
        string? description = null) where TMetadata : class;

    /// <summary>
    /// Retrieves a specific model version from the registry.
    /// </summary>
    public abstract RegisteredModel<T, TInput, TOutput> GetModel(string modelName, int? version = null);

    /// <summary>
    /// Gets the latest version of a model.
    /// </summary>
    public abstract RegisteredModel<T, TInput, TOutput> GetLatestModel(string modelName);

    /// <summary>
    /// Gets the model currently in a specific stage.
    /// </summary>
    public abstract RegisteredModel<T, TInput, TOutput>? GetModelByStage(string modelName, ModelStage stage);

    /// <summary>
    /// Transitions a model version to a different stage.
    /// </summary>
    public abstract void TransitionModelStage(string modelName, int version, ModelStage targetStage, bool archivePrevious = true);

    /// <summary>
    /// Lists all models in the registry.
    /// </summary>
    public abstract List<string> ListModels(string? filter = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Lists all versions of a specific model.
    /// </summary>
    public abstract List<ModelVersionInfo<T>> ListModelVersions(string modelName);

    /// <summary>
    /// Searches for models based on criteria.
    /// </summary>
    public abstract List<RegisteredModel<T, TInput, TOutput>> SearchModels(ModelSearchCriteria<T> searchCriteria);

    /// <summary>
    /// Updates the metadata for a model version.
    /// </summary>
    public abstract void UpdateModelMetadata(string modelName, int version, ModelMetadata<T> metadata);

    /// <summary>
    /// Adds or updates tags for a model.
    /// </summary>
    public abstract void UpdateModelTags(string modelName, int version, Dictionary<string, string> tags);

    /// <summary>
    /// Deletes a specific model version.
    /// </summary>
    public abstract void DeleteModelVersion(string modelName, int version);

    /// <summary>
    /// Deletes all versions of a model.
    /// </summary>
    public abstract void DeleteModel(string modelName);

    /// <summary>
    /// Compares two model versions.
    /// </summary>
    public abstract ModelComparison<T> CompareModels(string modelName, int version1, int version2);

    /// <summary>
    /// Gets the lineage information for a model.
    /// </summary>
    public abstract ModelLineage GetModelLineage(string modelName, int version);

    /// <summary>
    /// Archives a model version.
    /// </summary>
    public abstract void ArchiveModel(string modelName, int version);

    /// <summary>
    /// Gets the storage location for a model version.
    /// </summary>
    public abstract string GetModelStoragePath(string modelName, int version);

    #region Protected Helper Methods

    /// <summary>
    /// Ensures the registry directory exists.
    /// </summary>
    protected virtual void EnsureRegistryDirectoryExists()
    {
        if (!Directory.Exists(RegistryDirectory))
        {
            Directory.CreateDirectory(RegistryDirectory);
        }
    }

    /// <summary>
    /// Gets the directory path for a model.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <returns>The sanitized directory path for the model.</returns>
    protected virtual string GetModelDirectoryPath(string modelName)
    {
        var sanitizedName = GetSanitizedFileName(modelName);
        var path = Path.Combine(RegistryDirectory, sanitizedName);
        ValidatePathWithinDirectory(path, RegistryDirectory);
        return path;
    }

    /// <summary>
    /// Gets the file path for a specific model version.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <param name="version">The version number.</param>
    /// <returns>The sanitized file path for the model version.</returns>
    protected virtual string GetModelVersionPath(string modelName, int version)
    {
        var modelDir = GetModelDirectoryPath(modelName);
        var fileName = $"v{version}.json";
        var path = Path.Combine(modelDir, fileName);
        ValidatePathWithinDirectory(path, RegistryDirectory);
        return path;
    }

    /// <summary>
    /// Validates that a model name is valid.
    /// </summary>
    /// <param name="modelName">The model name to validate.</param>
    /// <exception cref="ArgumentException">Thrown when the model name is invalid.</exception>
    protected virtual void ValidateModelName(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
    }

    /// <summary>
    /// Serializes an object to JSON.
    /// </summary>
    protected virtual string SerializeToJson(object obj)
    {
        return JsonConvert.SerializeObject(obj, JsonSettings);
    }

    /// <summary>
    /// Deserializes a JSON string to an object.
    /// </summary>
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
}
