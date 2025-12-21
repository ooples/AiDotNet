using System.Collections.Concurrent;
using AiDotNet.Serving.Models;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Thread-safe singleton repository for managing loaded models.
/// Uses ConcurrentDictionary to ensure safe concurrent access to models.
/// </summary>
public class ModelRepository : IModelRepository
{
    /// <summary>
    /// Thread-safe storage for models, keyed by model name.
    /// </summary>
    private readonly ConcurrentDictionary<string, ModelEntry> _models = new();

    /// <summary>
    /// Loads a model and stores it with the given name.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model</typeparam>
    /// <param name="name">The unique name for the model</param>
    /// <param name="model">The model instance</param>
    /// <param name="sourcePath">Optional source path where the model was loaded from</param>
    /// <returns>True if the model was loaded successfully, false if a model with that name already exists</returns>
    public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Model name cannot be null or empty", nameof(name));
        }

        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        var entry = new ModelEntry
        {
            Model = model,
            NumericType = typeof(T).Name.ToLower(),
            LoadedAt = DateTime.UtcNow,
            SourcePath = sourcePath
        };

        return _models.TryAdd(name, entry);
    }

    /// <summary>
    /// Retrieves a model by name and type.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model</typeparam>
    /// <param name="name">The name of the model</param>
    /// <returns>The model if found, null otherwise</returns>
    public IServableModel<T>? GetModel<T>(string name)
    {
        if (!_models.TryGetValue(name, out var entry))
        {
            return null;
        }

        // Verify the numeric type matches
        var expectedType = typeof(T).Name.ToLower();
        if (entry.NumericType != expectedType)
        {
            throw new InvalidOperationException(
                $"Model '{name}' uses numeric type '{entry.NumericType}' but was requested with type '{expectedType}'");
        }

        return entry.Model as IServableModel<T>;
    }

    /// <summary>
    /// Unloads a model by name.
    /// </summary>
    /// <param name="name">The name of the model to unload</param>
    /// <returns>True if the model was unloaded, false if not found</returns>
    public bool UnloadModel(string name)
    {
        return _models.TryRemove(name, out _);
    }

    /// <summary>
    /// Gets information about all loaded models.
    /// </summary>
    /// <returns>A list of model information</returns>
    public List<ModelInfo> GetAllModelInfo()
    {
        return _models.Select(kvp => CreateModelInfo(kvp.Key, kvp.Value)).ToList();
    }

    /// <summary>
    /// Gets information about a specific model.
    /// </summary>
    /// <param name="name">The name of the model</param>
    /// <returns>Model information if found, null otherwise</returns>
    public ModelInfo? GetModelInfo(string name)
    {
        if (!_models.TryGetValue(name, out var entry))
        {
            return null;
        }

        return CreateModelInfo(name, entry);
    }

    /// <summary>
    /// Checks if a model with the given name exists.
    /// </summary>
    /// <param name="name">The name of the model</param>
    /// <returns>True if the model exists, false otherwise</returns>
    public bool ModelExists(string name)
    {
        return _models.ContainsKey(name);
    }

    /// <summary>
    /// Creates ModelInfo from a model entry.
    /// </summary>
    private static ModelInfo CreateModelInfo(string name, ModelEntry entry)
    {
        // Use reflection to get the model's input/output dimensions
        // We need to handle different generic types dynamically
        var modelType = entry.Model.GetType();
        var inputDimProperty = modelType.GetProperty("InputDimension");
        var outputDimProperty = modelType.GetProperty("OutputDimension");

        return new ModelInfo
        {
            Name = name,
            NumericType = entry.NumericType,
            InputDimension = (int)(inputDimProperty?.GetValue(entry.Model) ?? 0),
            OutputDimension = (int)(outputDimProperty?.GetValue(entry.Model) ?? 0),
            LoadedAt = entry.LoadedAt,
            SourcePath = entry.SourcePath,
            IsFromRegistry = entry.IsFromRegistry,
            RegistryVersion = entry.RegistryVersion,
            RegistryStage = entry.RegistryStage
        };
    }

    /// <summary>
    /// Loads a model with registry metadata.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="name">The unique name for the model.</param>
    /// <param name="model">The model instance.</param>
    /// <param name="registryVersion">The version from the model registry.</param>
    /// <param name="registryStage">The stage from the model registry.</param>
    /// <param name="sourcePath">Optional source path where the model was loaded from.</param>
    /// <returns>True if the model was loaded successfully, false if a model with that name already exists.</returns>
    public bool LoadModelFromRegistry<T>(
        string name,
        IServableModel<T> model,
        int registryVersion,
        string registryStage,
        string? sourcePath = null)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Model name cannot be null or empty", nameof(name));
        }

        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        var entry = new ModelEntry
        {
            Model = model,
            NumericType = typeof(T).Name.ToLower(),
            LoadedAt = DateTime.UtcNow,
            SourcePath = sourcePath,
            IsFromRegistry = true,
            RegistryVersion = registryVersion,
            RegistryStage = registryStage
        };

        return _models.TryAdd(name, entry);
    }

    /// <summary>
    /// Internal class to store model metadata along with the model instance.
    /// </summary>
    private class ModelEntry
    {
        public object Model { get; set; } = null!;
        public string NumericType { get; set; } = string.Empty;
        public DateTime LoadedAt { get; set; }
        public string? SourcePath { get; set; }
        public bool IsFromRegistry { get; set; }
        public int? RegistryVersion { get; set; }
        public string? RegistryStage { get; set; }
    }
}
