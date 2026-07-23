using AiDotNet.Serving.Models;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Interface for the model repository that manages loaded models.
/// </summary>
public interface IModelRepository
{
    /// <summary>
    /// Loads a model and stores it with the given name.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model</typeparam>
    /// <param name="name">The unique name for the model</param>
    /// <param name="model">The model instance</param>
    /// <param name="sourcePath">Optional source path where the model was loaded from</param>
    /// <returns>True if the model was loaded successfully, false if a model with that name already exists</returns>
    bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null);

    /// <summary>
    /// Retrieves a model by name and type.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model</typeparam>
    /// <param name="name">The name of the model</param>
    /// <returns>The model if found, null otherwise</returns>
    IServableModel<T>? GetModel<T>(string name);

    /// <summary>
    /// Unloads a model by name.
    /// </summary>
    /// <param name="name">The name of the model to unload</param>
    /// <returns>True if the model was unloaded, false if not found</returns>
    bool UnloadModel(string name);

    /// <summary>
    /// Gets information about all loaded models.
    /// </summary>
    /// <returns>A list of model information</returns>
    List<ModelInfo> GetAllModelInfo();

    /// <summary>
    /// Gets information about a specific model.
    /// </summary>
    /// <param name="name">The name of the model</param>
    /// <returns>Model information if found, null otherwise</returns>
    ModelInfo? GetModelInfo(string name);

    /// <summary>
    /// Checks if a model with the given name exists.
    /// </summary>
    /// <param name="name">The name of the model</param>
    /// <returns>True if the model exists, false otherwise</returns>
    bool ModelExists(string name);

    /// <summary>
    /// Loads a model with registry metadata.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="name">The unique name for the model.</param>
    /// <param name="model">The model instance.</param>
    /// <param name="registryVersion">The version from the model registry.</param>
    /// <param name="registryStage">The stage from the model registry (Development, Staging, Production).</param>
    /// <param name="sourcePath">Optional source path where the model was loaded from.</param>
    /// <returns>True if the model was loaded successfully, false if a model with that name already exists.</returns>
    bool LoadModelFromRegistry<T>(
        string name,
        IServableModel<T> model,
        int registryVersion,
        string registryStage,
        string? sourcePath = null);

    /// <summary>
    /// Loads a multimodal model and stores it with the given name.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="name">The unique name for the model.</param>
    /// <param name="model">The multimodal model instance.</param>
    /// <param name="sourcePath">Optional source path where the model was loaded from.</param>
    /// <returns>True if the model was loaded successfully, false if a model with that name already exists.</returns>
    bool LoadMultimodalModel<T>(string name, IServableMultimodalModel<T> model, string? sourcePath = null);

    /// <summary>
    /// Retrieves a multimodal model by name and type.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <param name="name">The name of the model.</param>
    /// <returns>The multimodal model if found and supports multimodal operations, null otherwise.</returns>
    IServableMultimodalModel<T>? GetMultimodalModel<T>(string name);
}
