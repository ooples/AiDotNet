using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Serving.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Service that loads models from a model registry and prepares them for serving.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This service acts as a bridge between the training infrastructure
/// (where models are versioned and stored) and the serving infrastructure (where models
/// handle predictions). It doesn't adapt interfaces - instead, it loads models from storage
/// and creates servable instances from them.
///
/// This follows the Factory/Loader pattern:
/// - Reads model metadata from the registry
/// - Loads the actual model from storage
/// - Creates a ServableModelWrapper for the Serving infrastructure
/// - Tracks registry information (version, stage) with the served model
/// </remarks>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public class ModelRegistryLoader<T, TInput, TOutput>
    where TInput : class
    where TOutput : class
{
    private readonly IModelRegistry<T, TInput, TOutput> _registry;
    private readonly IModelRepository _repository;
    private readonly object _refreshLock = new();

    /// <summary>
    /// Initializes a new instance of the ModelRegistryLoader.
    /// </summary>
    /// <param name="registry">The model registry to load models from.</param>
    /// <param name="repository">The model repository to load models into.</param>
    public ModelRegistryLoader(IModelRegistry<T, TInput, TOutput> registry, IModelRepository repository)
    {
        _registry = registry ?? throw new ArgumentNullException(nameof(registry));
        _repository = repository ?? throw new ArgumentNullException(nameof(repository));
    }

    /// <summary>
    /// Loads a specific version of a model from the registry into the serving repository.
    /// </summary>
    /// <param name="modelName">The name of the model in the registry.</param>
    /// <param name="version">The version to load. If null, loads the latest version.</param>
    /// <param name="servingName">Optional name to use in the serving repository. If null, uses modelName.</param>
    /// <param name="inputDimension">The expected number of input features.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    /// <param name="predictFunc">
    /// Reserved for future use when model deserialization from storage is implemented.
    /// Currently ignored as models require manual loading via LoadWithServableModel.
    /// </param>
    /// <returns>True if the model was loaded successfully, false otherwise.</returns>
    /// <remarks>
    /// <b>Limitation:</b> This method currently creates a placeholder wrapper that throws when prediction
    /// is attempted. For actual inference, use <see cref="LoadWithServableModel"/> with a pre-loaded model.
    /// Full model deserialization from StoragePath will be implemented in a future version.
    /// </remarks>
    public bool LoadFromRegistry(
        string modelName,
        int? version,
        string? servingName,
        int inputDimension,
        int outputDimension,
        Func<TOutput, Vector<T>>? predictFunc = null)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        // Note: predictFunc is reserved for future use when model deserialization is implemented

        // Get the registered model from the registry
        RegisteredModel<T, TInput, TOutput> registeredModel;
        try
        {
            registeredModel = version.HasValue
                ? _registry.GetModel(modelName, version.Value)
                : _registry.GetLatestModel(modelName);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to load model '{modelName}' version {version?.ToString() ?? "latest"} from registry.", ex);
        }

        // Create the servable model wrapper
        // Note: The actual model loading from StoragePath would require the model to implement IModelSerializer
        // For now, we create a wrapper that uses the registered model metadata
        var name = servingName ?? modelName;

        // Create a placeholder predict function that throws if the model isn't properly loaded
        // In a full implementation, you would load the serialized model from StoragePath
        var servableModel = new ServableModelWrapper<T>(
            name,
            inputDimension,
            outputDimension,
            input => throw new InvalidOperationException(
                $"Model '{name}' requires proper deserialization from storage path: {registeredModel.StoragePath}"),
            null,
            enableBatching: true,
            enableSpeculativeDecoding: false);

        // Load into the repository with registry metadata
        return _repository.LoadModelFromRegistry(
            name,
            servableModel,
            registeredModel.Version,
            registeredModel.Stage.ToString(),
            registeredModel.StoragePath);
    }

    /// <summary>
    /// Loads a model from a specific stage (e.g., Production) into the serving repository.
    /// </summary>
    /// <param name="modelName">The name of the model in the registry.</param>
    /// <param name="stage">The stage to load from (Development, Staging, Production).</param>
    /// <param name="servingName">Optional name to use in the serving repository.</param>
    /// <param name="inputDimension">The expected number of input features.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    /// <param name="predictFunc">
    /// Reserved for future use when model deserialization from storage is implemented.
    /// Currently ignored as models require manual loading via LoadWithServableModel.
    /// </param>
    /// <returns>True if the model was loaded successfully, false otherwise.</returns>
    /// <remarks>
    /// <b>Limitation:</b> This method currently creates a placeholder wrapper that throws when prediction
    /// is attempted. For actual inference, use <see cref="LoadWithServableModel"/> with a pre-loaded model.
    /// Full model deserialization from StoragePath will be implemented in a future version.
    /// </remarks>
    public bool LoadFromRegistryByStage(
        string modelName,
        ModelStage stage,
        string? servingName,
        int inputDimension,
        int outputDimension,
        Func<TOutput, Vector<T>>? predictFunc = null)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        // Note: predictFunc is reserved for future use when model deserialization is implemented

        // Get the model at the specified stage
        var registeredModel = _registry.GetModelByStage(modelName, stage);
        if (registeredModel == null)
        {
            throw new InvalidOperationException(
                $"No model '{modelName}' found in stage '{stage}'.");
        }

        // Create the servable model wrapper
        var name = servingName ?? modelName;

        var servableModel = new ServableModelWrapper<T>(
            name,
            inputDimension,
            outputDimension,
            input => throw new InvalidOperationException(
                $"Model '{name}' requires proper deserialization from storage path: {registeredModel.StoragePath}"),
            null,
            enableBatching: true,
            enableSpeculativeDecoding: false);

        // Load into the repository with registry metadata
        return _repository.LoadModelFromRegistry(
            name,
            servableModel,
            registeredModel.Version,
            registeredModel.Stage.ToString(),
            registeredModel.StoragePath);
    }

    /// <summary>
    /// Loads a model with a fully configured servable wrapper.
    /// </summary>
    /// <param name="modelName">The name of the model in the registry.</param>
    /// <param name="version">The version to load. If null, loads the latest version.</param>
    /// <param name="servableModel">The pre-configured servable model wrapper.</param>
    /// <param name="servingName">Optional name to use in the serving repository.</param>
    /// <returns>True if the model was loaded successfully, false otherwise.</returns>
    public bool LoadWithServableModel(
        string modelName,
        int? version,
        IServableModel<T> servableModel,
        string? servingName = null)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        if (servableModel == null)
        {
            throw new ArgumentNullException(nameof(servableModel));
        }

        // Get the registered model metadata
        RegisteredModel<T, TInput, TOutput> registeredModel;
        try
        {
            registeredModel = version.HasValue
                ? _registry.GetModel(modelName, version.Value)
                : _registry.GetLatestModel(modelName);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to get model '{modelName}' metadata from registry.", ex);
        }

        var name = servingName ?? modelName;
        return _repository.LoadModelFromRegistry(
            name,
            servableModel,
            registeredModel.Version,
            registeredModel.Stage.ToString(),
            registeredModel.StoragePath);
    }

    /// <summary>
    /// Lists all models in the registry that can potentially be loaded for serving.
    /// </summary>
    /// <param name="filter">Optional filter expression for model names.</param>
    /// <param name="tags">Optional tags to filter by.</param>
    /// <returns>List of model names available in the registry.</returns>
    public List<string> ListAvailableModels(string? filter = null, Dictionary<string, string>? tags = null)
    {
        return _registry.ListModels(filter, tags);
    }

    /// <summary>
    /// Gets information about all versions of a model in the registry.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <returns>List of version information for the model.</returns>
    public List<ModelVersionInfo<T>> GetModelVersions(string modelName)
    {
        return _registry.ListModelVersions(modelName);
    }

    /// <summary>
    /// Gets the production model for a given name if one exists.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <returns>The production model information, or null if none exists.</returns>
    public RegisteredModel<T, TInput, TOutput>? GetProductionModel(string modelName)
    {
        return _registry.GetModelByStage(modelName, ModelStage.Production);
    }

    /// <summary>
    /// Refreshes a model in the serving repository with the latest version from the registry.
    /// </summary>
    /// <param name="modelName">The name of the model to refresh.</param>
    /// <param name="servableModel">The new servable model wrapper with updated prediction logic.</param>
    /// <returns>True if the refresh was successful, false otherwise.</returns>
    public bool RefreshModel(string modelName, IServableModel<T> servableModel)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        if (servableModel == null)
        {
            throw new ArgumentNullException(nameof(servableModel));
        }

        // Lock to make unload + load atomic, preventing race conditions where
        // another thread could access a non-existent model between operations
        lock (_refreshLock)
        {
            // Unload the existing model first
            _repository.UnloadModel(modelName);

            // Load the latest version
            return LoadWithServableModel(modelName, null, servableModel, modelName);
        }
    }

    /// <summary>
    /// Promotes a model to production in the registry.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version to promote.</param>
    /// <param name="archivePrevious">Whether to archive the previous production model.</param>
    public void PromoteToProduction(string modelName, int version, bool archivePrevious = true)
    {
        _registry.TransitionModelStage(modelName, version, ModelStage.Production, archivePrevious);
    }
}
