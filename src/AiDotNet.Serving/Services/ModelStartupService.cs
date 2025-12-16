using AiDotNet.Models.Results;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Hosted service that loads models at application startup based on configuration.
/// </summary>
/// <remarks>
/// <para>
/// This service runs during application startup and loads models specified in the
/// ServingOptions.StartupModels configuration. Models are loaded as PredictionModelResult
/// instances to maintain the facade pattern and include all configured optimizations.
/// </para>
/// <para><b>For Beginners:</b> This service automatically loads models when your server starts.
///
/// Configure startup models in appsettings.json:
/// <code>
/// {
///   "ServingOptions": {
///     "StartupModels": [
///       { "Name": "my-model", "Path": "models/my-model.aidotnet", "NumericType": "double" }
///     ]
///   }
/// }
/// </code>
///
/// Benefits:
/// - Models are ready immediately when the server starts
/// - No cold start latency for first prediction
/// - Validates models exist and load correctly at startup
/// </para>
/// </remarks>
public class ModelStartupService : IHostedService
{
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<ModelStartupService> _logger;
    private readonly ServingOptions _options;

    /// <summary>
    /// Initializes a new instance of the ModelStartupService.
    /// </summary>
    /// <param name="modelRepository">The model repository to register loaded models.</param>
    /// <param name="logger">Logger for diagnostics.</param>
    /// <param name="options">Serving options containing startup model configuration.</param>
    public ModelStartupService(
        IModelRepository modelRepository,
        ILogger<ModelStartupService> logger,
        IOptions<ServingOptions> options)
    {
        _modelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Starts the service and loads configured startup models.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        if (_options.StartupModels == null || _options.StartupModels.Count == 0)
        {
            _logger.LogInformation("No startup models configured");
            return;
        }

        _logger.LogInformation("Loading {Count} startup model(s)...", _options.StartupModels.Count);

        var loadedCount = 0;
        var failedCount = 0;

        foreach (var modelConfig in _options.StartupModels)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                _logger.LogWarning("Model loading cancelled");
                break;
            }

            try
            {
                await LoadModelAsync(modelConfig);
                loadedCount++;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load startup model '{Name}' from '{Path}'",
                    modelConfig.Name, modelConfig.Path);
                failedCount++;
            }
        }

        _logger.LogInformation("Startup model loading complete: {Loaded} loaded, {Failed} failed",
            loadedCount, failedCount);

        if (failedCount > 0)
        {
            _logger.LogWarning("{Failed} startup model(s) failed to load. Check configuration and file paths.",
                failedCount);
        }
    }

    /// <summary>
    /// Stops the service. No cleanup needed for loaded models.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("ModelStartupService stopping");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Loads a single model from configuration.
    /// </summary>
    private async Task LoadModelAsync(StartupModel modelConfig)
    {
        if (string.IsNullOrWhiteSpace(modelConfig.Name))
        {
            throw new ArgumentException("Model name is required");
        }

        if (string.IsNullOrWhiteSpace(modelConfig.Path))
        {
            throw new ArgumentException($"Model path is required for '{modelConfig.Name}'");
        }

        // Resolve path relative to model directory if not absolute
        var modelPath = modelConfig.Path;
        if (!Path.IsPathRooted(modelPath))
        {
            modelPath = Path.Combine(_options.ModelDirectory, modelPath);
        }

        // Validate path is within model directory to prevent traversal attacks
        var modelsRoot = Path.GetFullPath(_options.ModelDirectory);
        if (!modelsRoot.EndsWith(Path.DirectorySeparatorChar.ToString()) &&
            !modelsRoot.EndsWith(Path.AltDirectorySeparatorChar.ToString()))
        {
            modelsRoot += Path.DirectorySeparatorChar;
        }

        var canonicalPath = Path.GetFullPath(modelPath);
        if (!canonicalPath.StartsWith(modelsRoot, StringComparison.OrdinalIgnoreCase))
        {
            throw new UnauthorizedAccessException(
                $"Model path '{modelConfig.Path}' resolves outside the allowed model directory");
        }
        modelPath = canonicalPath;

        // Validate model file exists
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        }

        _logger.LogInformation("Loading model '{Name}' from '{Path}' (type: {Type})",
            modelConfig.Name, modelPath, modelConfig.NumericType);

        // Load model based on numeric type
        // Using Task.Run to avoid blocking the startup thread for file I/O
        await Task.Run(() =>
        {
            switch (modelConfig.NumericType)
            {
                case NumericType.Float:
                    LoadTypedModel<float>(modelConfig.Name, modelPath);
                    break;
                case NumericType.Decimal:
                    LoadTypedModel<decimal>(modelConfig.Name, modelPath);
                    break;
                case NumericType.Double:
                default:
                    LoadTypedModel<double>(modelConfig.Name, modelPath);
                    break;
            }
        });

        _logger.LogInformation("Successfully loaded model '{Name}'", modelConfig.Name);
    }

    /// <summary>
    /// Loads a typed model and registers it with the repository.
    /// </summary>
    /// <remarks>
    /// This method loads a serialized PredictionModelResult from disk and wraps it
    /// in a ServableModelWrapper for serving. The facade pattern is maintained -
    /// all configuration (LoRA, inference opts, etc.) is preserved.
    /// </remarks>
    private void LoadTypedModel<T>(string name, string path)
    {
        // Load the serialized PredictionModelResult using internal constructor
        // This is accessible via InternalsVisibleTo
        var modelResult = new PredictionModelResult<T, Matrix<T>, Vector<T>>();
        modelResult.LoadFromFile(path);

        var inferenceConfig = modelResult.GetInferenceOptimizationConfigForServing();
        bool enableBatching = inferenceConfig?.EnableBatching ?? true;
        bool enableSpeculativeDecoding = inferenceConfig?.EnableSpeculativeDecoding ?? false;

        // Get dimensions from the model metadata
        var metadata = modelResult.GetModelMetadata();
        var inputDim = metadata.FeatureCount > 0 ? metadata.FeatureCount : 1;

        // Output dimension defaults to 1 for most regression/classification models
        // Use Convert.ToInt32 to handle various numeric types from JSON deserialization
        // (e.g., long, double, JsonElement)
        var outputDim = 1;
        if (metadata.Properties.TryGetValue("OutputDimension", out var outputDimValue) && outputDimValue != null)
        {
            try
            {
                outputDim = Convert.ToInt32(outputDimValue);
            }
            catch (Exception)
            {
                // If conversion fails, keep default of 1
                _logger.LogWarning("Failed to parse OutputDimension from metadata, defaulting to 1");
            }
        }

        // PredictionModelResult.Predict returns Vector<T> (single output per sample)
        // Multi-output models are not currently supported in the serving layer
        if (outputDim > 1)
        {
            _logger.LogWarning(
                "Multi-output models (outputDim={OutputDim}) are not fully supported in serving layer; using outputDim=1",
                outputDim);
            outputDim = 1;
        }

        // Create predict functions that delegate to PredictionModelResult
        // This preserves all facade functionality (LoRA, inference opts, etc.)
        // Note: PredictionModelResult<T, Matrix<T>, Vector<T>> has Predict(Matrix<T>) -> Vector<T>
        // We wrap single vectors in a matrix for prediction
        Func<Vector<T>, Vector<T>> predictFunc = input =>
        {
            // Wrap single vector as single-row matrix
            var inputMatrix = new Matrix<T>(1, input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                inputMatrix[0, i] = input[i];
            }
            return modelResult.Predict(inputMatrix);
        };

        Func<Matrix<T>, Matrix<T>> predictBatchFunc = inputs =>
        {
            // Pass entire batch for efficient batch inference
            // PredictionModelResult.Predict(Matrix<T>) returns Vector<T> with one value per sample
            var predictions = modelResult.Predict(inputs);

            // Convert Vector<T> result to Matrix<T> format (single output per sample)
            var results = new Matrix<T>(inputs.Rows, 1);
            for (int i = 0; i < predictions.Length && i < inputs.Rows; i++)
            {
                results[i, 0] = predictions[i];
            }
            return results;
        };

        // Create a servable wrapper that implements IServableModel
        var servableModel = new ServableModelWrapper<T>(
            name,
            inputDim,
            outputDim,
            predictFunc,
            predictBatchFunc,
            enableBatching: enableBatching,
            enableSpeculativeDecoding: enableSpeculativeDecoding);

        // Register with the repository
        var success = _modelRepository.LoadModel(name, servableModel, path);

        if (!success)
        {
            throw new InvalidOperationException($"A model with name '{name}' already exists");
        }

        _logger.LogDebug("Model '{Name}' registered with {InputDim} input dimensions and {OutputDim} output dimensions",
            name, inputDim, outputDim);
    }
}
