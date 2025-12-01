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
    private void LoadTypedModel<T>(string name, string path)
    {
        // Load the serialized PredictionModelResult
        // This maintains the facade pattern - all configuration (LoRA, inference opts, etc.) is included
        var modelResult = PredictionModelResult<T, Matrix<T>, Vector<T>>.Load(path);

        // Get dimensions from the model
        var inputDim = modelResult.OptimizationResult?.BestSolution?.InputShape?.Length > 0
            ? modelResult.OptimizationResult.BestSolution.InputShape[0]
            : 1;
        var outputDim = modelResult.OptimizationResult?.BestSolution?.OutputShape?.Length > 0
            ? modelResult.OptimizationResult.BestSolution.OutputShape[0]
            : 1;

        // Create predict functions that delegate to PredictionModelResult
        // This preserves all facade functionality (LoRA, inference opts, etc.)
        Func<Vector<T>, Vector<T>> predictFunc = input => modelResult.Predict(input);

        Func<Matrix<T>, Matrix<T>> predictBatchFunc = inputs =>
        {
            // Predict each row and combine results
            var results = new Matrix<T>(inputs.Rows, outputDim);
            for (int i = 0; i < inputs.Rows; i++)
            {
                var inputRow = inputs.GetRow(i);
                var output = modelResult.Predict(inputRow);
                for (int j = 0; j < output.Length && j < outputDim; j++)
                {
                    results[i, j] = output[j];
                }
            }
            return results;
        };

        // Create a servable wrapper that implements IServableModel
        var servableModel = new ServableModelWrapper<T>(
            name,
            inputDim,
            outputDim,
            predictFunc,
            predictBatchFunc);

        // Register with the repository
        var success = _modelRepository.LoadModel(name, servableModel, path);

        if (!success)
        {
            throw new InvalidOperationException($"A model with name '{name}' already exists");
        }
    }
}
