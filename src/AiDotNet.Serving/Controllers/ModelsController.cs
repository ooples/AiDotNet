using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Controller for managing loaded models.
/// Provides endpoints to load, list, and unload models from the serving framework.
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class ModelsController : ControllerBase
{
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<ModelsController> _logger;
    private readonly ServingOptions _servingOptions;

    /// <summary>
    /// Initializes a new instance of the ModelsController.
    /// </summary>
    /// <param name="modelRepository">The model repository service</param>
    /// <param name="logger">Logger for diagnostics</param>
    /// <param name="servingOptions">Configuration options for the serving framework</param>
    public ModelsController(
        IModelRepository modelRepository,
        ILogger<ModelsController> logger,
        IOptions<ServingOptions> servingOptions)
    {
        _modelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _servingOptions = servingOptions?.Value ?? throw new ArgumentNullException(nameof(servingOptions));
    }

    /// <summary>
    /// Loads a model from a file path.
    /// </summary>
    /// <param name="request">The model loading request containing name, path, and numeric type</param>
    /// <returns>Information about the loaded model</returns>
    /// <response code="200">Model loaded successfully</response>
    /// <response code="400">Invalid request or model file not found</response>
    /// <response code="409">A model with the same name already exists</response>
    [HttpPost]
    [ProducesResponseType(typeof(LoadModelResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status409Conflict)]
    public IActionResult LoadModel([FromBody] LoadModelRequest request)
    {
        try
        {
            _logger.LogInformation("Attempting to load model '{ModelName}' from path '{Path}'",
                request.Name, request.Path);

            // Validate request
            if (string.IsNullOrWhiteSpace(request.Name))
            {
                return BadRequest(new LoadModelResponse
                {
                    Success = false,
                    Error = "Model name is required"
                });
            }

            if (string.IsNullOrWhiteSpace(request.Path))
            {
                return BadRequest(new LoadModelResponse
                {
                    Success = false,
                    Error = "Model path is required"
                });
            }

            // Check if model already exists
            if (_modelRepository.ModelExists(request.Name))
            {
                _logger.LogWarning("Model '{ModelName}' already exists", request.Name);
                return Conflict(new LoadModelResponse
                {
                    Success = false,
                    Error = $"A model with name '{request.Name}' is already loaded"
                });
            }

            // Validate and canonicalize the path to prevent directory traversal
            var modelsRoot = Path.GetFullPath(_servingOptions.ModelDirectory);
            var candidatePath = Path.GetFullPath(Path.Combine(modelsRoot, request.Path));

            // Ensure the resolved path is within the models directory
            if (!candidatePath.StartsWith(modelsRoot, StringComparison.OrdinalIgnoreCase))
            {
                _logger.LogWarning("Attempted path traversal: requested path '{Path}' resolves outside model directory",
                    request.Path);
                return BadRequest(new LoadModelResponse
                {
                    Success = false,
                    Error = "Model file not found or access denied"
                });
            }

            // Check if file exists
            if (!System.IO.File.Exists(candidatePath))
            {
                _logger.LogWarning("Model file not found at canonical path: {Path}", candidatePath);
                return BadRequest(new LoadModelResponse
                {
                    Success = false,
                    Error = "Model file not found or access denied"
                });
            }

            // LoadModel from file requires a model metadata and type registry system.
            // This is deferred to a future feature that will include:
            // - Model serialization with type metadata headers
            // - Model type registry and factory pattern
            // - License verification for premium models
            // - Integration with AiDotNet Platform (web-based model creation)

            _logger.LogWarning("LoadModel endpoint requires model metadata system. " +
                "This feature is deferred to support the broader AiDotNet Platform integration.");

            return StatusCode(501, new LoadModelResponse
            {
                Success = false,
                Error = "LoadModel from file is not yet implemented. " +
                        "This endpoint requires a model metadata and type registry system.\n\n" +
                        "Current options:\n" +
                        "1. Use IModelRepository.LoadModel<T>(name, model) programmatically\n" +
                        "2. Configure StartupModels in appsettings.json\n" +
                        "3. Track GitHub issues for REST API support roadmap\n\n" +
                        "For production deployments, see documentation at: " +
                        "https://github.com/ooples/AiDotNet/wiki"
            });

            // TODO: Implement actual model loading logic
            // Example pseudocode:
            // var model = ModelSerializer.Load<T>(request.Path);
            // var success = _modelRepository.LoadModel(request.Name, model, request.Path);
            // return Ok(new LoadModelResponse { Success = true, ModelInfo = ... });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading model '{ModelName}'", request.Name);
            return BadRequest(new LoadModelResponse
            {
                Success = false,
                Error = $"Error loading model: {ex.Message}"
            });
        }
    }

    /// <summary>
    /// Gets a list of all loaded models.
    /// </summary>
    /// <returns>List of model information</returns>
    /// <response code="200">Returns the list of loaded models</response>
    [HttpGet]
    [ProducesResponseType(typeof(List<ModelInfo>), StatusCodes.Status200OK)]
    public ActionResult<List<ModelInfo>> GetModels()
    {
        _logger.LogDebug("Retrieving list of loaded models");
        var models = _modelRepository.GetAllModelInfo();
        _logger.LogInformation("Found {Count} loaded models", models.Count);
        return Ok(models);
    }

    /// <summary>
    /// Gets information about a specific model.
    /// </summary>
    /// <param name="modelName">The name of the model</param>
    /// <returns>Model information</returns>
    /// <response code="200">Returns the model information</response>
    /// <response code="404">Model not found</response>
    [HttpGet("{modelName}")]
    [ProducesResponseType(typeof(ModelInfo), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public ActionResult<ModelInfo> GetModel(string modelName)
    {
        _logger.LogDebug("Retrieving information for model '{ModelName}'", modelName);

        var modelInfo = _modelRepository.GetModelInfo(modelName);
        if (modelInfo == null)
        {
            _logger.LogWarning("Model '{ModelName}' not found", modelName);
            return NotFound(new { error = $"Model '{modelName}' not found" });
        }

        return Ok(modelInfo);
    }

    /// <summary>
    /// Unloads a model from memory.
    /// </summary>
    /// <param name="modelName">The name of the model to unload</param>
    /// <returns>Success status</returns>
    /// <response code="200">Model unloaded successfully</response>
    /// <response code="404">Model not found</response>
    [HttpDelete("{modelName}")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult UnloadModel(string modelName)
    {
        _logger.LogInformation("Attempting to unload model '{ModelName}'", modelName);

        var success = _modelRepository.UnloadModel(modelName);
        if (!success)
        {
            _logger.LogWarning("Model '{ModelName}' not found for unloading", modelName);
            return NotFound(new { error = $"Model '{modelName}' not found" });
        }

        _logger.LogInformation("Model '{ModelName}' unloaded successfully", modelName);
        return Ok(new { message = $"Model '{modelName}' unloaded successfully" });
    }
}
