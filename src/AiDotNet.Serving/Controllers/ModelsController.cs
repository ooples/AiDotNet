using AiDotNet.Models.Results;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.Attestation;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

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
    private readonly ITierResolver _tierResolver;
    private readonly ITierPolicyProvider _tierPolicyProvider;
    private readonly IModelArtifactService _artifactService;
    private readonly IAttestationVerifier _attestationVerifier;

    /// <summary>
    /// Initializes a new instance of the ModelsController.
    /// </summary>
    /// <param name="modelRepository">The model repository service</param>
    /// <param name="logger">Logger for diagnostics</param>
    /// <param name="servingOptions">Configuration options for the serving framework</param>
    /// <param name="tierResolver">Resolves the subscription tier for the current request</param>
    /// <param name="tierPolicyProvider">Provides tier policies for artifact/key access</param>
    /// <param name="artifactService">Artifact service for tier-aware download and key release</param>
    /// <param name="attestationVerifier">Verifies attestation evidence for key release</param>
    public ModelsController(
        IModelRepository modelRepository,
        ILogger<ModelsController> logger,
        IOptions<ServingOptions> servingOptions,
        ITierResolver tierResolver,
        ITierPolicyProvider tierPolicyProvider,
        IModelArtifactService artifactService,
        IAttestationVerifier attestationVerifier)
    {
        Guard.NotNull(modelRepository);
        _modelRepository = modelRepository;
        Guard.NotNull(logger);
        _logger = logger;
        Guard.NotNull(servingOptions);
        _servingOptions = servingOptions.Value;
        Guard.NotNull(tierResolver);
        _tierResolver = tierResolver;
        Guard.NotNull(tierPolicyProvider);
        _tierPolicyProvider = tierPolicyProvider;
        Guard.NotNull(artifactService);
        _artifactService = artifactService;
        Guard.NotNull(attestationVerifier);
        _attestationVerifier = attestationVerifier;
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
            if (request == null)
            {
                return BadRequest(new LoadModelResponse
                {
                    Success = false,
                    Error = "Request body is required"
                });
            }

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

            // Ensure modelsRoot ends with directory separator for proper boundary checking
            if (!modelsRoot.EndsWith(Path.DirectorySeparatorChar.ToString()) &&
                !modelsRoot.EndsWith(Path.AltDirectorySeparatorChar.ToString()))
            {
                modelsRoot += Path.DirectorySeparatorChar;
            }

            var candidatePath = Path.GetFullPath(Path.Combine(modelsRoot, request.Path));

            // Ensure the resolved path is within the models directory (with directory boundary check)
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

            // Load model based on numeric type
            ModelInfo? loadedModelInfo;
            try
            {
                loadedModelInfo = request.NumericType switch
                {
                    NumericType.Float => LoadTypedModel<float>(request.Name, candidatePath, NumericType.Float),
                    NumericType.Decimal => LoadTypedModel<decimal>(request.Name, candidatePath, NumericType.Decimal),
                    _ => LoadTypedModel<double>(request.Name, candidatePath, NumericType.Double)
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load model '{ModelName}' from '{Path}'",
                    request.Name, candidatePath);
                return BadRequest(new LoadModelResponse
                {
                    Success = false,
                    Error = "Failed to load model."
                });
            }

            _logger.LogInformation("Successfully loaded model '{ModelName}' from '{Path}'",
                request.Name, candidatePath);

            return Ok(new LoadModelResponse
            {
                Success = true,
                ModelInfo = loadedModelInfo
            });
        }
        catch (UnauthorizedAccessException ex)
        {
            _logger.LogError(ex, "Access denied when loading model '{ModelName}'", request.Name);
            return StatusCode(403, new LoadModelResponse
            {
                Success = false,
                Error = "Access denied to model file"
            });
        }
        catch (FileNotFoundException ex)
        {
            _logger.LogError(ex, "Model file not found for '{ModelName}'", request.Name);
            return BadRequest(new LoadModelResponse
            {
                Success = false,
                Error = "Model file not found or access denied"
            });
        }
        catch (IOException ex)
        {
            _logger.LogError(ex, "I/O error loading model '{ModelName}'", request.Name);
            return StatusCode(500, new LoadModelResponse
            {
                Success = false,
                Error = "File I/O error while loading model."
            });
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "Invalid operation when loading model '{ModelName}'", request.Name);
            return StatusCode(500, new LoadModelResponse
            {
                Success = false,
                Error = "Model operation error while loading model."
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error loading model '{ModelName}'", request.Name);
            return StatusCode(500, new LoadModelResponse
            {
                Success = false,
                Error = "An unexpected error occurred while loading model."
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

        _artifactService.RemoveProtectedArtifact(modelName);

        _logger.LogInformation("Model '{ModelName}' unloaded successfully", modelName);
        return Ok(new { message = $"Model '{modelName}' unloaded successfully" });
    }

    /// <summary>
    /// Downloads the model artifact for the specified model, subject to tier enforcement.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Option A (Free) keeps the model on the server. Higher tiers may download an artifact.
    /// Enterprise (Option C) receives an encrypted artifact and must request a decryption key using attestation.
    /// </remarks>
    [HttpGet("{modelName}/artifact")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult DownloadModelArtifact(string modelName)
    {
        var tier = _tierResolver.ResolveTier(HttpContext);
        var policy = _tierPolicyProvider.GetPolicy(tier);

        if (!policy.AllowArtifactDownload)
        {
            return StatusCode(StatusCodes.Status403Forbidden, new { error = "Model artifact download is not available for this tier." });
        }

        try
        {
            if (policy.ArtifactIsEncrypted)
            {
                var protectedArtifact = _artifactService.GetOrCreateEncryptedArtifact(modelName);
                Response.Headers["X-AiDotNet-Artifact-Encrypted"] = "true";
                Response.Headers["X-AiDotNet-Artifact-Algorithm"] = protectedArtifact.Algorithm;
                Response.Headers["X-AiDotNet-Artifact-KeyId"] = protectedArtifact.KeyId;
                return PhysicalFile(protectedArtifact.EncryptedPath, "application/octet-stream", $"{modelName}.aidn.enc");
            }

            var path = _artifactService.GetPlainArtifactPath(modelName);
            Response.Headers["X-AiDotNet-Artifact-Encrypted"] = "false";
            return PhysicalFile(path, "application/octet-stream", Path.GetFileName(path));
        }
        catch (FileNotFoundException)
        {
            return NotFound(new { error = $"Model '{modelName}' not found" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to download model artifact for '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError, new { error = "Failed to download model artifact." });
        }
    }

    /// <summary>
    /// Releases the decryption key for an encrypted model artifact (Enterprise / Option C) after attestation verification.
    /// </summary>
    [HttpPost("{modelName}/artifact/key")]
    [ProducesResponseType(typeof(ModelArtifactKeyResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> ReleaseModelArtifactKey(string modelName, [FromBody] AttestationEvidence? evidence)
    {
        var tier = _tierResolver.ResolveTier(HttpContext);
        var policy = _tierPolicyProvider.GetPolicy(tier);

        if (!policy.AllowKeyRelease)
        {
            return StatusCode(StatusCodes.Status403Forbidden, new { error = "Model artifact key release is not available for this tier." });
        }

        if (policy.RequireAttestationForKeyRelease)
        {
            if (evidence == null)
            {
                return BadRequest(new { error = "Attestation evidence is required for key release." });
            }

            var attestation = await _attestationVerifier.VerifyAsync(evidence, HttpContext.RequestAborted);
            if (!attestation.IsSuccess)
            {
                return StatusCode(StatusCodes.Status403Forbidden, new { error = attestation.FailureReason ?? "Attestation failed." });
            }
        }

        try
        {
            var protectedArtifact = _artifactService.GetOrCreateEncryptedArtifact(modelName);
            var response = _artifactService.CreateKeyResponse(protectedArtifact);
            return Ok(response);
        }
        catch (FileNotFoundException)
        {
            return NotFound(new { error = $"Model '{modelName}' not found" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to release model artifact key for '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError, new { error = "Failed to release model artifact key." });
        }
    }

    /// <summary>
    /// Loads a typed model and registers it with the repository.
    /// </summary>
    /// <remarks>
    /// This method loads a serialized AiModelResult from disk and wraps it
    /// in a ServableModelWrapper for serving. The facade pattern is maintained -
    /// all configuration (LoRA, inference opts, etc.) is preserved.
    /// </remarks>
    private ModelInfo LoadTypedModel<T>(string name, string path, NumericType numericType)
    {
        // Load the serialized AiModelResult using internal constructor
        // This is accessible via InternalsVisibleTo
        var modelResult = new AiModelResult<T, Matrix<T>, Vector<T>>();
        modelResult.LoadFromFile(path);

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

        // AiModelResult.Predict returns Vector<T> (single output per sample)
        // Multi-output models are not currently supported in the serving layer
        if (outputDim > 1)
        {
            _logger.LogWarning(
                "Multi-output models (outputDim={OutputDim}) are not fully supported in serving layer; using outputDim=1",
                outputDim);
            outputDim = 1;
        }

        // Create predict functions that delegate to AiModelResult
        // This preserves all facade functionality (LoRA, inference opts, etc.)
        // Note: AiModelResult<T, Matrix<T>, Vector<T>> has Predict(Matrix<T>) -> Vector<T>
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
            // AiModelResult.Predict(Matrix<T>) returns Vector<T> with one value per sample
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
            predictBatchFunc);

        // Register with the repository
        var success = _modelRepository.LoadModel(name, servableModel, path);

        if (!success)
        {
            throw new InvalidOperationException($"A model with name '{name}' already exists");
        }

        _logger.LogDebug("Model '{Name}' registered with {InputDim} input dimensions and {OutputDim} output dimensions",
            name, inputDim, outputDim);

        return new ModelInfo
        {
            Name = name,
            SourcePath = path,
            NumericType = numericType,
            InputDimension = inputDim,
            OutputDimension = outputDim,
            LoadedAt = DateTime.UtcNow
        };
    }
}
