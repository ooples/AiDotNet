using System.Diagnostics;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Controller for model inference operations.
/// Handles prediction requests and routes them through the request batcher
/// for high-performance batch processing.
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class InferenceController : ControllerBase
{
    private readonly IModelRepository _modelRepository;
    private readonly IRequestBatcher _requestBatcher;
    private readonly ILogger<InferenceController> _logger;

    /// <summary>
    /// Initializes a new instance of the InferenceController.
    /// </summary>
    /// <param name="modelRepository">The model repository service</param>
    /// <param name="requestBatcher">The request batcher service</param>
    /// <param name="logger">Logger for diagnostics</param>
    public InferenceController(
        IModelRepository modelRepository,
        IRequestBatcher requestBatcher,
        ILogger<InferenceController> logger)
    {
        _modelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        _requestBatcher = requestBatcher ?? throw new ArgumentNullException(nameof(requestBatcher));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Performs prediction using the specified model.
    /// Requests are automatically batched for optimal throughput.
    /// </summary>
    /// <param name="modelName">The name of the model to use</param>
    /// <param name="request">The prediction request containing input features</param>
    /// <returns>Prediction results</returns>
    /// <response code="200">Prediction completed successfully</response>
    /// <response code="400">Invalid request format</response>
    /// <response code="404">Model not found</response>
    /// <response code="500">Error during prediction</response>
    [HttpPost("predict/{modelName}")]
    [ProducesResponseType(typeof(PredictionResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Predict(string modelName, [FromBody] PredictionRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received prediction request for model '{ModelName}'", modelName);

            // Validate request
            if (request.Features == null || request.Features.Length == 0)
            {
                return BadRequest(new { error = "Features array is required and cannot be empty" });
            }

            // Check if model exists
            var modelInfo = _modelRepository.GetModelInfo(modelName);
            if (modelInfo == null)
            {
                _logger.LogWarning("Model '{ModelName}' not found", modelName);
                return NotFound(new { error = $"Model '{modelName}' not found" });
            }

            // Validate feature dimensions
            for (int i = 0; i < request.Features.Length; i++)
            {
                if (request.Features[i].Length != modelInfo.InputDimension)
                {
                    return BadRequest(new
                    {
                        error = $"Feature vector at index {i} has {request.Features[i].Length} dimensions, " +
                                $"but model '{modelName}' expects {modelInfo.InputDimension} dimensions"
                    });
                }
            }

            // Process based on numeric type
            double[][] predictions;
            int batchSize = request.Features.Length;

            switch (modelInfo.NumericType)
            {
                case NumericType.Double:
                    predictions = await PredictWithType<double>(modelName, request.Features);
                    break;
                case NumericType.Float:
                    predictions = await PredictWithType<float>(modelName, request.Features);
                    break;
                case NumericType.Decimal:
                    predictions = await PredictWithType<decimal>(modelName, request.Features);
                    break;
                default:
                    return BadRequest(new { error = "Unsupported numeric type." });
            }

            sw.Stop();

            var response = new PredictionResponse
            {
                Predictions = predictions,
                RequestId = request.RequestId,
                ProcessingTimeMs = sw.ElapsedMilliseconds,
                BatchSize = batchSize
            };

            _logger.LogInformation(
                "Prediction completed for model '{ModelName}' in {ElapsedMs}ms (batch size: {BatchSize})",
                modelName, sw.ElapsedMilliseconds, batchSize);

            return Ok(response);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "Invalid operation during prediction for model '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError, new { error = "Model operation error." });
        }
        catch (NotSupportedException ex)
        {
            _logger.LogError(ex, "Unsupported operation for model '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError, new { error = "Unsupported operation." });
        }
        catch (ArgumentException ex)
        {
            _logger.LogError(ex, "Invalid argument during prediction for model '{ModelName}'", modelName);

            if (ex.Message.Contains("maximum allowed when batching is disabled", StringComparison.OrdinalIgnoreCase))
            {
                return StatusCode(StatusCodes.Status413PayloadTooLarge, new
                {
                    error = "Request batch size exceeds the allowed maximum when batching is disabled. Reduce the batch size or enable batching."
                });
            }

            return BadRequest(new { error = "Invalid input." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during prediction for model '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError, new { error = "An unexpected error occurred during prediction." });
        }
    }

    /// <summary>
    /// Performs prediction with a specific numeric type.
    /// </summary>
    private async Task<double[][]> PredictWithType<T>(string modelName, double[][] features)
    {
        string effectiveModelName = ResolveModelNameWithAdapter(modelName);
        var model = _modelRepository.GetModel<T>(effectiveModelName) ?? _modelRepository.GetModel<T>(modelName);
        if (model == null)
        {
            string attemptedNames = effectiveModelName != modelName
                ? $"'{effectiveModelName}' (with adapter) or '{modelName}'"
                : $"'{modelName}'";
            throw new InvalidOperationException($"Model {attemptedNames} was not found.");
        }

        // Respect per-model inference configuration: bypass batching when disabled.
        if (model is AiDotNet.Serving.Models.IServableModelInferenceOptions opts && !opts.EnableBatching)
        {
            const int MaxUnbatchedItems = 1000;
            if (features.Length > MaxUnbatchedItems)
            {
                _logger.LogWarning(
                    "Rejected large unbatched request ({Count} items) for model '{ModelName}' (batching disabled)",
                    features.Length,
                    modelName);

                throw new ArgumentException(
                    $"Request batch size ({features.Length}) exceeds the maximum allowed when batching is disabled ({MaxUnbatchedItems}). " +
                    $"Enable batching for model '{modelName}' or split the request into smaller batches.");
            }

            _logger.LogDebug(
                "Batching disabled for model '{ModelName}', processing {Count} items individually",
                modelName,
                features.Length);

            var predictions = new double[features.Length][];
            for (int i = 0; i < features.Length; i++)
            {
                var inputVector = ConvertToVector<T>(features[i]);
                var resultVector = model.Predict(inputVector);
                predictions[i] = ConvertFromVector(resultVector);
            }

            return predictions;
        }

        // Queue all requests first to enable batching
        var tasks = features.Select(featureArray =>
        {
            var inputVector = ConvertToVector<T>(featureArray);
            return _requestBatcher.QueueRequest(effectiveModelName, inputVector);
        }).ToArray();

        // Await all requests together
        var resultVectors = await Task.WhenAll(tasks);

        // Convert results back to double arrays
        var batchedPredictions = new double[resultVectors.Length][];
        for (int i = 0; i < resultVectors.Length; i++)
        {
            batchedPredictions[i] = ConvertFromVector(resultVectors[i]);
        }

        return batchedPredictions;
    }

    private string ResolveModelNameWithAdapter(string modelName)
    {
        // Multi-LoRA / adapter routing (serving-first): select a pre-loaded model variant via request header.
        // This keeps adapter details out of the public model facade while enabling per-request selection.
        if (Request?.Headers == null)
        {
            _logger.LogDebug("No request headers available; routing to base model '{ModelName}'.", modelName);
            return modelName;
        }

        if (!Request.Headers.TryGetValue("X-AiDotNet-Lora", out var adapterValues) &&
            !Request.Headers.TryGetValue("X-AiDotNet-Adapter", out adapterValues))
        {
            _logger.LogDebug("No adapter header present; routing to base model '{ModelName}'.", modelName);
            return modelName;
        }

        var adapterId = adapterValues.ToString()?.Trim();
        if (string.IsNullOrWhiteSpace(adapterId) || adapterId.Length > 64 || !IsSafeAdapterId(adapterId))
        {
            if (!string.IsNullOrWhiteSpace(adapterId))
            {
                string reason = adapterId.Length > 64 ? "TooLong" : (!IsSafeAdapterId(adapterId) ? "UnsafeCharacters" : "EmptyOrWhitespace");
                var level = adapterId.Length > 64 || !IsSafeAdapterId(adapterId) ? LogLevel.Warning : LogLevel.Debug;
                _logger.Log(level,
                    "Ignoring invalid adapter ID '{AdapterId}' for model '{ModelName}' (reason: {Reason}).",
                    adapterId,
                    modelName,
                    reason);
            }
            return modelName;
        }

        _logger.LogDebug("Routing to adapter model '{EffectiveModelName}'.", $"{modelName}__{adapterId}");
        return $"{modelName}__{adapterId}";
    }

    private static bool IsSafeAdapterId(string adapterId)
    {
        for (int i = 0; i < adapterId.Length; i++)
        {
            char c = adapterId[i];
            bool ok = (c >= 'a' && c <= 'z') ||
                      (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') ||
                      c == '-' || c == '_' || c == '.';
            if (!ok) return false;
        }

        return true;
    }

    /// <summary>
    /// Converts a double array to a Vector of the specified type.
    /// </summary>
    private static Vector<T> ConvertToVector<T>(double[] values)
    {
        var result = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = ConvertValue<T>(values[i]);
        }
        return result;
    }

    /// <summary>
    /// Converts a Vector back to a double array.
    /// </summary>
    private static double[] ConvertFromVector<T>(Vector<T> vector)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = Convert.ToDouble(vector[i]);
        }
        return result;
    }

    /// <summary>
    /// Converts a double value to the specified type.
    /// </summary>
    private static T ConvertValue<T>(double value)
    {
        return (T)Convert.ChangeType(value, typeof(T));
    }

    /// <summary>
    /// Gets statistics about the request batcher's performance.
    /// </summary>
    /// <returns>Batcher statistics</returns>
    /// <response code="200">Returns batcher statistics</response>
    [HttpGet("stats")]
    [ProducesResponseType(typeof(Dictionary<string, object>), StatusCodes.Status200OK)]
    public ActionResult<Dictionary<string, object>> GetStatistics()
    {
        var stats = _requestBatcher.GetStatistics();
        return Ok(stats);
    }

    /// <summary>
    /// Gets detailed performance metrics including latency percentiles, throughput,
    /// batch utilization, and queue depth monitoring.
    /// </summary>
    /// <returns>Detailed performance metrics</returns>
    /// <response code="200">Returns detailed performance metrics</response>
    [HttpGet("metrics")]
    [ProducesResponseType(typeof(Dictionary<string, object>), StatusCodes.Status200OK)]
    public ActionResult<Dictionary<string, object>> GetPerformanceMetrics()
    {
        var metrics = _requestBatcher.GetPerformanceMetrics();
        return Ok(metrics);
    }

    /// <summary>
    /// Performs text generation using speculative decoding for accelerated inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Speculative decoding uses a smaller draft model to propose candidate tokens,
    /// which are then verified by the target model. This can significantly speed up
    /// inference for large language models.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of generating one token at a time (slow),
    /// this endpoint uses a fast draft model to guess multiple tokens at once.
    /// The main model then verifies these guesses in parallel, accepting correct ones
    /// and regenerating incorrect ones. This typically provides 2-3x speedup.
    /// </para>
    /// </remarks>
    /// <param name="modelName">The name of the target model to use for generation</param>
    /// <param name="request">The speculative decoding request</param>
    /// <returns>Generated tokens and statistics</returns>
    /// <response code="200">Generation completed successfully</response>
    /// <response code="400">Invalid request format</response>
    /// <response code="404">Model not found</response>
    /// <response code="501">Speculative decoding not supported for this model</response>
    [HttpPost("generate/{modelName}")]
    [ProducesResponseType(typeof(SpeculativeDecodingResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status501NotImplemented)]
    public IActionResult GenerateWithSpeculativeDecoding(string modelName, [FromBody] SpeculativeDecodingRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received speculative decoding request for model '{ModelName}'", modelName);

            // Validate request
            var validationError = request.Validate();
            if (validationError != null)
            {
                return BadRequest(new SpeculativeDecodingResponse
                {
                    Error = validationError,
                    RequestId = request.RequestId
                });
            }

            // Check if model exists
            var modelInfo = _modelRepository.GetModelInfo(modelName);
            if (modelInfo == null)
            {
                _logger.LogWarning("Model '{ModelName}' not found", modelName);
                return NotFound(new SpeculativeDecodingResponse
                {
                    Error = $"Model '{modelName}' not found",
                    RequestId = request.RequestId
                });
            }

            // Speculative decoding requires text generation capability which
            // depends on the model architecture. Currently, IServableModel only
            // supports vector-to-vector predictions. This endpoint documents
            // the API contract for when text generation models are supported.
            //
            // For full speculative decoding support, models need to implement:
            // - Token-level forward pass (logits for each position)
            // - Vocabulary mapping (token IDs to embeddings)
            // - Draft model integration
            //
            // This is planned for a future release with transformer model support.

            sw.Stop();

            return StatusCode(501, new SpeculativeDecodingResponse
            {
                Error = "Speculative decoding is not available via the REST API in the current version.",
                RequestId = request.RequestId,
                ProcessingTimeMs = sw.ElapsedMilliseconds
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during speculative decoding for model '{ModelName}'", modelName);
            sw.Stop();
            return StatusCode(500, new SpeculativeDecodingResponse
            {
                Error = "An unexpected error occurred during speculative decoding.",
                RequestId = request.RequestId,
                ProcessingTimeMs = sw.ElapsedMilliseconds
            });
        }
    }

    /// <summary>
    /// Applies LoRA (Low-Rank Adaptation) fine-tuning to a loaded model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// LoRA enables efficient fine-tuning by adding small adapter layers that learn
    /// task-specific adjustments without modifying the original model weights.
    /// This dramatically reduces memory and compute requirements.
    /// </para>
    /// <para><b>For Beginners:</b> LoRA lets you customize a pre-trained model
    /// for your specific use case using much less memory than traditional fine-tuning.
    /// The original model weights stay frozen while small "adapter" layers learn
    /// the adjustments needed. Typical parameter reduction: 100x or more!
    /// </para>
    /// </remarks>
    /// <param name="request">The LoRA fine-tuning request</param>
    /// <returns>Fine-tuning results and statistics</returns>
    /// <response code="200">Fine-tuning completed successfully</response>
    /// <response code="400">Invalid request format</response>
    /// <response code="404">Model not found</response>
    /// <response code="501">LoRA fine-tuning not supported for this model</response>
    [HttpPost("finetune/lora")]
    [ProducesResponseType(typeof(LoRAFineTuneResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status501NotImplemented)]
    public IActionResult FineTuneWithLoRA([FromBody] LoRAFineTuneRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received LoRA fine-tuning request for model '{ModelName}'", request.ModelName);

            // Validate request
            var validationError = request.Validate();
            if (validationError != null)
            {
                return BadRequest(new LoRAFineTuneResponse
                {
                    Success = false,
                    Error = validationError,
                    RequestId = request.RequestId,
                    ModelName = request.ModelName
                });
            }

            // Check if model exists
            var modelInfo = _modelRepository.GetModelInfo(request.ModelName);
            if (modelInfo == null)
            {
                _logger.LogWarning("Model '{ModelName}' not found", request.ModelName);
                return NotFound(new LoRAFineTuneResponse
                {
                    Success = false,
                    Error = $"Model '{request.ModelName}' not found",
                    RequestId = request.RequestId,
                    ModelName = request.ModelName
                });
            }

            // LoRA fine-tuning through REST API requires:
            // - Access to model internals (layer structure)
            // - Training loop implementation
            // - Gradient computation and backpropagation
            //
            // The current IServableModel interface encapsulates prediction only,
            // not training. For fine-tuning support, models need to expose:
            // - GetLayers() to identify adaptable layers
            // - Training API (forward, backward, update)
            //
            // LoRA adapters are available programmatically via:
            // - AiDotNet.LoRA.Adapters namespace (30+ adapter types)
            // - ILoRAConfiguration for selective layer adaptation
            // - PredictionModelBuilder.ConfigureLoRA() for model configuration

            sw.Stop();

            return StatusCode(501, new LoRAFineTuneResponse
            {
                Success = false,
                Error = "LoRA fine-tuning is not available via the REST API in the current version.",
                RequestId = request.RequestId,
                ModelName = request.ModelName,
                ProcessingTimeMs = sw.ElapsedMilliseconds
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during LoRA fine-tuning for model '{ModelName}'", request.ModelName);
            sw.Stop();
            return StatusCode(500, new LoRAFineTuneResponse
            {
                Success = false,
                Error = "An unexpected error occurred during LoRA fine-tuning.",
                RequestId = request.RequestId,
                ModelName = request.ModelName,
                ProcessingTimeMs = sw.ElapsedMilliseconds
            });
        }
    }
}
