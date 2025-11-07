using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using AiDotNet.LinearAlgebra;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;

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

            // Process based on numeric type
            double[][] predictions;
            int batchSize = request.Features.Length;

            switch (modelInfo.NumericType.ToLower())
            {
                case "double":
                    predictions = await PredictWithType<double>(modelName, request.Features);
                    break;
                case "single":
                    predictions = await PredictWithType<float>(modelName, request.Features);
                    break;
                case "decimal":
                    predictions = await PredictWithType<decimal>(modelName, request.Features);
                    break;
                default:
                    return BadRequest(new { error = $"Unsupported numeric type: {modelInfo.NumericType}" });
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
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during prediction for model '{ModelName}'", modelName);
            return StatusCode(500, new { error = $"Error during prediction: {ex.Message}" });
        }
    }

    /// <summary>
    /// Performs prediction with a specific numeric type.
    /// </summary>
    private async Task<double[][]> PredictWithType<T>(string modelName, double[][] features) where T : struct
    {
        var predictions = new List<double[]>();

        // Process each input through the batcher
        foreach (var featureArray in features)
        {
            // Convert double array to Vector<T>
            var inputVector = ConvertToVector<T>(featureArray);

            // Queue the request - it will be batched with other concurrent requests
            var resultVector = await _requestBatcher.QueueRequest(modelName, inputVector);

            // Convert result back to double array
            var resultArray = ConvertFromVector(resultVector);
            predictions.Add(resultArray);
        }

        return predictions.ToArray();
    }

    /// <summary>
    /// Converts a double array to a Vector of the specified type.
    /// </summary>
    private static Vector<T> ConvertToVector<T>(double[] values) where T : struct
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
    private static double[] ConvertFromVector<T>(Vector<T> vector) where T : struct
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
    private static T ConvertValue<T>(double value) where T : struct
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
}
