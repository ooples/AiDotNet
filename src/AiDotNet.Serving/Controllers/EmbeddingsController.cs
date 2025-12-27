using System.Diagnostics;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Controller for multimodal embedding operations.
/// Handles text and image encoding, similarity computation, and zero-shot classification
/// for multimodal models like CLIP.
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class EmbeddingsController : ControllerBase
{
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<EmbeddingsController> _logger;

    /// <summary>
    /// Initializes a new instance of the EmbeddingsController.
    /// </summary>
    /// <param name="modelRepository">The model repository service.</param>
    /// <param name="logger">Logger for diagnostics.</param>
    public EmbeddingsController(
        IModelRepository modelRepository,
        ILogger<EmbeddingsController> logger)
    {
        _modelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Encodes text into an embedding vector using a multimodal model.
    /// </summary>
    /// <param name="modelName">The name of the multimodal model to use.</param>
    /// <param name="request">The text embedding request.</param>
    /// <returns>The embedding vector for the text.</returns>
    /// <response code="200">Text encoded successfully.</response>
    /// <response code="400">Invalid request format.</response>
    /// <response code="404">Model not found or does not support multimodal.</response>
    /// <response code="500">Error during encoding.</response>
    [HttpPost("text/{modelName}")]
    [ProducesResponseType(typeof(TextEmbeddingResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public IActionResult EncodeText(string modelName, [FromBody] TextEmbeddingRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received text embedding request for model '{ModelName}'", modelName);

            // Validate request
            if (request.Texts == null || request.Texts.Length == 0)
            {
                return BadRequest(new { error = "Texts array is required and cannot be empty" });
            }

            // Check if model exists and is multimodal
            var modelInfo = _modelRepository.GetModelInfo(modelName);
            if (modelInfo == null)
            {
                _logger.LogWarning("Model '{ModelName}' not found", modelName);
                return NotFound(new { error = $"Model '{modelName}' not found" });
            }

            if (!modelInfo.IsMultimodal)
            {
                _logger.LogWarning("Model '{ModelName}' does not support multimodal operations", modelName);
                return BadRequest(new { error = $"Model '{modelName}' is not a multimodal model" });
            }

            // Process based on numeric type
            double[][] embeddings;
            switch (modelInfo.NumericType)
            {
                case NumericType.Double:
                    embeddings = EncodeTextWithType<double>(modelName, request.Texts);
                    break;
                case NumericType.Float:
                    embeddings = EncodeTextWithType<float>(modelName, request.Texts);
                    break;
                case NumericType.Decimal:
                    embeddings = EncodeTextWithType<decimal>(modelName, request.Texts);
                    break;
                default:
                    return BadRequest(new { error = "Unsupported numeric type." });
            }

            sw.Stop();

            var response = new TextEmbeddingResponse
            {
                Embeddings = embeddings,
                RequestId = request.RequestId,
                ProcessingTimeMs = sw.ElapsedMilliseconds,
                EmbeddingDimension = embeddings.Length > 0 ? embeddings[0].Length : 0
            };

            _logger.LogInformation(
                "Text embedding completed for model '{ModelName}' in {ElapsedMs}ms (batch size: {BatchSize})",
                modelName, sw.ElapsedMilliseconds, request.Texts.Length);

            return Ok(response);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during text embedding for model '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError,
                new { error = "An unexpected error occurred during text embedding." });
        }
    }

    /// <summary>
    /// Encodes an image into an embedding vector using a multimodal model.
    /// </summary>
    /// <param name="modelName">The name of the multimodal model to use.</param>
    /// <param name="request">The image embedding request.</param>
    /// <returns>The embedding vector for the image.</returns>
    /// <response code="200">Image encoded successfully.</response>
    /// <response code="400">Invalid request format.</response>
    /// <response code="404">Model not found or does not support multimodal.</response>
    /// <response code="500">Error during encoding.</response>
    [HttpPost("image/{modelName}")]
    [ProducesResponseType(typeof(ImageEmbeddingResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public IActionResult EncodeImage(string modelName, [FromBody] ImageEmbeddingRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received image embedding request for model '{ModelName}'", modelName);

            // Validate request
            if (request.Images == null || request.Images.Length == 0)
            {
                return BadRequest(new { error = "Images array is required and cannot be empty" });
            }

            // Check if model exists and is multimodal
            var modelInfo = _modelRepository.GetModelInfo(modelName);
            if (modelInfo == null)
            {
                _logger.LogWarning("Model '{ModelName}' not found", modelName);
                return NotFound(new { error = $"Model '{modelName}' not found" });
            }

            if (!modelInfo.IsMultimodal)
            {
                _logger.LogWarning("Model '{ModelName}' does not support multimodal operations", modelName);
                return BadRequest(new { error = $"Model '{modelName}' is not a multimodal model" });
            }

            // Process based on numeric type
            double[][] embeddings;
            switch (modelInfo.NumericType)
            {
                case NumericType.Double:
                    embeddings = EncodeImageWithType<double>(modelName, request.Images);
                    break;
                case NumericType.Float:
                    embeddings = EncodeImageWithType<float>(modelName, request.Images);
                    break;
                case NumericType.Decimal:
                    embeddings = EncodeImageWithType<decimal>(modelName, request.Images);
                    break;
                default:
                    return BadRequest(new { error = "Unsupported numeric type." });
            }

            sw.Stop();

            var response = new ImageEmbeddingResponse
            {
                Embeddings = embeddings,
                RequestId = request.RequestId,
                ProcessingTimeMs = sw.ElapsedMilliseconds,
                EmbeddingDimension = embeddings.Length > 0 ? embeddings[0].Length : 0
            };

            _logger.LogInformation(
                "Image embedding completed for model '{ModelName}' in {ElapsedMs}ms (batch size: {BatchSize})",
                modelName, sw.ElapsedMilliseconds, request.Images.Length);

            return Ok(response);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during image embedding for model '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError,
                new { error = "An unexpected error occurred during image embedding." });
        }
    }

    /// <summary>
    /// Computes similarity between text and image embeddings.
    /// </summary>
    /// <param name="modelName">The name of the multimodal model to use.</param>
    /// <param name="request">The similarity computation request.</param>
    /// <returns>Similarity scores between text and image embeddings.</returns>
    /// <response code="200">Similarity computed successfully.</response>
    /// <response code="400">Invalid request format.</response>
    /// <response code="404">Model not found or does not support multimodal.</response>
    /// <response code="500">Error during computation.</response>
    [HttpPost("similarity/{modelName}")]
    [ProducesResponseType(typeof(SimilarityResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public IActionResult ComputeSimilarity(string modelName, [FromBody] SimilarityRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received similarity request for model '{ModelName}'", modelName);

            // Validate request
            if (request.TextEmbeddings == null || request.TextEmbeddings.Length == 0)
            {
                return BadRequest(new { error = "TextEmbeddings array is required and cannot be empty" });
            }

            if (request.ImageEmbeddings == null || request.ImageEmbeddings.Length == 0)
            {
                return BadRequest(new { error = "ImageEmbeddings array is required and cannot be empty" });
            }

            // Check if model exists and is multimodal
            var modelInfo = _modelRepository.GetModelInfo(modelName);
            if (modelInfo == null)
            {
                _logger.LogWarning("Model '{ModelName}' not found", modelName);
                return NotFound(new { error = $"Model '{modelName}' not found" });
            }

            if (!modelInfo.IsMultimodal)
            {
                _logger.LogWarning("Model '{ModelName}' does not support multimodal operations", modelName);
                return BadRequest(new { error = $"Model '{modelName}' is not a multimodal model" });
            }

            // Process based on numeric type
            double[][] similarities;
            switch (modelInfo.NumericType)
            {
                case NumericType.Double:
                    similarities = ComputeSimilarityWithType<double>(
                        modelName, request.TextEmbeddings, request.ImageEmbeddings);
                    break;
                case NumericType.Float:
                    similarities = ComputeSimilarityWithType<float>(
                        modelName, request.TextEmbeddings, request.ImageEmbeddings);
                    break;
                case NumericType.Decimal:
                    similarities = ComputeSimilarityWithType<decimal>(
                        modelName, request.TextEmbeddings, request.ImageEmbeddings);
                    break;
                default:
                    return BadRequest(new { error = "Unsupported numeric type." });
            }

            sw.Stop();

            var response = new SimilarityResponse
            {
                Similarities = similarities,
                RequestId = request.RequestId,
                ProcessingTimeMs = sw.ElapsedMilliseconds
            };

            _logger.LogInformation(
                "Similarity computed for model '{ModelName}' in {ElapsedMs}ms",
                modelName, sw.ElapsedMilliseconds);

            return Ok(response);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during similarity computation for model '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError,
                new { error = "An unexpected error occurred during similarity computation." });
        }
    }

    /// <summary>
    /// Performs zero-shot image classification using a multimodal model.
    /// </summary>
    /// <param name="modelName">The name of the multimodal model to use.</param>
    /// <param name="request">The zero-shot classification request.</param>
    /// <returns>Classification results with probabilities for each label.</returns>
    /// <response code="200">Classification completed successfully.</response>
    /// <response code="400">Invalid request format.</response>
    /// <response code="404">Model not found or does not support multimodal.</response>
    /// <response code="500">Error during classification.</response>
    [HttpPost("classify/{modelName}")]
    [ProducesResponseType(typeof(ZeroShotClassifyResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public IActionResult ZeroShotClassify(string modelName, [FromBody] ZeroShotClassifyRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received zero-shot classification request for model '{ModelName}'", modelName);

            // Validate request
            if (request.ImageData == null || request.ImageData.Length == 0)
            {
                return BadRequest(new { error = "ImageData is required and cannot be empty" });
            }

            if (request.ClassLabels == null || request.ClassLabels.Length == 0)
            {
                return BadRequest(new { error = "ClassLabels array is required and cannot be empty" });
            }

            // Check if model exists and is multimodal
            var modelInfo = _modelRepository.GetModelInfo(modelName);
            if (modelInfo == null)
            {
                _logger.LogWarning("Model '{ModelName}' not found", modelName);
                return NotFound(new { error = $"Model '{modelName}' not found" });
            }

            if (!modelInfo.IsMultimodal)
            {
                _logger.LogWarning("Model '{ModelName}' does not support multimodal operations", modelName);
                return BadRequest(new { error = $"Model '{modelName}' is not a multimodal model" });
            }

            // Process based on numeric type
            Dictionary<string, double> predictions;
            switch (modelInfo.NumericType)
            {
                case NumericType.Double:
                    predictions = ZeroShotClassifyWithType<double>(
                        modelName, request.ImageData, request.ClassLabels);
                    break;
                case NumericType.Float:
                    predictions = ZeroShotClassifyWithType<float>(
                        modelName, request.ImageData, request.ClassLabels);
                    break;
                case NumericType.Decimal:
                    predictions = ZeroShotClassifyWithType<decimal>(
                        modelName, request.ImageData, request.ClassLabels);
                    break;
                default:
                    return BadRequest(new { error = "Unsupported numeric type." });
            }

            sw.Stop();

            var response = new ZeroShotClassifyResponse
            {
                Predictions = predictions,
                RequestId = request.RequestId,
                ProcessingTimeMs = sw.ElapsedMilliseconds,
                TopLabel = predictions.OrderByDescending(p => p.Value).First().Key
            };

            _logger.LogInformation(
                "Zero-shot classification completed for model '{ModelName}' in {ElapsedMs}ms (labels: {LabelCount})",
                modelName, sw.ElapsedMilliseconds, request.ClassLabels.Length);

            return Ok(response);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during zero-shot classification for model '{ModelName}'", modelName);
            return StatusCode(StatusCodes.Status500InternalServerError,
                new { error = "An unexpected error occurred during zero-shot classification." });
        }
    }

    /// <summary>
    /// Gets information about a multimodal model including its capabilities.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <returns>Model information and multimodal capabilities.</returns>
    /// <response code="200">Model information retrieved successfully.</response>
    /// <response code="404">Model not found.</response>
    [HttpGet("info/{modelName}")]
    [ProducesResponseType(typeof(MultimodalModelInfo), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult GetModelInfo(string modelName)
    {
        var modelInfo = _modelRepository.GetModelInfo(modelName);
        if (modelInfo == null)
        {
            return NotFound(new { error = $"Model '{modelName}' not found" });
        }

        var response = new MultimodalModelInfo
        {
            Name = modelInfo.Name,
            IsMultimodal = modelInfo.IsMultimodal,
            EmbeddingDimension = modelInfo.EmbeddingDimension,
            MaxSequenceLength = modelInfo.MaxSequenceLength,
            ImageSize = modelInfo.ImageSize,
            LoadedAt = modelInfo.LoadedAt,
            SourcePath = modelInfo.SourcePath
        };

        return Ok(response);
    }

    private double[][] EncodeTextWithType<T>(string modelName, string[] texts)
    {
        var model = _modelRepository.GetMultimodalModel<T>(modelName);
        if (model == null)
        {
            throw new InvalidOperationException($"Multimodal model '{modelName}' not found or type mismatch");
        }

        if (texts.Length == 1)
        {
            var embedding = model.EncodeText(texts[0]);
            return new[] { ConvertFromVector(embedding) };
        }

        var embeddings = model.EncodeTextBatch(texts);
        return ConvertFromMatrix(embeddings);
    }

    private double[][] EncodeImageWithType<T>(string modelName, double[][] images)
    {
        var model = _modelRepository.GetMultimodalModel<T>(modelName);
        if (model == null)
        {
            throw new InvalidOperationException($"Multimodal model '{modelName}' not found or type mismatch");
        }

        if (images.Length == 1)
        {
            var imageVector = ConvertToVector<T>(images[0]);
            var embedding = model.EncodeImage(imageVector);
            return new[] { ConvertFromVector(embedding) };
        }

        var imageVectors = images.Select(ConvertToVector<T>);
        var embeddings = model.EncodeImageBatch(imageVectors);
        return ConvertFromMatrix(embeddings);
    }

    private double[][] ComputeSimilarityWithType<T>(string modelName, double[][] textEmbeddings, double[][] imageEmbeddings)
    {
        var model = _modelRepository.GetMultimodalModel<T>(modelName);
        if (model == null)
        {
            throw new InvalidOperationException($"Multimodal model '{modelName}' not found or type mismatch");
        }

        var similarities = new double[textEmbeddings.Length][];
        for (int i = 0; i < textEmbeddings.Length; i++)
        {
            similarities[i] = new double[imageEmbeddings.Length];
            var textVec = ConvertToVector<T>(textEmbeddings[i]);

            for (int j = 0; j < imageEmbeddings.Length; j++)
            {
                var imageVec = ConvertToVector<T>(imageEmbeddings[j]);
                var similarity = model.ComputeSimilarity(textVec, imageVec);
                similarities[i][j] = Convert.ToDouble(similarity);
            }
        }

        return similarities;
    }

    private Dictionary<string, double> ZeroShotClassifyWithType<T>(string modelName, double[] imageData, string[] classLabels)
    {
        var model = _modelRepository.GetMultimodalModel<T>(modelName);
        if (model == null)
        {
            throw new InvalidOperationException($"Multimodal model '{modelName}' not found or type mismatch");
        }

        var imageVector = ConvertToVector<T>(imageData);
        var result = model.ZeroShotClassify(imageVector, classLabels);
        return result.ToDictionary(kvp => kvp.Key, kvp => Convert.ToDouble(kvp.Value));
    }

    private static Vector<T> ConvertToVector<T>(double[] values)
    {
        var result = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = (T)Convert.ChangeType(values[i], typeof(T));
        }
        return result;
    }

    private static double[] ConvertFromVector<T>(Vector<T> vector)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = Convert.ToDouble(vector[i]);
        }
        return result;
    }

    private static double[][] ConvertFromMatrix<T>(Matrix<T> matrix)
    {
        var result = new double[matrix.Rows][];
        for (int i = 0; i < matrix.Rows; i++)
        {
            result[i] = new double[matrix.Columns];
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i][j] = Convert.ToDouble(matrix[i, j]);
            }
        }
        return result;
    }
}

#region Request/Response DTOs

/// <summary>
/// Request for encoding text into embeddings.
/// </summary>
public class TextEmbeddingRequest
{
    /// <summary>
    /// The texts to encode.
    /// </summary>
    public string[] Texts { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Optional request identifier for tracking.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Response containing text embeddings.
/// </summary>
public class TextEmbeddingResponse
{
    /// <summary>
    /// The embedding vectors for each input text.
    /// </summary>
    public double[][] Embeddings { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// The request identifier if provided.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// The dimension of each embedding vector.
    /// </summary>
    public int EmbeddingDimension { get; set; }
}

/// <summary>
/// Request for encoding images into embeddings.
/// </summary>
public class ImageEmbeddingRequest
{
    /// <summary>
    /// The preprocessed image data as flattened arrays.
    /// Each inner array should be [channels * height * width] in CHW format.
    /// </summary>
    public double[][] Images { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Optional request identifier for tracking.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Response containing image embeddings.
/// </summary>
public class ImageEmbeddingResponse
{
    /// <summary>
    /// The embedding vectors for each input image.
    /// </summary>
    public double[][] Embeddings { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// The request identifier if provided.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// The dimension of each embedding vector.
    /// </summary>
    public int EmbeddingDimension { get; set; }
}

/// <summary>
/// Request for computing similarity between text and image embeddings.
/// </summary>
public class SimilarityRequest
{
    /// <summary>
    /// The text embeddings to compare.
    /// </summary>
    public double[][] TextEmbeddings { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// The image embeddings to compare.
    /// </summary>
    public double[][] ImageEmbeddings { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Optional request identifier for tracking.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Response containing similarity scores.
/// </summary>
public class SimilarityResponse
{
    /// <summary>
    /// Similarity matrix where [i,j] is the similarity between text i and image j.
    /// Values range from -1 to 1 for normalized embeddings.
    /// </summary>
    public double[][] Similarities { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// The request identifier if provided.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }
}

/// <summary>
/// Request for zero-shot image classification.
/// </summary>
public class ZeroShotClassifyRequest
{
    /// <summary>
    /// The preprocessed image data as a flattened array.
    /// </summary>
    public double[] ImageData { get; set; } = Array.Empty<double>();

    /// <summary>
    /// The candidate class labels for classification.
    /// </summary>
    public string[] ClassLabels { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Optional request identifier for tracking.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Response containing zero-shot classification results.
/// </summary>
public class ZeroShotClassifyResponse
{
    /// <summary>
    /// Probability scores for each class label.
    /// </summary>
    public Dictionary<string, double> Predictions { get; set; } = new();

    /// <summary>
    /// The class label with the highest probability.
    /// </summary>
    public string TopLabel { get; set; } = string.Empty;

    /// <summary>
    /// The request identifier if provided.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }
}

/// <summary>
/// Information about a multimodal model.
/// </summary>
public class MultimodalModelInfo
{
    /// <summary>
    /// The model name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Whether the model supports multimodal operations.
    /// </summary>
    public bool IsMultimodal { get; set; }

    /// <summary>
    /// The embedding dimension (null if not a multimodal model).
    /// </summary>
    public int? EmbeddingDimension { get; set; }

    /// <summary>
    /// The maximum sequence length for text (null if not a multimodal model).
    /// </summary>
    public int? MaxSequenceLength { get; set; }

    /// <summary>
    /// The expected image size in pixels (null if not a multimodal model).
    /// </summary>
    public int? ImageSize { get; set; }

    /// <summary>
    /// When the model was loaded.
    /// </summary>
    public DateTime LoadedAt { get; set; }

    /// <summary>
    /// The source path of the model (if applicable).
    /// </summary>
    public string? SourcePath { get; set; }
}

#endregion
