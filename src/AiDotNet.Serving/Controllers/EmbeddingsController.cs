using System.Diagnostics;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Controller for multimodal embedding operations.
/// Handles text and image encoding, similarity computation, and zero-shot classification.
/// </summary>
/// <remarks>
/// <para>
/// This controller provides REST endpoints for CLIP-style multimodal models that can
/// encode both text and images into a shared embedding space for similarity comparison.
/// </para>
/// <para><b>For Beginners:</b> This controller lets you:
///
/// 1. **Encode text**: Convert text descriptions to vectors
/// 2. **Encode images**: Convert images to vectors
/// 3. **Compare**: Find how similar text and images are
/// 4. **Classify**: Label images using text descriptions
///
/// Example: Search for "sunset over mountains" in a photo collection by encoding
/// the text and finding images with similar embeddings.
/// </para>
/// </remarks>
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
    public EmbeddingsController(
        IModelRepository modelRepository,
        ILogger<EmbeddingsController> logger)
    {
        _modelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Encodes text into embedding vectors using a multimodal model.
    /// </summary>
    /// <param name="modelName">The name of the multimodal model to use.</param>
    /// <param name="request">The text encoding request.</param>
    /// <returns>Embedding vectors for the input texts.</returns>
    /// <response code="200">Encoding completed successfully.</response>
    /// <response code="400">Invalid request format.</response>
    /// <response code="404">Model not found or does not support text encoding.</response>
    [HttpPost("text/{modelName}")]
    [ProducesResponseType(typeof(TextEmbeddingResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult EncodeText(string modelName, [FromBody] TextEmbeddingRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received text embedding request for model '{ModelName}'", modelName);

            if (request.Texts == null || request.Texts.Length == 0)
            {
                return BadRequest(new { error = "Texts array is required and cannot be empty." });
            }

            var modelInfo = _modelRepository.GetModelInfo(modelName);
            if (modelInfo == null)
            {
                return NotFound(new { error = $"Model '{modelName}' not found." });
            }

            // Get the multimodal model
            var model = _modelRepository.GetMultimodalModel<float>(modelName);
            if (model == null)
            {
                return NotFound(new { error = $"Model '{modelName}' does not support multimodal embeddings." });
            }

            // Encode texts
            var embeddings = new List<double[]>();
            foreach (var text in request.Texts)
            {
                var embedding = model.EncodeText(text);
                embeddings.Add(ConvertToDoubleArray(embedding));
            }

            sw.Stop();

            return Ok(new TextEmbeddingResponse
            {
                Embeddings = embeddings.ToArray(),
                ModelName = modelName,
                EmbeddingDimension = model.EmbeddingDimension,
                ProcessingTimeMs = sw.ElapsedMilliseconds,
                RequestId = request.RequestId
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error encoding text for model '{ModelName}'", modelName);
            return StatusCode(500, new { error = "An unexpected error occurred during text encoding." });
        }
    }

    /// <summary>
    /// Encodes images into embedding vectors using a multimodal model.
    /// </summary>
    /// <param name="modelName">The name of the multimodal model to use.</param>
    /// <param name="request">The image encoding request.</param>
    /// <returns>Embedding vectors for the input images.</returns>
    /// <response code="200">Encoding completed successfully.</response>
    /// <response code="400">Invalid request format.</response>
    /// <response code="404">Model not found or does not support image encoding.</response>
    [HttpPost("image/{modelName}")]
    [ProducesResponseType(typeof(ImageEmbeddingResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult EncodeImage(string modelName, [FromBody] ImageEmbeddingRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Received image embedding request for model '{ModelName}'", modelName);

            if (request.Images == null || request.Images.Length == 0)
            {
                return BadRequest(new { error = "Images array is required and cannot be empty." });
            }

            var modelInfo = _modelRepository.GetModelInfo(modelName);
            if (modelInfo == null)
            {
                return NotFound(new { error = $"Model '{modelName}' not found." });
            }

            var model = _modelRepository.GetMultimodalModel<float>(modelName);
            if (model == null)
            {
                return NotFound(new { error = $"Model '{modelName}' does not support multimodal embeddings." });
            }

            // Validate image dimensions
            var expectedSize = model.ImageSize * model.ImageSize * 3;
            foreach (var image in request.Images)
            {
                if (image.Length != expectedSize)
                {
                    return BadRequest(new
                    {
                        error = $"Image data must have {expectedSize} values " +
                               $"(3 channels × {model.ImageSize} × {model.ImageSize}). Got {image.Length}."
                    });
                }
            }

            // Encode images
            var embeddings = new List<double[]>();
            foreach (var image in request.Images)
            {
                var embedding = model.EncodeImage(image);
                embeddings.Add(ConvertToDoubleArray(embedding));
            }

            sw.Stop();

            return Ok(new ImageEmbeddingResponse
            {
                Embeddings = embeddings.ToArray(),
                ModelName = modelName,
                EmbeddingDimension = model.EmbeddingDimension,
                ImageSize = model.ImageSize,
                ProcessingTimeMs = sw.ElapsedMilliseconds,
                RequestId = request.RequestId
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error encoding images for model '{ModelName}'", modelName);
            return StatusCode(500, new { error = "An unexpected error occurred during image encoding." });
        }
    }

    /// <summary>
    /// Computes similarity between text and image embeddings.
    /// </summary>
    /// <param name="modelName">The name of the multimodal model to use.</param>
    /// <param name="request">The similarity request.</param>
    /// <returns>Similarity scores between texts and images.</returns>
    [HttpPost("similarity/{modelName}")]
    [ProducesResponseType(typeof(SimilarityResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult ComputeSimilarity(string modelName, [FromBody] SimilarityRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            if (request.Texts == null || request.Texts.Length == 0)
            {
                return BadRequest(new { error = "Texts array is required." });
            }

            if (request.Images == null || request.Images.Length == 0)
            {
                return BadRequest(new { error = "Images array is required." });
            }

            var model = _modelRepository.GetMultimodalModel<float>(modelName);
            if (model == null)
            {
                return NotFound(new { error = $"Model '{modelName}' not found or does not support multimodal embeddings." });
            }

            // Encode all texts and images
            var textEmbeddings = request.Texts.Select(t => model.EncodeText(t)).ToList();
            var imageEmbeddings = request.Images.Select(i => model.EncodeImage(i)).ToList();

            // Compute similarity matrix [texts × images]
            var similarities = new double[request.Texts.Length][];
            for (int t = 0; t < textEmbeddings.Count; t++)
            {
                similarities[t] = new double[imageEmbeddings.Count];
                for (int i = 0; i < imageEmbeddings.Count; i++)
                {
                    similarities[t][i] = Convert.ToDouble(model.ComputeSimilarity(textEmbeddings[t], imageEmbeddings[i]));
                }
            }

            sw.Stop();

            return Ok(new SimilarityResponse
            {
                Similarities = similarities,
                TextCount = request.Texts.Length,
                ImageCount = request.Images.Length,
                ProcessingTimeMs = sw.ElapsedMilliseconds,
                RequestId = request.RequestId
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error computing similarity for model '{ModelName}'", modelName);
            return StatusCode(500, new { error = "An unexpected error occurred during similarity computation." });
        }
    }

    /// <summary>
    /// Performs zero-shot image classification.
    /// </summary>
    /// <param name="modelName">The name of the multimodal model to use.</param>
    /// <param name="request">The classification request.</param>
    /// <returns>Classification probabilities for each label.</returns>
    [HttpPost("classify/{modelName}")]
    [ProducesResponseType(typeof(ZeroShotClassifyResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult ZeroShotClassify(string modelName, [FromBody] ZeroShotClassifyRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            if (request.Image == null || request.Image.Length == 0)
            {
                return BadRequest(new { error = "Image data is required." });
            }

            if (request.Labels == null || request.Labels.Length == 0)
            {
                return BadRequest(new { error = "At least one class label is required." });
            }

            var model = _modelRepository.GetMultimodalModel<float>(modelName);
            if (model == null)
            {
                return NotFound(new { error = $"Model '{modelName}' not found or does not support multimodal embeddings." });
            }

            var classifications = model.ZeroShotClassify(request.Image, request.Labels);

            sw.Stop();

            return Ok(new ZeroShotClassifyResponse
            {
                Classifications = classifications.ToDictionary(kv => kv.Key, kv => Convert.ToDouble(kv.Value)),
                TopLabel = classifications.OrderByDescending(kv => Convert.ToDouble(kv.Value)).First().Key,
                ProcessingTimeMs = sw.ElapsedMilliseconds,
                RequestId = request.RequestId
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error classifying image for model '{ModelName}'", modelName);
            return StatusCode(500, new { error = "An unexpected error occurred during classification." });
        }
    }

    /// <summary>
    /// Gets information about multimodal model capabilities.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <returns>Model capabilities and configuration.</returns>
    [HttpGet("info/{modelName}")]
    [ProducesResponseType(typeof(MultimodalModelInfoResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult GetModelInfo(string modelName)
    {
        var model = _modelRepository.GetMultimodalModel<float>(modelName);
        if (model == null)
        {
            return NotFound(new { error = $"Model '{modelName}' not found or does not support multimodal embeddings." });
        }

        return Ok(new MultimodalModelInfoResponse
        {
            ModelName = modelName,
            EmbeddingDimension = model.EmbeddingDimension,
            MaxSequenceLength = model.MaxSequenceLength,
            ImageSize = model.ImageSize,
            SupportedModalities = model.SupportedModalities.Select(m => m.ToString()).ToArray()
        });
    }

    private static double[] ConvertToDoubleArray<T>(Vector<T> vector)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = Convert.ToDouble(vector[i]);
        }
        return result;
    }
}

#region Request/Response DTOs

/// <summary>
/// Request for text embedding generation.
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
    /// The model used for encoding.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// The dimension of each embedding vector.
    /// </summary>
    public int EmbeddingDimension { get; set; }

    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// The request identifier if provided.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Request for image embedding generation.
/// </summary>
public class ImageEmbeddingRequest
{
    /// <summary>
    /// The preprocessed image data as flattened arrays.
    /// Each array should have size: 3 × ImageSize × ImageSize.
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
    /// The model used for encoding.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// The dimension of each embedding vector.
    /// </summary>
    public int EmbeddingDimension { get; set; }

    /// <summary>
    /// The expected image size.
    /// </summary>
    public int ImageSize { get; set; }

    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// The request identifier if provided.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Request for computing text-image similarity.
/// </summary>
public class SimilarityRequest
{
    /// <summary>
    /// The texts to compare.
    /// </summary>
    public string[] Texts { get; set; } = Array.Empty<string>();

    /// <summary>
    /// The preprocessed images to compare.
    /// </summary>
    public double[][] Images { get; set; } = Array.Empty<double[]>();

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
    /// Similarity matrix [texts × images].
    /// </summary>
    public double[][] Similarities { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Number of texts compared.
    /// </summary>
    public int TextCount { get; set; }

    /// <summary>
    /// Number of images compared.
    /// </summary>
    public int ImageCount { get; set; }

    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// The request identifier if provided.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Request for zero-shot image classification.
/// </summary>
public class ZeroShotClassifyRequest
{
    /// <summary>
    /// The preprocessed image data.
    /// </summary>
    public double[] Image { get; set; } = Array.Empty<double>();

    /// <summary>
    /// The candidate class labels.
    /// </summary>
    public string[] Labels { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Optional request identifier for tracking.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Response containing classification results.
/// </summary>
public class ZeroShotClassifyResponse
{
    /// <summary>
    /// The classification probabilities for each label.
    /// </summary>
    public Dictionary<string, double> Classifications { get; set; } = new();

    /// <summary>
    /// The label with the highest probability.
    /// </summary>
    public string TopLabel { get; set; } = string.Empty;

    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// The request identifier if provided.
    /// </summary>
    public string? RequestId { get; set; }
}

/// <summary>
/// Response containing multimodal model information.
/// </summary>
public class MultimodalModelInfoResponse
{
    /// <summary>
    /// The model name.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// The embedding dimension.
    /// </summary>
    public int EmbeddingDimension { get; set; }

    /// <summary>
    /// The maximum text sequence length.
    /// </summary>
    public int MaxSequenceLength { get; set; }

    /// <summary>
    /// The expected image size.
    /// </summary>
    public int ImageSize { get; set; }

    /// <summary>
    /// The supported modalities.
    /// </summary>
    public string[] SupportedModalities { get; set; } = Array.Empty<string>();
}

#endregion
