using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using Newtonsoft.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;

/// <summary>
/// Google PaLM embedding model integration via Vertex AI.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// Provides access to Google's PaLM (Pathways Language Model) and Gemini embedding capabilities
/// through the Google Cloud Vertex AI platform.
/// </remarks>
public class GooglePalmEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private readonly string _projectId;
    private readonly string _location;
    private readonly string _model;
    private readonly string _apiKey;
    private readonly int _dimension;
    private readonly HttpClient _httpClient;
    private readonly bool _ownsHttpClient;
    private bool _disposed;

    public override int EmbeddingDimension => _dimension;
    public override int MaxTokens => 2048;

    /// <summary>
    /// Initializes a new instance of the <see cref="GooglePalmEmbeddingModel{T}"/> class.
    /// </summary>
    /// <param name="projectId">The Google Cloud project ID.</param>
    /// <param name="location">The Google Cloud location (e.g., "us-central1").</param>
    /// <param name="model">The model name (e.g., "text-embedding-004").</param>
    /// <param name="apiKey">The API key or Access Token for authentication.</param>
    /// <param name="dimension">The embedding dimension.</param>
    /// <param name="httpClient">Optional HttpClient to use for requests. If provided, the caller is responsible for its lifecycle and configuration.</param>
    public GooglePalmEmbeddingModel(
        string projectId,
        string location,
        string model,
        string apiKey,
        int dimension = 768,
        HttpClient? httpClient = null)
    {
        _projectId = projectId ?? throw new ArgumentNullException(nameof(projectId));
        _location = location ?? throw new ArgumentNullException(nameof(location));
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        _dimension = dimension;
        _ownsHttpClient = httpClient == null;
        _httpClient = httpClient ?? new HttpClient();

        if (!string.IsNullOrEmpty(_apiKey) && _httpClient.DefaultRequestHeaders.Authorization == null)
        {
            _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        }
    }

    /// <summary>
    /// Generates embeddings using Google Vertex AI API.
    /// </summary>
    protected override async Task<Vector<T>> EmbedCoreAsync(string text, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var endpoint = $"https://{_location}-aiplatform.googleapis.com/v1/projects/{_projectId}/locations/{_location}/publishers/google/models/{_model}:predict";

        var requestBody = new
        {
            instances = new[]
            {
                new { content = text }
            }
        };

        using var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
        using var request = new HttpRequestMessage(HttpMethod.Post, endpoint)
        {
            Content = content
        };
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Vertex AI API request failed with status code {response.StatusCode}: {errorContent}");
        }

        var responseJson = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var result = JsonConvert.DeserializeObject<VertexAIPredictionResponse>(responseJson);

        if (result?.Predictions == null || result.Predictions.Count == 0)
        {
            throw new InvalidOperationException("Vertex AI API returned an empty or invalid response.");
        }

        var embeddingValues = result.Predictions[0].Embeddings.Values;
        var values = new T[embeddingValues.Length];
        for (int i = 0; i < embeddingValues.Length; i++)
        {
            values[i] = NumOps.FromDouble(embeddingValues[i]);
        }

        return new Vector<T>(values);
    }

    /// <summary>
    /// Asynchronously generates embeddings for the specified texts using Google Vertex AI API.
    /// </summary>
    /// <param name="texts">The collection of texts to encode.</param>
    /// <returns>A task representing the async operation, with the resulting matrix.</returns>
    public override async Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default)
    {
        var textList = texts.ToList();
        if (textList.Count == 0)
            return new Matrix<T>(0, _dimension);

        foreach (var text in textList)
            ValidateText(text);

        return await EmbedBatchCoreAsync(textList, cancellationToken).ConfigureAwait(false);
    }

    protected override async Task<Matrix<T>> EmbedBatchCoreAsync(IList<string> texts, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var endpoint = $"https://{_location}-aiplatform.googleapis.com/v1/projects/{_projectId}/locations/{_location}/publishers/google/models/{_model}:predict";

        var requestBody = new
        {
            instances = texts.Select(t => new { content = t }).ToArray()
        };

        using var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
        using var request = new HttpRequestMessage(HttpMethod.Post, endpoint)
        {
            Content = content
        };
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Vertex AI API request failed with status code {response.StatusCode}: {errorContent}");
        }

        var responseJson = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var result = JsonConvert.DeserializeObject<VertexAIPredictionResponse>(responseJson);

        if (result?.Predictions == null || result.Predictions.Count != texts.Count)
        {
            throw new InvalidOperationException("Vertex AI API returned an empty or mismatched response.");
        }

        var matrix = new Matrix<T>(texts.Count, _dimension);
        for (int i = 0; i < result.Predictions.Count; i++)
        {
            var embeddingValues = result.Predictions[i].Embeddings.Values;
            for (int j = 0; j < embeddingValues.Length; j++)
            {
                matrix[i, j] = NumOps.FromDouble(embeddingValues[j]);
            }
        }

        return matrix;
    }

    private class VertexAIPredictionResponse
    {
        [JsonProperty("predictions")]
        public List<VertexAIPrediction> Predictions { get; set; } = new();
    }

    private class VertexAIPrediction
    {
        [JsonProperty("embeddings")]
        public VertexAIEmbedding Embeddings { get; set; } = new();
    }

    private class VertexAIEmbedding
    {
        [JsonProperty("values")]
        public double[] Values { get; set; } = Array.Empty<double>();
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing && _ownsHttpClient)
            {
                _httpClient.Dispose();
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }
}
