using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
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
public class GooglePalmEmbeddingModel<T> : EmbeddingModelBase<T>, IDisposable
{
    private readonly string _projectId;
    private readonly string _location;
    private readonly string _model;
    private readonly string _apiKey;
    private readonly int _dimension;
    private readonly HttpClient _httpClient;
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
    /// <param name="httpClient">Optional HttpClient to use for requests.</param>
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
        _httpClient = httpClient ?? new HttpClient();

        if (httpClient == null)
        {
            _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        }
    }

    /// <summary>
    /// Generates embeddings using Google Vertex AI API.
    /// </summary>
    protected override Vector<T> EmbedCore(string text)
    {
        return Task.Run(() => EmbedAsync(text)).GetAwaiter().GetResult();
    }

    private async Task<Vector<T>> EmbedAsync(string text)
    {
        var endpoint = $"https://{_location}-aiplatform.googleapis.com/v1/projects/{_projectId}/locations/{_location}/publishers/google/models/{_model}:predict";

        var requestBody = new
        {
            instances = new[]
            {
                new { content = text }
            }
        };

        var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync(endpoint, content);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
            throw new HttpRequestException($"Vertex AI API request failed with status code {response.StatusCode}: {errorContent}");
        }

        var responseJson = await response.Content.ReadAsStringAsync();
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

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _httpClient.Dispose();
            }
            _disposed = true;
        }
    }
}

