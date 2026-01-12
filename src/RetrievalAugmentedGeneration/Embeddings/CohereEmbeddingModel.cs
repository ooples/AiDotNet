using System;
using System.Collections.Generic;
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
/// Cohere embedding model integration for high-performance embeddings.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// Cohere provides state-of-the-art embeddings with multiple model sizes optimized
/// for different use cases (English, multilingual, search, classification).
/// </remarks>
public class CohereEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private readonly string _apiKey;
    private readonly string _model;
    private readonly string _inputType;
    private readonly int _dimension;
    private readonly HttpClient _httpClient;
    private bool _disposed;

    public override int EmbeddingDimension => _dimension;
    public override int MaxTokens => 512;

    public CohereEmbeddingModel(string apiKey, string model, string inputType, int dimension = 1024, HttpClient? httpClient = null)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
            throw new ArgumentException("API key cannot be empty", nameof(apiKey));
        if (string.IsNullOrWhiteSpace(model))
            throw new ArgumentException("Model cannot be empty", nameof(model));
        if (string.IsNullOrWhiteSpace(inputType))
            throw new ArgumentException("Input type cannot be empty", nameof(inputType));
        if (dimension <= 0)
            throw new ArgumentException("Dimension must be positive", nameof(dimension));

        _apiKey = apiKey;
        _model = model;
        _inputType = inputType;
        _dimension = dimension;
        _httpClient = httpClient ?? new HttpClient();

        if (httpClient == null)
        {
            _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        }
    }

    protected override Vector<T> EmbedCore(string text)
    {
        return Task.Run(() => EmbedAsync(text)).GetAwaiter().GetResult();
    }

    private async Task<Vector<T>> EmbedAsync(string text)
    {
        var requestBody = new
        {
            texts = new[] { text },
            model = _model,
            input_type = _inputType
        };

        var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync("https://api.cohere.ai/v1/embed", content);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
            throw new HttpRequestException($"Cohere API request failed with status code {response.StatusCode}: {errorContent}");
        }

        var responseJson = await response.Content.ReadAsStringAsync();
        var result = JsonConvert.DeserializeObject<CohereEmbeddingResponse>(responseJson);

        if (result?.Embeddings == null || result.Embeddings.Count == 0)
        {
            throw new InvalidOperationException("Cohere API returned an empty or invalid response.");
        }

        var embedding = result.Embeddings[0];
        var values = new T[embedding.Length];
        for (int i = 0; i < embedding.Length; i++)
        {
            values[i] = NumOps.FromDouble(embedding[i]);
        }

        return new Vector<T>(values);
    }

    private class CohereEmbeddingResponse
    {
        [JsonProperty("embeddings")]
        public List<double[]> Embeddings { get; set; } = new();
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _httpClient.Dispose();
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }
}
