using System;
using System.Collections.Generic;
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
    private readonly bool _ownsHttpClient;
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
        _ownsHttpClient = httpClient == null;
        _httpClient = httpClient ?? new HttpClient();

    }

    protected override async Task<Vector<T>> EmbedCoreAsync(string text, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var requestBody = new
        {
            texts = new[] { text },
            model = _model,
            input_type = _inputType
        };

        using var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
        using var request = CreateRequest("https://api.cohere.ai/v1/embed", content);
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Cohere API request failed with status code {response.StatusCode}: {errorContent}");
        }

        var responseJson = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
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

    /// <summary>
    /// Asynchronously generates embeddings for a batch of texts using Cohere API.
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
        var requestBody = new
        {
            texts = texts,
            model = _model,
            input_type = _inputType
        };

        using var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
        using var request = CreateRequest("https://api.cohere.ai/v1/embed", content);
        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Cohere API request failed with status code {response.StatusCode}: {errorContent}");
        }

        var responseJson = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var result = JsonConvert.DeserializeObject<CohereEmbeddingResponse>(responseJson);

        if (result?.Embeddings == null || result.Embeddings.Count != texts.Count)
        {
            throw new InvalidOperationException("Cohere API returned an empty or mismatched response.");
        }

        var matrix = new Matrix<T>(texts.Count, _dimension);
        for (int i = 0; i < result.Embeddings.Count; i++)
        {
            var embedding = result.Embeddings[i];
            for (int j = 0; j < embedding.Length; j++)
            {
                matrix[i, j] = NumOps.FromDouble(embedding[j]);
            }
        }

        return matrix;
    }

    private class CohereEmbeddingResponse
    {
        [JsonProperty("embeddings")]
        public List<double[]> Embeddings { get; set; } = new();
    }

    private HttpRequestMessage CreateRequest(string url, HttpContent content)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = content
        };
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        if (!string.IsNullOrWhiteSpace(_apiKey))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        }

        return request;
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
