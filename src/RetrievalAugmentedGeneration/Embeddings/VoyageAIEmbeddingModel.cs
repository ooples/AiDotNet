using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using Newtonsoft.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;

/// <summary>
/// Voyage AI embedding model that calls the Voyage AI embeddings REST API.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// Sends requests to <c>https://api.voyageai.com/v1/embeddings</c> using the supplied
/// API key (Bearer authentication) and parses the returned embeddings. This is a real
/// remote model - it does not fabricate vectors when the API is unavailable.
/// </remarks>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class VoyageAIEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private const int DefaultMaxTokens = 16000;
    private const string EmbeddingsEndpoint = "https://api.voyageai.com/v1/embeddings";

    private readonly string _apiKey;
    private readonly string _model;
    private readonly string _inputType;
    private readonly int _dimension;
    private readonly HttpClient _httpClient;
    private readonly bool _ownsHttpClient;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="VoyageAIEmbeddingModel{T}"/> class.
    /// </summary>
    /// <param name="apiKey">The Voyage AI API key.</param>
    /// <param name="model">The Voyage model name (e.g., "voyage-3", "voyage-3-lite").</param>
    /// <param name="inputType">The input type ("document" or "query").</param>
    /// <param name="dimension">The embedding dimension produced by the selected model.</param>
    /// <param name="httpClient">Optional external HttpClient. If provided, the caller is responsible for configuring authentication and managing its lifecycle.</param>
    public VoyageAIEmbeddingModel(
        string apiKey,
        string model,
        string inputType,
        int dimension,
        HttpClient? httpClient = null)
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

        if (_ownsHttpClient)
        {
            _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        }
    }

    /// <inheritdoc />
    public override int EmbeddingDimension => _dimension;

    /// <inheritdoc />
    public override int MaxTokens => DefaultMaxTokens;

    /// <inheritdoc />
    protected override Vector<T> EmbedCore(string text)
    {
        // Surface API/transport failures instead of fabricating a fake vector.
        return EmbedAsync(text).ConfigureAwait(false).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Asynchronously generates an embedding for the specified text using the Voyage AI API.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>A task representing the async operation, with the resulting vector.</returns>
    public override async Task<Vector<T>> EmbedAsync(string text)
    {
        ValidateText(text);

        var result = await PostEmbeddingsAsync(new[] { text });

        if (result?.Data == null || result.Data.Count == 0)
        {
            throw new InvalidOperationException("Voyage AI API returned an empty or invalid response.");
        }

        var embedding = result.Data[0].Embedding;
        var values = new T[embedding.Length];
        for (int i = 0; i < embedding.Length; i++)
        {
            values[i] = NumOps.FromDouble(embedding[i]);
        }

        return new Vector<T>(values);
    }

    /// <summary>
    /// Asynchronously generates embeddings for the specified texts using the Voyage AI API.
    /// </summary>
    /// <param name="texts">The collection of texts to encode.</param>
    /// <returns>A task representing the async operation, with the resulting matrix.</returns>
    public override async Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
    {
        var textList = texts.ToList();
        if (textList.Count == 0)
            return new Matrix<T>(0, _dimension);

        foreach (var text in textList)
            ValidateText(text);

        return await EmbedBatchCoreAsync(textList);
    }

    /// <inheritdoc />
    protected override async Task<Matrix<T>> EmbedBatchCoreAsync(IList<string> texts)
    {
        var result = await PostEmbeddingsAsync(texts);

        if (result?.Data == null || result.Data.Count != texts.Count)
        {
            throw new InvalidOperationException("Voyage AI API returned an empty or mismatched response.");
        }

        var matrix = new Matrix<T>(texts.Count, _dimension);
        for (int i = 0; i < result.Data.Count; i++)
        {
            var embedding = result.Data[i].Embedding;
            for (int j = 0; j < embedding.Length; j++)
            {
                matrix[i, j] = NumOps.FromDouble(embedding[j]);
            }
        }

        return matrix;
    }

    private async Task<VoyageEmbeddingResponse?> PostEmbeddingsAsync(IEnumerable<string> input)
    {
        var requestBody = new
        {
            input = input,
            model = _model,
            input_type = _inputType
        };

        using var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
        using var response = await _httpClient.PostAsync(EmbeddingsEndpoint, content);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
            throw new HttpRequestException($"Voyage AI API request failed with status code {response.StatusCode}: {errorContent}");
        }

        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<VoyageEmbeddingResponse>(responseJson);
    }

    private class VoyageEmbeddingResponse
    {
        [JsonProperty("data")]
        public List<VoyageEmbeddingData> Data { get; set; } = new();
    }

    private class VoyageEmbeddingData
    {
        [JsonProperty("embedding")]
        public double[] Embedding { get; set; } = Array.Empty<double>();

        [JsonProperty("index")]
        public int Index { get; set; }
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
