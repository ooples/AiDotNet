using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using Newtonsoft.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// OpenAI embedding model for generating embeddings via OpenAI API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class OpenAIEmbeddingModel<T> : EmbeddingModelBase<T>
    {
        private readonly string _apiKey;
        private readonly string _modelName;
        private readonly int _dimension;
        private readonly int _maxTokens;
        private readonly HttpClient _httpClient;
        private readonly bool _ownsHttpClient;
        private bool _disposed;

        public override int EmbeddingDimension => _dimension;
        public override int MaxTokens => _maxTokens;

        /// <summary>
        /// Initializes a new instance of the <see cref="OpenAIEmbeddingModel{T}"/> class.
        /// </summary>
        /// <param name="apiKey">The OpenAI API key.</param>
        /// <param name="modelName">The model name (default: "text-embedding-ada-002").</param>
        /// <param name="dimension">The embedding dimension (default: 1536).</param>
        /// <param name="maxTokens">The maximum tokens (default: 8191).</param>
        /// <param name="httpClient">Optional external HttpClient. If provided, the caller is responsible for configuring authentication and managing its lifecycle.</param>
        public OpenAIEmbeddingModel(string apiKey, string modelName = "text-embedding-ada-002", int dimension = 1536, int maxTokens = 8191, HttpClient? httpClient = null)
        {
            if (string.IsNullOrWhiteSpace(apiKey))
                throw new ArgumentException("API key cannot be empty", nameof(apiKey));
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be empty", nameof(modelName));
            if (dimension <= 0)
                throw new ArgumentException("Dimension must be positive", nameof(dimension));
            if (maxTokens <= 0)
                throw new ArgumentException("Max tokens must be positive", nameof(maxTokens));

            _apiKey = apiKey;
            _modelName = modelName;
            _dimension = dimension;
            _maxTokens = maxTokens;
            _ownsHttpClient = httpClient == null;
            _httpClient = httpClient ?? new HttpClient();
            
            if (_ownsHttpClient)
            {
                _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            }
        }

        protected override Vector<T> EmbedCore(string text)
        {
            return EmbedAsync(text).ConfigureAwait(false).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Asynchronously encodes a batch of texts into vector representations via the OpenAI API.
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

        protected override async Task<Matrix<T>> EmbedBatchCoreAsync(IList<string> texts)
        {
            var requestBody = new
            {
                input = texts,
                model = _modelName
            };

            var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("https://api.openai.com/v1/embeddings", content);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"OpenAI API request failed with status code {response.StatusCode}: {errorContent}");
            }

            var responseJson = await response.Content.ReadAsStringAsync();
            var result = JsonConvert.DeserializeObject<OpenAIEmbeddingResponse>(responseJson);

            if (result?.Data == null || result.Data.Count != texts.Count)
            {
                throw new InvalidOperationException("OpenAI API returned an empty or mismatched response.");
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

            var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("https://api.openai.com/v1/embeddings", content);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"OpenAI API request failed with status code {response.StatusCode}: {errorContent}");
            }

            var responseJson = await response.Content.ReadAsStringAsync();
            var result = JsonConvert.DeserializeObject<OpenAIEmbeddingResponse>(responseJson);

            if (result?.Data == null || result.Data.Count == 0)
            {
                throw new InvalidOperationException("OpenAI API returned an empty or invalid response.");
            }

            var embedding = result.Data[0].Embedding;
            var values = new T[embedding.Length];
            for (int i = 0; i < embedding.Length; i++)
            {
                values[i] = NumOps.FromDouble(embedding[i]);
            }

            return new Vector<T>(values);
        }

        private class OpenAIEmbeddingResponse
        {
            [JsonProperty("data")]
            public System.Collections.Generic.List<OpenAIEmbeddingData> Data { get; set; } = new();
        }

        private class OpenAIEmbeddingData
        {
            [JsonProperty("embedding")]
            public double[] Embedding { get; set; } = Array.Empty<double>();
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
}
