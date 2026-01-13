using System;
using System.Collections.Generic;
using System.Linq;
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
    /// HuggingFace-based embedding model for generating embeddings via Inference API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HuggingFaceEmbeddingModel<T> : EmbeddingModelBase<T>
    {
        private readonly string _modelName;
        private readonly string _apiKey;
        private readonly int _dimension;
        private readonly int _maxTokens;
        private readonly HttpClient _httpClient;
        private readonly bool _ownsHttpClient;
        private bool _disposed;

        public override int EmbeddingDimension => _dimension;
        public override int MaxTokens => _maxTokens;

        public HuggingFaceEmbeddingModel(string modelName, string apiKey = "", int dimension = 768, int maxTokens = 512, HttpClient? httpClient = null)
        {
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be empty", nameof(modelName));
            if (dimension <= 0)
                throw new ArgumentException("Dimension must be positive", nameof(dimension));
            if (maxTokens <= 0)
                throw new ArgumentException("Max tokens must be positive", nameof(maxTokens));

            _modelName = modelName;
            _apiKey = apiKey ?? string.Empty;
            _dimension = dimension;
            _maxTokens = maxTokens;
            _ownsHttpClient = httpClient == null;
            _httpClient = httpClient ?? new HttpClient();

            if (!string.IsNullOrEmpty(_apiKey) && _httpClient.DefaultRequestHeaders.Authorization == null)
            {
                _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            }
        }

        protected override Vector<T> EmbedCore(string text)
        {
            return EmbedAsync(text).ConfigureAwait(false).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Asynchronously generates an embedding for the specified text using HuggingFace Inference API.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>A task representing the async operation, with the resulting vector.</returns>
        public override async Task<Vector<T>> EmbedAsync(string text)
        {
            ValidateText(text);
            var endpoint = $"https://api-inference.huggingface.co/pipeline/feature-extraction/{_modelName}";
            var requestBody = new { inputs = new[] { text } };

            var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync(endpoint, content);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HuggingFace API request failed with status code {response.StatusCode}: {errorContent}");
            }

            var responseJson = await response.Content.ReadAsStringAsync();
            var result = JsonConvert.DeserializeObject<double[][]>(responseJson);

            if (result == null || result.Length == 0)
            {
                throw new InvalidOperationException("HuggingFace API returned an empty or invalid response.");
            }

            var embedding = result[0];
            var values = new T[embedding.Length];
            for (int i = 0; i < embedding.Length; i++)
            {
                values[i] = NumOps.FromDouble(embedding[i]);
            }

            return new Vector<T>(values);
        }

        /// <summary>
        /// Asynchronously generates embeddings for the specified texts using HuggingFace Inference API.
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
            var endpoint = $"https://api-inference.huggingface.co/pipeline/feature-extraction/{_modelName}";
            var requestBody = new { inputs = texts };

            var content = new StringContent(JsonConvert.SerializeObject(requestBody), Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync(endpoint, content);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new HttpRequestException($"HuggingFace API request failed with status code {response.StatusCode}: {errorContent}");
            }

            var responseJson = await response.Content.ReadAsStringAsync();
            var result = JsonConvert.DeserializeObject<double[][]>(responseJson);

            if (result == null || result.Length != texts.Count)
            {
                throw new InvalidOperationException("HuggingFace API returned an empty or mismatched response.");
            }

            var matrix = new Matrix<T>(texts.Count, _dimension);
            for (int i = 0; i < result.Length; i++)
            {
                var embedding = result[i];
                for (int j = 0; j < embedding.Length; j++)
                {
                    matrix[i, j] = NumOps.FromDouble(embedding[j]);
                }
            }

            return matrix;
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