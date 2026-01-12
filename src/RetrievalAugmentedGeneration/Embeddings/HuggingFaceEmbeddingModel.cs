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
    /// HuggingFace-based embedding model for generating embeddings via Inference API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HuggingFaceEmbeddingModel<T> : EmbeddingModelBase<T>, IDisposable
    {
        private readonly string _modelName;
        private readonly string _apiKey;
        private readonly int _dimension;
        private readonly int _maxTokens;
        private readonly HttpClient _httpClient;
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
            _httpClient = httpClient ?? new HttpClient();

            if (httpClient == null && !string.IsNullOrEmpty(_apiKey))
            {
                _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            }
        }

        protected override Vector<T> EmbedCore(string text)
        {
            return Task.Run(() => EmbedAsync(text)).GetAwaiter().GetResult();
        }

        private async Task<Vector<T>> EmbedAsync(string text)
        {
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
            // HF Inference API usually returns double[][] for feature-extraction with batch input
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
}
