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
    public class OpenAIEmbeddingModel<T> : EmbeddingModelBase<T>, IDisposable
    {
        private readonly string _apiKey;
        private readonly string _modelName;
        private readonly int _dimension;
        private readonly int _maxTokens;
        private readonly HttpClient _httpClient;
        private bool _disposed;

        public override int EmbeddingDimension => _dimension;
        public override int MaxTokens => _maxTokens;

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
            _httpClient = httpClient ?? new HttpClient();
            
            if (httpClient == null)
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
            var requestBody = new
            {
                input = text,
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
