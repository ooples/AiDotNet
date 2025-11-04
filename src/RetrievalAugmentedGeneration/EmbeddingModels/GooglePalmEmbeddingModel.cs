using AiDotNet.LinearAlgebra;
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Embedding model using Google PaLM API for text embeddings
    /// </summary>
    /// <typeparam name="T">The numeric type for embeddings</typeparam>
    public class GooglePalmEmbeddingModel<T> : EmbeddingModelBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly string _apiKey;
        private readonly string _model;
        private readonly HttpClient _httpClient;
        private const string ApiBaseUrl = "https://generativelanguage.googleapis.com/v1beta";

        public GooglePalmEmbeddingModel(string apiKey, string model = "embedding-001", INormalizer<T>? normalizer = null)
            : base(normalizer)
        {
            if (string.IsNullOrEmpty(apiKey))
                throw new ArgumentException("API key cannot be null or empty", nameof(apiKey));

            _apiKey = apiKey;
            _model = model;
            _httpClient = new HttpClient();
        }

        protected override async Task<Vector<T>> GenerateEmbeddingCoreAsync(string text)
        {
            var requestBody = new
            {
                model = $"models/{_model}",
                content = new { parts = new[] { new { text } } }
            };

            var json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync($"{ApiBaseUrl}/models/{_model}:embedContent?key={_apiKey}", content);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            var document = JsonDocument.Parse(responseJson);
            
            var embedding = document.RootElement.GetProperty("embedding").GetProperty("values");
            var values = new T[embedding.GetArrayLength()];
            
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (T)Convert.ChangeType(embedding[i].GetDouble(), typeof(T));
            }

            var vector = new Vector<T>(values, NumOps);
            return Normalizer?.Normalize(vector) ?? vector;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _httpClient?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
