#if NETCOREAPP || NETSTANDARD2_1_OR_GREATER
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies
{
    /// <summary>
    /// Reranking using Cohere's reranking API
    /// </summary>
    /// <typeparam name="T">The numeric type for scoring</typeparam>
    public class CohereReranker<T> : RerankingStrategyBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly string _apiKey;
        private readonly string _model;
        private readonly HttpClient _httpClient;
        private const string ApiBaseUrl = "https://api.cohere.ai/v1";

        public CohereReranker(string apiKey, string model = "rerank-english-v2.0")
        {
            if (string.IsNullOrEmpty(apiKey))
                throw new ArgumentException("API key cannot be null or empty", nameof(apiKey));

            _apiKey = apiKey;
            _model = model;
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {_apiKey}");
        }

        protected override async Task<List<Document<T>>> RerankCoreAsync(string query, List<Document<T>> documents, int topK)
        {
            if (documents == null || documents.Count == 0)
                return new List<Document<T>>();

            var requestBody = new
            {
                model = _model,
                query,
                documents = documents.Select(d => d.Content).ToArray(),
                top_n = topK
            };

            var json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync($"{ApiBaseUrl}/rerank", content);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            var document = JsonDocument.Parse(responseJson);
            var results = document.RootElement.GetProperty("results");

            var rerankedDocs = new List<Document<T>>();
            foreach (var result in results.EnumerateArray())
            {
                var index = result.GetProperty("index").GetInt32();
                if (index < documents.Count)
                {
                    rerankedDocs.Add(documents[index]);
                }
            }

            return rerankedDocs;
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

#endif
