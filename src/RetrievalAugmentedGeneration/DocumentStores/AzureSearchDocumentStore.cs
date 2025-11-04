using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Document store using Azure Cognitive Search
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class AzureSearchDocumentStore<T> : DocumentStoreBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly string _serviceName;
        private readonly string _apiKey;
        private readonly string _indexName;
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public AzureSearchDocumentStore(string serviceName, string apiKey, string indexName)
        {
            if (string.IsNullOrEmpty(serviceName))
                throw new ArgumentException("Service name cannot be null or empty", nameof(serviceName));
            if (string.IsNullOrEmpty(apiKey))
                throw new ArgumentException("API key cannot be null or empty", nameof(apiKey));
            if (string.IsNullOrEmpty(indexName))
                throw new ArgumentException("Index name cannot be null or empty", nameof(indexName));

            _serviceName = serviceName;
            _apiKey = apiKey;
            _indexName = indexName;
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("api-key", _apiKey);
            _baseUrl = $"https://{_serviceName}.search.windows.net";
        }

        public override async Task AddDocumentAsync(Document<T> document)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));

            var uploadDoc = new
            {
                value = new[]
                {
                    new
                    {
                        id = document.Id,
                        content = document.Content,
                        embedding = ConvertVectorToDoubleArray(document.Embedding),
                        metadata = document.Metadata
                    }
                }
            };

            var json = JsonSerializer.Serialize(uploadDoc);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/indexes/{_indexName}/docs/index?api-version=2023-11-01",
                content);

            response.EnsureSuccessStatusCode();
        }

        public override async Task<List<Document<T>>> SearchAsync(Vector<T> queryEmbedding, int topK = 5)
        {
            if (queryEmbedding == null)
                throw new ArgumentNullException(nameof(queryEmbedding));

            var searchRequest = new
            {
                vector = new
                {
                    value = ConvertVectorToDoubleArray(queryEmbedding),
                    fields = "embedding",
                    k = topK
                },
                select = "id,content,metadata"
            };

            var json = JsonSerializer.Serialize(searchRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/indexes/{_indexName}/docs/search?api-version=2023-11-01",
                content);

            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            var document = JsonDocument.Parse(responseJson);
            var results = new List<Document<T>>();

            foreach (var result in document.RootElement.GetProperty("value").EnumerateArray())
            {
                var doc = new Document<T>
                {
                    Id = result.GetProperty("id").GetString() ?? string.Empty,
                    Content = result.GetProperty("content").GetString() ?? string.Empty,
                    Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                        result.GetProperty("metadata").GetRawText()) ?? new Dictionary<string, string>()
                };

                results.Add(doc);
            }

            return results;
        }

        public override async Task DeleteDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            var deleteDoc = new
            {
                value = new[]
                {
                    new
                    {
                        id = documentId,
                        atDelete = "delete"
                    }
                }
            };

            var json = JsonSerializer.Serialize(deleteDoc);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/indexes/{_indexName}/docs/index?api-version=2023-11-01",
                content);

            response.EnsureSuccessStatusCode();
        }

        public override async Task<Document<T>?> GetDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            try
            {
                var response = await _httpClient.GetAsync(
                    $"{_baseUrl}/indexes/{_indexName}/docs/{documentId}?api-version=2023-11-01");

                if (!response.IsSuccessStatusCode)
                    return null;

                var json = await response.Content.ReadAsStringAsync();
                var element = JsonDocument.Parse(json).RootElement;

                return new Document<T>
                {
                    Id = element.GetProperty("id").GetString() ?? string.Empty,
                    Content = element.GetProperty("content").GetString() ?? string.Empty,
                    Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                        element.GetProperty("metadata").GetRawText()) ?? new Dictionary<string, string>()
                };
            }
            catch
            {
                return null;
            }
        }

        private double[] ConvertVectorToDoubleArray(Vector<T> vector)
        {
            var result = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = Convert.ToDouble(vector[i]);
            }
            return result;
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
