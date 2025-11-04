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

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Document store using Qdrant vector database
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class QdrantDocumentStore<T> : DocumentStoreBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly string _host;
        private readonly int _port;
        private readonly string _collectionName;
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public QdrantDocumentStore(string host = "localhost", int port = 6333, string collectionName = "documents")
        {
            _host = host;
            _port = port;
            _collectionName = collectionName;
            _httpClient = new HttpClient();
            _baseUrl = $"http://{_host}:{_port}";
        }

        private async Task EnsureCollectionExistsAsync()
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/collections/{_collectionName}");
            if (response.IsSuccessStatusCode)
                return;

            var createRequest = new
            {
                vectors = new
                {
                    size = 768,
                    distance = "Cosine"
                }
            };

            var json = JsonSerializer.Serialize(createRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            await _httpClient.PutAsync($"{_baseUrl}/collections/{_collectionName}", content);
        }

        public override async Task AddDocumentAsync(Document<T> document)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));

            await EnsureCollectionExistsAsync();

            var point = new
            {
                points = new[]
                {
                    new
                    {
                        id = document.Id.GetHashCode(),
                        vector = ConvertVectorToDoubleArray(document.Embedding),
                        payload = new
                        {
                            id = document.Id,
                            content = document.Content,
                            metadata = document.Metadata
                        }
                    }
                }
            };

            var json = JsonSerializer.Serialize(point);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PutAsync(
                $"{_baseUrl}/collections/{_collectionName}/points",
                content);

            response.EnsureSuccessStatusCode();
        }

        public override async Task<List<Document<T>>> SearchAsync(Vector<T> queryEmbedding, int topK = 5)
        {
            if (queryEmbedding == null)
                throw new ArgumentNullException(nameof(queryEmbedding));

            await EnsureCollectionExistsAsync();

            var searchRequest = new
            {
                vector = ConvertVectorToDoubleArray(queryEmbedding),
                limit = topK,
                with_payload = true
            };

            var json = JsonSerializer.Serialize(searchRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/collections/{_collectionName}/points/search",
                content);

            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            var document = JsonDocument.Parse(responseJson);
            var results = new List<Document<T>>();

            foreach (var result in document.RootElement.GetProperty("result").EnumerateArray())
            {
                var payload = result.GetProperty("payload");
                var doc = new Document<T>
                {
                    Id = payload.GetProperty("id").GetString() ?? string.Empty,
                    Content = payload.GetProperty("content").GetString() ?? string.Empty,
                    Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                        payload.GetProperty("metadata").GetRawText()) ?? new Dictionary<string, string>()
                };

                results.Add(doc);
            }

            return results;
        }

        public override async Task DeleteDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            var deleteRequest = new
            {
                points = new[] { documentId.GetHashCode() }
            };

            var json = JsonSerializer.Serialize(deleteRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/collections/{_collectionName}/points/delete",
                content);

            response.EnsureSuccessStatusCode();
        }

        public override async Task<Document<T>?> GetDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            var response = await _httpClient.GetAsync(
                $"{_baseUrl}/collections/{_collectionName}/points/{documentId.GetHashCode()}");

            if (!response.IsSuccessStatusCode)
                return null;

            var json = await response.Content.ReadAsStringAsync();
            var element = JsonDocument.Parse(json).RootElement;
            
            var payload = element.GetProperty("result").GetProperty("payload");
            return new Document<T>
            {
                Id = payload.GetProperty("id").GetString() ?? string.Empty,
                Content = payload.GetProperty("content").GetString() ?? string.Empty,
                Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                    payload.GetProperty("metadata").GetRawText()) ?? new Dictionary<string, string>()
            };
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

#endif
