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
    /// Document store using ChromaDB vector database
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class ChromaDBDocumentStore<T> : DocumentStoreBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly string _host;
        private readonly int _port;
        private readonly string _collectionName;
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;
        private string? _collectionId;

        public ChromaDBDocumentStore(string host = "localhost", int port = 8000, string collectionName = "documents")
        {
            _host = host;
            _port = port;
            _collectionName = collectionName;
            _httpClient = new HttpClient();
            _baseUrl = $"http://{_host}:{_port}/api/v1";
        }

        private async Task EnsureCollectionExistsAsync()
        {
            if (_collectionId != null)
                return;

            var createRequest = new
            {
                name = _collectionName,
                metadata = new { description = "AiDotNet document collection" }
            };

            var json = JsonSerializer.Serialize(createRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            try
            {
                var response = await _httpClient.PostAsync($"{_baseUrl}/collections", content);
                var responseJson = await response.Content.ReadAsStringAsync();
                var document = JsonDocument.Parse(responseJson);
                _collectionId = document.RootElement.GetProperty("id").GetString();
            }
            catch
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/collections/{_collectionName}");
                if (response.IsSuccessStatusCode)
                {
                    var responseJson = await response.Content.ReadAsStringAsync();
                    var document = JsonDocument.Parse(responseJson);
                    _collectionId = document.RootElement.GetProperty("id").GetString();
                }
            }
        }

        public override async Task AddDocumentAsync(Document<T> document)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));

            await EnsureCollectionExistsAsync();

            var addRequest = new
            {
                ids = new[] { document.Id },
                embeddings = new[] { ConvertVectorToDoubleArray(document.Embedding) },
                documents = new[] { document.Content },
                metadatas = new[] { document.Metadata }
            };

            var json = JsonSerializer.Serialize(addRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/collections/{_collectionId}/add",
                content);

            response.EnsureSuccessStatusCode();
        }

        public override async Task<List<Document<T>>> SearchAsync(Vector<T> queryEmbedding, int topK = 5)
        {
            if (queryEmbedding == null)
                throw new ArgumentNullException(nameof(queryEmbedding));

            await EnsureCollectionExistsAsync();

            var queryRequest = new
            {
                query_embeddings = new[] { ConvertVectorToDoubleArray(queryEmbedding) },
                n_results = topK,
                include = new[] { "documents", "metadatas", "distances" }
            };

            var json = JsonSerializer.Serialize(queryRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/collections/{_collectionId}/query",
                content);

            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            var document = JsonDocument.Parse(responseJson);
            var results = new List<Document<T>>();

            var ids = document.RootElement.GetProperty("ids")[0];
            var documents = document.RootElement.GetProperty("documents")[0];
            var metadatas = document.RootElement.GetProperty("metadatas")[0];

            for (int i = 0; i < ids.GetArrayLength(); i++)
            {
                var doc = new Document<T>
                {
                    Id = ids[i].GetString() ?? string.Empty,
                    Content = documents[i].GetString() ?? string.Empty,
                    Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                        metadatas[i].GetRawText()) ?? new Dictionary<string, string>()
                };

                results.Add(doc);
            }

            return results;
        }

        public override async Task DeleteDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            await EnsureCollectionExistsAsync();

            var deleteRequest = new
            {
                ids = new[] { documentId }
            };

            var json = JsonSerializer.Serialize(deleteRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/collections/{_collectionId}/delete",
                content);

            response.EnsureSuccessStatusCode();
        }

        public override async Task<Document<T>?> GetDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            await EnsureCollectionExistsAsync();

            var getRequest = new
            {
                ids = new[] { documentId },
                include = new[] { "documents", "metadatas" }
            };

            var json = JsonSerializer.Serialize(getRequest);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/collections/{_collectionId}/get",
                content);

            if (!response.IsSuccessStatusCode)
                return null;

            var responseJson = await response.Content.ReadAsStringAsync();
            var document = JsonDocument.Parse(responseJson);

            var ids = document.RootElement.GetProperty("ids");
            if (ids.GetArrayLength() == 0)
                return null;

            return new Document<T>
            {
                Id = ids[0].GetString() ?? string.Empty,
                Content = document.RootElement.GetProperty("documents")[0].GetString() ?? string.Empty,
                Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                    document.RootElement.GetProperty("metadatas")[0].GetRawText()) ?? new Dictionary<string, string>()
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
