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
    /// Document store using Elasticsearch with vector search
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class ElasticsearchDocumentStore<T> : DocumentStoreBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly string _host;
        private readonly int _port;
        private readonly string _indexName;
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public ElasticsearchDocumentStore(string host = "localhost", int port = 9200, string indexName = "documents")
        {
            _host = host;
            _port = port;
            _indexName = indexName;
            _httpClient = new HttpClient();
            _baseUrl = $"http://{_host}:{_port}";
        }

        private async Task EnsureIndexExistsAsync()
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/{_indexName}");
            if (response.IsSuccessStatusCode)
                return;

            var indexMapping = new
            {
                mappings = new
                {
                    properties = new
                    {
                        id = new { type = "keyword" },
                        content = new { type = "text" },
                        embedding = new
                        {
                            type = "dense_vector",
                            dims = 768,
                            index = true,
                            similarity = "cosine"
                        },
                        metadata = new { type = "object", enabled = true }
                    }
                }
            };

            var json = JsonSerializer.Serialize(indexMapping);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            await _httpClient.PutAsync($"{_baseUrl}/{_indexName}", content);
        }

        public override async Task AddDocumentAsync(Document<T> document)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));

            await EnsureIndexExistsAsync();

            var doc = new
            {
                id = document.Id,
                content = document.Content,
                embedding = ConvertVectorToDoubleArray(document.Embedding),
                metadata = document.Metadata
            };

            var json = JsonSerializer.Serialize(doc);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/{_indexName}/_doc/{document.Id}",
                content);

            response.EnsureSuccessStatusCode();
        }

        public override async Task<List<Document<T>>> SearchAsync(Vector<T> queryEmbedding, int topK = 5)
        {
            if (queryEmbedding == null)
                throw new ArgumentNullException(nameof(queryEmbedding));

            await EnsureIndexExistsAsync();

            var searchQuery = new
            {
                size = topK,
                query = new
                {
                    script_score = new
                    {
                        query = new { match_all = new { } },
                        script = new
                        {
                            source = "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            @params = new
                            {
                                query_vector = ConvertVectorToDoubleArray(queryEmbedding)
                            }
                        }
                    }
                }
            };

            var json = JsonSerializer.Serialize(searchQuery);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/{_indexName}/_search",
                content);

            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            var document = JsonDocument.Parse(responseJson);
            var results = new List<Document<T>>();

            foreach (var hit in document.RootElement.GetProperty("hits").GetProperty("hits").EnumerateArray())
            {
                var source = hit.GetProperty("_source");
                var doc = new Document<T>
                {
                    Id = source.GetProperty("id").GetString() ?? string.Empty,
                    Content = source.GetProperty("content").GetString() ?? string.Empty,
                    Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                        source.GetProperty("metadata").GetRawText()) ?? new Dictionary<string, string>()
                };

                results.Add(doc);
            }

            return results;
        }

        public override async Task DeleteDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            var response = await _httpClient.DeleteAsync($"{_baseUrl}/{_indexName}/_doc/{documentId}");
            response.EnsureSuccessStatusCode();
        }

        public override async Task<Document<T>?> GetDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            var response = await _httpClient.GetAsync($"{_baseUrl}/{_indexName}/_doc/{documentId}");
            
            if (!response.IsSuccessStatusCode)
                return null;

            var json = await response.Content.ReadAsStringAsync();
            var element = JsonDocument.Parse(json).RootElement;
            
            if (!element.GetProperty("found").GetBoolean())
                return null;

            var source = element.GetProperty("_source");
            return new Document<T>
            {
                Id = source.GetProperty("id").GetString() ?? string.Empty,
                Content = source.GetProperty("content").GetString() ?? string.Empty,
                Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                    source.GetProperty("metadata").GetRawText()) ?? new Dictionary<string, string>()
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
