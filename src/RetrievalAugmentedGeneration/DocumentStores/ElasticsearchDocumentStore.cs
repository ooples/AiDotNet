using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Elasticsearch-based document store providing hybrid search capabilities (BM25 + dense vectors).
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// Elasticsearch combines traditional full-text search (BM25) with vector similarity search,
/// making it ideal for hybrid retrieval scenarios where both keyword matching and semantic
/// similarity are important.
/// </remarks>
public class ElasticsearchDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly HttpClient _httpClient;
    private readonly string _indexName;
    private int _vectorDimension;
    private int _documentCount;
    private readonly Dictionary<string, VectorDocument<T>> _cache;

    public override int DocumentCount => _documentCount;
    public override int VectorDimension => _vectorDimension;

    public ElasticsearchDocumentStore(string endpoint, string indexName, string apiKey, string username, string password)
    {
        if (string.IsNullOrWhiteSpace(endpoint))
            throw new ArgumentException("Endpoint cannot be empty", nameof(endpoint));
        if (string.IsNullOrWhiteSpace(indexName))
            throw new ArgumentException("Index name cannot be empty", nameof(indexName));

        _httpClient = new HttpClient { BaseAddress = new Uri(endpoint) };
        
        if (!string.IsNullOrWhiteSpace(apiKey))
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"ApiKey {apiKey}");
        else if (!string.IsNullOrWhiteSpace(username) && !string.IsNullOrWhiteSpace(password))
        {
            var auth = Convert.ToBase64String(Encoding.UTF8.GetBytes($"{username}:{password}"));
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"Basic {auth}");
        }

        _indexName = indexName.ToLowerInvariant();
        _vectorDimension = 0;
        _documentCount = 0;
        _cache = new Dictionary<string, VectorDocument<T>>();

        EnsureIndex();
    }

    private void EnsureIndex()
    {
        var checkResponse = _httpClient.GetAsync($"/{_indexName}").Result;
        if (checkResponse.IsSuccessStatusCode) 
        {
            UpdateDocumentCount();
            return;
        }

        var mapping = new
        {
            mappings = new
            {
                properties = new
                {
                    id = new { type = "keyword" },
                    content = new { type = "text" },
                    embedding = new { type = "dense_vector", dims = 1536 },
                    metadata = new { type = "object", enabled = true }
                }
            }
        };

        var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(mapping),
            Encoding.UTF8,
            "application/json");

        _httpClient.PutAsync($"/{_indexName}", content).Wait();
    }

    private void UpdateDocumentCount()
    {
        var response = _httpClient.GetAsync($"/{_indexName}/_count").Result;
        if (response.IsSuccessStatusCode)
        {
            var responseContent = response.Content.ReadAsStringAsync().Result;
            var result = JObject.Parse(responseContent);
            _documentCount = result["count"]?.Value<int>() ?? 0;
        }
    }

    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        if (_vectorDimension == 0)
            _vectorDimension = vectorDocument.Embedding.Length;

        _cache[vectorDocument.Document.Id] = vectorDocument;

        var embedding = vectorDocument.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
        
        var doc = new
        {
            id = vectorDocument.Document.Id,
            content = vectorDocument.Document.Content,
            embedding,
            metadata = vectorDocument.Document.Metadata
        };

        var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(doc),
            Encoding.UTF8,
            "application/json");

        var response = _httpClient.PutAsync($"/{_indexName}/_doc/{vectorDocument.Document.Id}", content).Result;
        if (response.IsSuccessStatusCode)
            _documentCount++;
    }

    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0) return;
        
        if (_vectorDimension == 0)
            _vectorDimension = vectorDocuments[0].Embedding.Length;

        foreach (var vd in vectorDocuments)
            _cache[vd.Document.Id] = vd;

        var bulkBody = new StringBuilder();
        foreach (var vd in vectorDocuments)
        {
            var indexAction = new { index = new { _index = _indexName, _id = vd.Document.Id } };
            bulkBody.AppendLine(Newtonsoft.Json.JsonConvert.SerializeObject(indexAction));

            var embedding = vd.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
            var doc = new
            {
                id = vd.Document.Id,
                content = vd.Document.Content,
                embedding,
                metadata = vd.Document.Metadata
            };
            bulkBody.AppendLine(Newtonsoft.Json.JsonConvert.SerializeObject(doc));
        }

        var content = new StringContent(bulkBody.ToString(), Encoding.UTF8, "application/x-ndjson");
        var response = _httpClient.PostAsync($"/{_indexName}/_bulk", content).Result;
        if (response.IsSuccessStatusCode)
            _documentCount += vectorDocuments.Count;
    }

    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        var embedding = queryVector.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
        
        var query = new
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
                        @params = new { query_vector = embedding }
                    }
                }
            }
        };

        var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(query),
            Encoding.UTF8,
            "application/json");

        var response = _httpClient.PostAsync($"/{_indexName}/_search", content).Result;
        var responseContent = response.Content.ReadAsStringAsync().Result;
        var result = JObject.Parse(responseContent);

        var results = new List<Document<T>>();
        var hits = result["hits"]?["hits"];
        if (hits == null) return results;

        foreach (var hit in hits)
        {
            var source = hit["_source"];
            if (source == null) continue;

            var id = source["id"]?.ToString() ?? string.Empty;
            var docContent = source["content"]?.ToString() ?? string.Empty;
            var metadataObj = source["metadata"]?.ToObject<Dictionary<string, string>>() ?? new Dictionary<string, string>();
            var metadata = new Dictionary<string, object>();
            foreach (var kvp in metadataObj)
                metadata[kvp.Key] = kvp.Value;

            var score = Convert.ToDouble(hit["_score"]);

            var document = new Document<T>(id, docContent, metadata);
            document.RelevanceScore = NumOps.FromDouble(score);

            results.Add(document);
        }

        return results;
    }

    protected override Document<T>? GetByIdCore(string documentId)
    {
        if (_cache.TryGetValue(documentId, out var vectorDoc))
            return vectorDoc.Document;

        return null;
    }

    /// <summary>
    /// Core logic for removing a document from the Elasticsearch index.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>True if the document was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Removes the document from both the cache and the Elasticsearch index via DELETE API.
    /// If successful, decrements the document count.
    /// </para>
    /// <para><b>For Beginners:</b> Deletes a document from Elasticsearch.
    /// 
    /// In Elasticsearch terms, this is like:
    /// <code>
    /// DELETE /index/_doc/document_id
    /// </code>
    /// 
    /// Example:
    /// <code>
    /// if (store.Remove("doc-123"))
    ///     Console.WriteLine("Document removed from Elasticsearch");
    /// </code>
    /// </para>
    /// </remarks>
    protected override bool RemoveCore(string documentId)
    {
        _cache.Remove(documentId);

        var response = _httpClient.DeleteAsync($"/{_indexName}/_doc/{documentId}").Result;
        if (response.IsSuccessStatusCode && _documentCount > 0)
        {
            _documentCount--;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Core logic for retrieving all documents from the Elasticsearch index.
    /// </summary>
    /// <returns>An enumerable of all documents without their vector embeddings.</returns>
    /// <remarks>
    /// <para>
    /// Returns all documents from the cache (in-memory representation) of the Elasticsearch index.
    /// In a real Elasticsearch deployment with large indices, use the scroll API for efficient
    /// retrieval of all documents without loading everything into memory at once.
    /// </para>
    /// <para><b>For Beginners:</b> Gets every document from the Elasticsearch index.
    /// 
    /// Use cases:
    /// - Export index contents for backup
    /// - Migrate to a different index or cluster
    /// - Bulk reindexing operations
    /// - Debugging to see all indexed documents
    /// 
    /// Warning: For large indices (> 10K documents), this can use significant memory.
    /// In production Elasticsearch, use the scroll API with pagination:
    /// - POST /index/_search?scroll=1m with size parameter
    /// - Iterate through scroll_id responses
    /// - Clear scroll when done
    /// 
    /// Example:
    /// <code>
    /// // Get all documents
    /// var allDocs = store.GetAll().ToList();
    /// Console.WriteLine($"Total documents in {_indexName}: {allDocs.Count}");
    /// 
    /// // Export to JSON
    /// var json = JsonConvert.SerializeObject(allDocs);
    /// File.WriteAllText($"elasticsearch_{_indexName}_export.json", json);
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetAllCore()
    {
        return _cache.Values.Select(vd => vd.Document).ToList();
    }

    /// <summary>
    /// Removes all documents from the Elasticsearch index and recreates it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Clears the cache, deletes the Elasticsearch index, resets counters, and recreates the index
    /// with the same mapping. The index name remains unchanged and is ready to accept new documents.
    /// </para>
    /// <para><b>For Beginners:</b> Completely empties the Elasticsearch index and recreates it.
    /// 
    /// After calling Clear():
    /// - Index is deleted from Elasticsearch
    /// - Cache is cleared
    /// - Document count resets to 0
    /// - Vector dimension resets to 0
    /// - Index is recreated with fresh mapping
    /// - Ready for new documents
    /// 
    /// Use with caution - this cannot be undone!
    /// 
    /// In Elasticsearch terms, this is like:
    /// <code>
    /// DELETE /index_name
    /// PUT /index_name with mappings
    /// </code>
    /// 
    /// Example:
    /// <code>
    /// store.Clear();
    /// Console.WriteLine($"Documents in index: {store.DocumentCount}"); // 0
    /// </code>
    /// </para>
    /// </remarks>
    public override void Clear()
    {
        _cache.Clear();
        _httpClient.DeleteAsync($"/{_indexName}").Wait();
        _documentCount = 0;
        _vectorDimension = 0;
        EnsureIndex();
    }
}
