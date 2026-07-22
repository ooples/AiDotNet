using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Filtering;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json.Linq;

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
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class ElasticsearchDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly HttpClient _httpClient;
    private readonly string _indexName;
    private int _vectorDimension;
    private int _documentCount;
    private readonly Dictionary<string, VectorDocument<T>> _cache;

    public override int DocumentCount => _documentCount;
    public override int VectorDimension => _vectorDimension;

    public ElasticsearchDocumentStore(string endpoint, string indexName, string apiKey, string username, string password, int vectorDimension = 1536)
    {
        if (string.IsNullOrWhiteSpace(endpoint))
            throw new ArgumentException("Endpoint cannot be empty", nameof(endpoint));
        if (string.IsNullOrWhiteSpace(indexName))
            throw new ArgumentException("Index name cannot be empty", nameof(indexName));
        if (vectorDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(vectorDimension), "Vector dimension must be positive");

        var hasApiKey = !string.IsNullOrWhiteSpace(apiKey);
        var hasBasicAuth = !string.IsNullOrWhiteSpace(username) && !string.IsNullOrWhiteSpace(password);
        if (!hasApiKey && !hasBasicAuth)
            throw new ArgumentException("Either apiKey or both username and password must be provided for authentication");

        _httpClient = new HttpClient { BaseAddress = new Uri(endpoint) };

        if (hasApiKey)
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"ApiKey {apiKey}");
        else
        {
            var auth = Convert.ToBase64String(Encoding.UTF8.GetBytes($"{username}:{password}"));
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"Basic {auth}");
        }

        _indexName = indexName.ToLowerInvariant();
        _vectorDimension = vectorDimension;
        _documentCount = 0;
        _cache = new Dictionary<string, VectorDocument<T>>();

        EnsureIndex();
    }

    private void EnsureIndex()
    {
        try
        {
            using var checkResponse = _httpClient.GetAsync($"/{_indexName}").GetAwaiter().GetResult();
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
                        embedding = new { type = "dense_vector", dims = _vectorDimension },
                        metadata = new { type = "object", enabled = true }
                    }
                }
            };

            using var content = new StringContent(
                Newtonsoft.Json.JsonConvert.SerializeObject(mapping),
                Encoding.UTF8,
                "application/json");

            using var response = _httpClient.PutAsync($"/{_indexName}", content).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException("Failed to ensure Elasticsearch index exists", ex);
        }
    }

    private void UpdateDocumentCount()
    {
        using var response = _httpClient.GetAsync($"/{_indexName}/_count").GetAwaiter().GetResult();
        if (response.IsSuccessStatusCode)
        {
            var responseContent = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            var result = JObject.Parse(responseContent);
            _documentCount = result["count"]?.Value<int>() ?? 0;
        }
    }

    protected override void AddCore(VectorDocument<T> vectorDocument)
        => AddCoreImplAsync(vectorDocument, CancellationToken.None).GetAwaiter().GetResult();

    /// <inheritdoc/>
    protected override Task AddCoreAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken)
        => AddCoreImplAsync(vectorDocument, cancellationToken);

    private async Task AddCoreImplAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken)
    {
        if (vectorDocument.Embedding.Length != _vectorDimension)
            throw new ArgumentException($"Document embedding dimension ({vectorDocument.Embedding.Length}) does not match the store's configured dimension ({_vectorDimension}).");

        var embedding = vectorDocument.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToArray();

        var doc = new
        {
            id = vectorDocument.Document.Id,
            content = vectorDocument.Document.Content,
            embedding,
            metadata = vectorDocument.Document.Metadata
        };

        using var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(doc),
            Encoding.UTF8,
            "application/json");

        using var response = await _httpClient.PutAsync($"/{_indexName}/_doc/{vectorDocument.Document.Id}", content, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        _cache[vectorDocument.Document.Id] = vectorDocument;
        _documentCount++;
    }

    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
        => AddBatchCoreImplAsync(vectorDocuments, CancellationToken.None).GetAwaiter().GetResult();

    /// <inheritdoc/>
    protected override Task AddBatchCoreAsync(IList<VectorDocument<T>> vectorDocuments, CancellationToken cancellationToken)
        => AddBatchCoreImplAsync(vectorDocuments, cancellationToken);

    private async Task AddBatchCoreImplAsync(IList<VectorDocument<T>> vectorDocuments, CancellationToken cancellationToken)
    {
        if (vectorDocuments.Count == 0) return;

        foreach (var vd in vectorDocuments)
        {
            if (vd.Embedding.Length != _vectorDimension)
                throw new ArgumentException($"Document embedding dimension ({vd.Embedding.Length}) does not match the store's configured dimension ({_vectorDimension}).");
        }

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

        using var content = new StringContent(bulkBody.ToString(), Encoding.UTF8, "application/x-ndjson");
        using var response = await _httpClient.PostAsync($"/{_indexName}/_bulk", content, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var responseContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var result = Newtonsoft.Json.Linq.JObject.Parse(responseContent);

        if (result["errors"]?.Value<bool>() == true)
        {
            // Check which items succeeded
            var items = result["items"];
            int addedCount = 0;
            if (items != null)
            {
                for (int i = 0; i < vectorDocuments.Count && i < items.Count(); i++)
                {
                    var item = items[i]?["index"];
                    var status = item?["status"]?.Value<int>() ?? 500;
                    if (status >= 200 && status < 300)
                    {
                        bool isNew = !_cache.ContainsKey(vectorDocuments[i].Document.Id);
                        _cache[vectorDocuments[i].Document.Id] = vectorDocuments[i];
                        if (isNew)
                            addedCount++;
                    }
                }
            }
            _documentCount += addedCount;
            throw new InvalidOperationException($"Bulk operation had partial failures");
        }

        int newDocCount = 0;
        foreach (var vd in vectorDocuments)
        {
            bool isNew = !_cache.ContainsKey(vd.Document.Id);
            _cache[vd.Document.Id] = vd;
            if (isNew)
                newDocCount++;
        }
        _documentCount += newDocCount;
    }

    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
        => GetSimilarCoreImplAsync(queryVector, topK, metadataFilters, CancellationToken.None).GetAwaiter().GetResult();

    /// <inheritdoc/>
    protected override Task<IEnumerable<Document<T>>> GetSimilarCoreAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken)
        => GetSimilarCoreImplAsync(queryVector, topK, metadataFilters, cancellationToken);

    private Task<IEnumerable<Document<T>>> GetSimilarCoreImplAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken)
        => SearchImplAsync(queryVector, topK, BuildDictQueryClause(metadataFilters), cancellationToken);

    /// <inheritdoc/>
    protected override IEnumerable<Document<T>> GetSimilarWithFilterCore(Vector<T> queryVector, MetadataFilter filter, int topK)
        => SearchImplAsync(queryVector, topK, ElasticsearchVectorFilterBuilder.Build(filter), CancellationToken.None).GetAwaiter().GetResult();

    /// <inheritdoc/>
    protected override Task<IEnumerable<Document<T>>> GetSimilarWithFilterCoreAsync(Vector<T> queryVector, MetadataFilter filter, int topK, CancellationToken cancellationToken)
        => SearchImplAsync(queryVector, topK, ElasticsearchVectorFilterBuilder.Build(filter), cancellationToken);

    private static object BuildDictQueryClause(Dictionary<string, object> metadataFilters)
    {
        if (metadataFilters != null && metadataFilters.Any())
        {
            var mustClauses = new List<object>();
            foreach (var filter in metadataFilters)
            {
                mustClauses.Add(new
                {
                    term = new Dictionary<string, object>
                    {
                        [$"metadata.{filter.Key}"] = filter.Value
                    }
                });
            }
            return new
            {
                @bool = new
                {
                    must = mustClauses
                }
            };
        }

        return new { match_all = new { } };
    }

    private async Task<IEnumerable<Document<T>>> SearchImplAsync(Vector<T> queryVector, int topK, object queryClause, CancellationToken cancellationToken)
    {
        var embedding = queryVector.ToArray().Select(v => Convert.ToDouble(v)).ToArray();

        var query = new
        {
            size = topK,
            query = new
            {
                script_score = new
                {
                    query = queryClause,
                    script = new
                    {
                        source = "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        @params = new { query_vector = embedding }
                    }
                }
            }
        };

        using var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(query),
            Encoding.UTF8,
            "application/json");

        using var response = await _httpClient.PostAsync($"/{_indexName}/_search", content, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Elasticsearch search failed with status {response.StatusCode}: {errorContent}");
        }

        var responseContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
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
            var metadataObj = source["metadata"]?.ToObject<Dictionary<string, object>>() ?? new Dictionary<string, object>();

            var score = Convert.ToDouble(hit["_score"]);

            var document = new Document<T>(id, docContent, metadataObj)
            {
                RelevanceScore = NumOps.FromDouble(score),
                HasRelevanceScore = true
            };

            results.Add(document);
        }

        return results;
    }

    protected override Document<T>? GetByIdCore(string documentId)
        => GetByIdCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

    /// <inheritdoc/>
    protected override Task<Document<T>?> GetByIdCoreAsync(string documentId, CancellationToken cancellationToken)
        => GetByIdCoreImplAsync(documentId, cancellationToken);

    private async Task<Document<T>?> GetByIdCoreImplAsync(string documentId, CancellationToken cancellationToken)
    {
        if (_cache.TryGetValue(documentId, out var vectorDoc))
            return vectorDoc.Document;

        using var response = await _httpClient.GetAsync($"/{_indexName}/_doc/{documentId}", cancellationToken).ConfigureAwait(false);

        if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            return null;

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Failed to retrieve document '{documentId}' from Elasticsearch. Status: {response.StatusCode}, Error: {errorContent}");
        }

        var responseContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var result = JObject.Parse(responseContent);

        if (result["found"]?.Value<bool>() != true)
            return null;

        var source = result["_source"];
        var id = source?["id"]?.ToString();
        var content = source?["content"]?.ToString();
        var metadataObj = source?["metadata"]?.ToObject<Dictionary<string, object>>() ?? new Dictionary<string, object>();

        if (id == null || content == null)
            return null;

        return new Document<T>(id, content, metadataObj);
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
        => RemoveCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

    /// <inheritdoc/>
    protected override Task<bool> RemoveCoreAsync(string documentId, CancellationToken cancellationToken)
        => RemoveCoreImplAsync(documentId, cancellationToken);

    private async Task<bool> RemoveCoreImplAsync(string documentId, CancellationToken cancellationToken)
    {
        using var response = await _httpClient.DeleteAsync($"/{_indexName}/_doc/{documentId}", cancellationToken).ConfigureAwait(false);
        if (response.IsSuccessStatusCode && _documentCount > 0)
        {
            _cache.Remove(documentId);
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
    /// Uses the Elasticsearch scroll API to efficiently retrieve all documents from the index
    /// in batches, avoiding memory issues with large result sets. The scroll API maintains a
    /// search context and streams results in manageable batches.
    /// </para>
    /// <para><b>For Beginners:</b> Retrieves all documents using Elasticsearch's scroll API.
    ///
    /// The scroll API is the recommended way to retrieve large numbers of documents because:
    /// - Processes results in batches (default: 1000 documents per batch)
    /// - Maintains a search context between requests
    /// - Much more memory-efficient than large "from/size" pagination
    /// - Automatically expires after timeout (default: 1 minute)
    ///
    /// How it works:
    /// 1. Initial scroll request creates search context, returns first batch
    /// 2. Subsequent requests using scroll_id return next batches
    /// 3. Continues until all documents retrieved
    /// 4. Context automatically cleaned up after timeout
    ///
    /// Example:
    /// <code>
    /// // Efficiently retrieve all documents
    /// var allDocs = store.GetAll().ToList();
    /// // Result is available in the returned value
    ///
    /// // Export to JSON
    /// var json = JsonConvert.SerializeObject(allDocs);
    /// File.WriteAllText("elasticsearch_export.json", json);
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetAllCore()
        => GetAllCoreImplAsync(CancellationToken.None).GetAwaiter().GetResult();

    /// <inheritdoc/>
    protected override Task<IEnumerable<Document<T>>> GetAllCoreAsync(CancellationToken cancellationToken)
        => GetAllCoreImplAsync(cancellationToken);

    private async Task<IEnumerable<Document<T>>> GetAllCoreImplAsync(CancellationToken cancellationToken)
    {
        const string scrollTimeout = "1m";
        const int batchSize = 1000;

        var documents = new List<Document<T>>();

        // Initial scroll request
        var searchRequest = new
        {
            size = batchSize,
            query = new { match_all = new { } }
        };

        using var searchResponse = await _httpClient.PostAsync(
            $"{_indexName}/_search?scroll={scrollTimeout}",
            new StringContent(
                Newtonsoft.Json.JsonConvert.SerializeObject(searchRequest),
                Encoding.UTF8,
                "application/json"
            ),
            cancellationToken
        ).ConfigureAwait(false);

        if (!searchResponse.IsSuccessStatusCode)
        {
            var error = await searchResponse.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new InvalidOperationException($"Elasticsearch scroll initialization failed: {error}");
        }

        var initialResult = JObject.Parse(await searchResponse.Content.ReadAsStringAsync().ConfigureAwait(false));
        var scrollId = initialResult["_scroll_id"]?.ToString();

        if (string.IsNullOrEmpty(scrollId))
            return documents;

        // Process initial batch
        var hits = initialResult["hits"]?["hits"] as JArray;
        if (hits != null)
        {
            foreach (var hit in hits)
            {
                var doc = ParseElasticsearchHit(hit as JObject);
                if (doc != null)
                    documents.Add(doc);
            }
        }

        // Continue scrolling until no more results
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var scrollRequest = new { scroll = scrollTimeout, scroll_id = scrollId };

            using var scrollResponse = await _httpClient.PostAsync(
                "_search/scroll",
                new StringContent(
                    Newtonsoft.Json.JsonConvert.SerializeObject(scrollRequest),
                    Encoding.UTF8,
                    "application/json"
                ),
                cancellationToken
            ).ConfigureAwait(false);

            if (!scrollResponse.IsSuccessStatusCode)
                break;

            var scrollResult = JObject.Parse(await scrollResponse.Content.ReadAsStringAsync().ConfigureAwait(false));
            hits = scrollResult["hits"]?["hits"] as JArray;

            if (hits == null || hits.Count == 0)
                break;

            foreach (var hit in hits)
            {
                var doc = ParseElasticsearchHit(hit as JObject);
                if (doc != null)
                    documents.Add(doc);
            }
        }

        // Clean up scroll context
        try
        {
            var deleteScrollRequest = new { scroll_id = new[] { scrollId } };
            var deleteContent = new StringContent(
                Newtonsoft.Json.JsonConvert.SerializeObject(deleteScrollRequest),
                Encoding.UTF8,
                "application/json"
            );

            // DELETE with body requires using HttpRequestMessage
            var request = new HttpRequestMessage(HttpMethod.Delete, "_search/scroll")
            {
                Content = deleteContent
            };

            using var deleteResponse = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
            deleteResponse.EnsureSuccessStatusCode();
        }
        catch
        {
            // Scroll context will expire automatically
        }

        return documents;
    }

    private Document<T>? ParseElasticsearchHit(JObject? hit)
    {
        if (hit == null)
            return null;

        try
        {
            var id = hit["_id"]?.ToString();
            var source = hit["_source"] as JObject;

            if (string.IsNullOrEmpty(id) || source == null)
                return null;

            var content = source["content"]?.ToString() ?? string.Empty;
            var metadata = source["metadata"]?.ToObject<Dictionary<string, object>>() ?? new Dictionary<string, object>();

            return new Document<T>(id ?? Guid.NewGuid().ToString(), content, metadata);
        }
        catch
        {
            return null;
        }
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
    /// // Result is available in the returned value // 0
    /// </code>
    /// </para>
    /// </remarks>
    public override void Clear()
        => ClearImplAsync(CancellationToken.None).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public override Task ClearAsync(CancellationToken cancellationToken = default)
        => ClearImplAsync(cancellationToken);

    private async Task ClearImplAsync(CancellationToken cancellationToken)
    {
        try
        {
            using var response = await _httpClient.DeleteAsync($"/{_indexName}", cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            _cache.Clear();
            _documentCount = 0;
            EnsureIndex();
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException("Failed to clear Elasticsearch index", ex);
        }
    }
}
