using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// ChromaDB-based document store designed for simplicity and developer experience.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// ChromaDB is an open-source vector database that emphasizes ease of use while maintaining
/// high performance for similarity search operations.
/// </remarks>
public class ChromaDBDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly HttpClient _httpClient;
    private readonly string _collectionName;
    private int _vectorDimension;
    private int _documentCount;
    private readonly Dictionary<string, VectorDocument<T>> _cache;

    public override int DocumentCount => _documentCount;
    public override int VectorDimension => _vectorDimension;

    public ChromaDBDocumentStore(string endpoint, string collectionName, string apiKey)
    {
        if (string.IsNullOrWhiteSpace(endpoint))
            throw new ArgumentException("Endpoint cannot be empty", nameof(endpoint));
        if (string.IsNullOrWhiteSpace(collectionName))
            throw new ArgumentException("Collection name cannot be empty", nameof(collectionName));

        _httpClient = new HttpClient { BaseAddress = new Uri(endpoint) };
        if (!string.IsNullOrWhiteSpace(apiKey))
            _httpClient.DefaultRequestHeaders.Add("X-Chroma-Token", apiKey);

        _collectionName = collectionName;
        _vectorDimension = 0;
        _documentCount = 0;
        _cache = new Dictionary<string, VectorDocument<T>>();

        EnsureCollection();
    }

    private void EnsureCollection()
    {
        try
        {
            var payload = new { name = _collectionName };
            using var content = new StringContent(
                Newtonsoft.Json.JsonConvert.SerializeObject(payload),
                Encoding.UTF8,
                "application/json");

            using var response = _httpClient.PostAsync("/api/v1/collections", content).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
        }
        catch (HttpRequestException)
        {
            // Collection may already exist, which is acceptable
        }
    }

    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        if (_vectorDimension == 0)
            _vectorDimension = vectorDocument.Embedding.Length;
        else if (vectorDocument.Embedding.Length != _vectorDimension)
            throw new ArgumentException($"Vector dimension mismatch. Expected {_vectorDimension}, got {vectorDocument.Embedding.Length}", nameof(vectorDocument));

        var embedding = vectorDocument.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToList();

        var payload = new
        {
            ids = new[] { vectorDocument.Document.Id },
            embeddings = new[] { embedding },
            documents = new[] { vectorDocument.Document.Content },
            metadatas = new[] { vectorDocument.Document.Metadata }
        };

        using var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(payload),
            Encoding.UTF8,
            "application/json");

        using var response = _httpClient.PostAsync($"/api/v1/collections/{_collectionName}/add", content).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();

        _cache[vectorDocument.Document.Id] = vectorDocument;
        _documentCount++;
    }

    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0) return;

        if (_vectorDimension == 0)
            _vectorDimension = vectorDocuments[0].Embedding.Length;

        foreach (var vd in vectorDocuments)
        {
            if (vd.Embedding.Length != _vectorDimension)
                throw new ArgumentException($"Vector dimension mismatch. Expected {_vectorDimension}, got {vd.Embedding.Length}");
        }

        var ids = vectorDocuments.Select(vd => vd.Document.Id).ToList();
        var embeddings = vectorDocuments.Select(vd =>
            vd.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToList()).ToList();
        var documents = vectorDocuments.Select(vd => vd.Document.Content).ToList();
        var metadatas = vectorDocuments.Select(vd => vd.Document.Metadata).ToList();

        var payload = new { ids, embeddings, documents, metadatas };
        using var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(payload),
            Encoding.UTF8,
            "application/json");

        using var response = _httpClient.PostAsync($"/api/v1/collections/{_collectionName}/add", content).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();

        foreach (var vd in vectorDocuments)
            _cache[vd.Document.Id] = vd;
        _documentCount += vectorDocuments.Count;
    }

    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        var embedding = queryVector.ToArray().Select(v => Convert.ToDouble(v)).ToList();

        var payload = new
        {
            query_embeddings = new[] { embedding },
            n_results = topK
        };

        using var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(payload),
            Encoding.UTF8,
            "application/json");

        using var response = _httpClient.PostAsync($"/api/v1/collections/{_collectionName}/query", content).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();
        var responseContent = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        var result = JObject.Parse(responseContent);

        var results = new List<Document<T>>();
        var ids = result["ids"]?[0];
        var documents = result["documents"]?[0];
        var metadatas = result["metadatas"]?[0];
        var distances = result["distances"]?[0];

        if (ids == null || documents == null) return results;

        for (int i = 0; i < ids.Count(); i++)
        {
            var idToken = ids[i];
            var docToken = documents[i];
            if (idToken == null || docToken == null) continue;

            var id = idToken.ToString();
            var doc = docToken.ToString();
            var metadataObj = metadatas?[i]?.ToObject<Dictionary<string, object>>() ?? new Dictionary<string, object>();

            var distance = distances != null ? Convert.ToDouble(distances[i]) : 0.0;

            var document = new Document<T>(id, doc, metadataObj)
            {
                RelevanceScore = NumOps.FromDouble(1.0 / (1.0 + distance)),
                HasRelevanceScore = true
            };

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
    /// Core logic for removing a document from the ChromaDB collection.
    /// </summary>
    /// <param name="documentId">The validated document ID.</param>
    /// <returns>True if the document was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Removes the document from both the cache and the ChromaDB collection via API call.
    /// If successful, decrements the document count.
    /// </para>
    /// <para><b>For Beginners:</b> Deletes a document from ChromaDB.
    /// 
    /// Example:
    /// <code>
    /// if (store.Remove("doc-123"))
    ///     Console.WriteLine("Document removed from ChromaDB");
    /// </code>
    /// </para>
    /// </remarks>
    protected override bool RemoveCore(string documentId)
    {
        var payload = new { ids = new[] { documentId } };
        using var content = new StringContent(
            Newtonsoft.Json.JsonConvert.SerializeObject(payload),
            Encoding.UTF8,
            "application/json");

        using var response = _httpClient.PostAsync($"/api/v1/collections/{_collectionName}/delete", content).GetAwaiter().GetResult();
        if (response.IsSuccessStatusCode && _documentCount > 0)
        {
            _cache.Remove(documentId);
            _documentCount--;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Core logic for retrieving all documents from the ChromaDB collection.
    /// </summary>
    /// <returns>An enumerable of all documents without their vector embeddings.</returns>
    /// <remarks>
    /// <para>
    /// Returns all documents from the cache (in-memory representation) of the ChromaDB collection.
    /// In a real ChromaDB deployment, this would query the collection's entire dataset.
    /// Vector embeddings are not included in the results.
    /// </para>
    /// <para><b>For Beginners:</b> Gets every document from the ChromaDB collection.
    /// 
    /// Use cases:
    /// - Export collection contents for backup
    /// - Migrate to a different ChromaDB collection or database
    /// - Bulk processing or reindexing
    /// - Debugging to see all stored documents
    /// 
    /// Warning: For large collections (> 10K documents), this can use significant memory.
    /// In production ChromaDB, consider using pagination with limit/offset parameters.
    /// 
    /// Example:
    /// <code>
    /// // Get all documents
    /// var allDocs = store.GetAll().ToList();
    /// Console.WriteLine($"Total documents in {_collectionName}: {allDocs.Count}");
    /// 
    /// // Export to JSON
    /// var json = JsonConvert.SerializeObject(allDocs);
    /// File.WriteAllText($"chroma_{_collectionName}_export.json", json);
    /// </code>
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> GetAllCore()
    {
        return _cache.Values.Select(vd => vd.Document).ToList();
    }

    /// <summary>
    /// Removes all documents from the ChromaDB collection and recreates it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Clears the cache, deletes the ChromaDB collection, resets counters, and recreates an empty collection.
    /// The collection name remains unchanged and is ready to accept new documents.
    /// </para>
    /// <para><b>For Beginners:</b> Completely empties the ChromaDB collection.
    /// 
    /// After calling Clear():
    /// - All documents are removed from ChromaDB
    /// - Cache is cleared
    /// - Document count resets to 0
    /// - Vector dimension resets to 0
    /// - Collection is recreated (empty)
    /// - Ready for new documents
    /// 
    /// Use with caution - this cannot be undone!
    /// 
    /// Example:
    /// <code>
    /// store.Clear();
    /// Console.WriteLine($"Documents in collection: {store.DocumentCount}"); // 0
    /// </code>
    /// </para>
    /// </remarks>
    public override void Clear()
    {
        try
        {
            using var response = _httpClient.DeleteAsync($"/api/v1/collections/{_collectionName}").GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();

            _cache.Clear();
            _documentCount = 0;
            _vectorDimension = 0;
            EnsureCollection();
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException("Failed to clear ChromaDB collection. Collection may not exist or network error occurred.", ex);
        }
    }
}
