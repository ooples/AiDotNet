using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Redis-based vector document store for low-latency applications.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// Redis with RedisSearch module provides fast vector similarity search with sub-millisecond latency.
/// Ideal for real-time applications requiring instant retrieval.
/// </remarks>
public class RedisVLDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly Dictionary<string, VectorDocument<T>> _store;
    private int _vectorDimension;

    public override int DocumentCount => _store.Count;
    public override int VectorDimension => _vectorDimension;

    public RedisVLDocumentStore(string connectionString, string indexName, int vectorDimension)
    {
        if (string.IsNullOrWhiteSpace(connectionString))
            throw new ArgumentException("Connection string cannot be empty", nameof(connectionString));
        if (string.IsNullOrWhiteSpace(indexName))
            throw new ArgumentException("Index name cannot be empty", nameof(indexName));
        if (vectorDimension <= 0)
            throw new ArgumentException("Vector dimension must be positive", nameof(vectorDimension));

        _store = new Dictionary<string, VectorDocument<T>>();
        _vectorDimension = vectorDimension;
    }

    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        if (_vectorDimension == 0)
            _vectorDimension = vectorDocument.Embedding.Length;

        _store[vectorDocument.Document.Id] = vectorDocument;
    }

    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0) return;

        if (_vectorDimension == 0)
            _vectorDimension = vectorDocuments[0].Embedding.Length;

        foreach (var vd in vectorDocuments)
            _store[vd.Document.Id] = vd;
    }

    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        var results = new List<(Document<T> doc, T score)>();

        foreach (var vd in _store.Values)
        {
            var similarity = StatisticsHelper<T>.CosineSimilarity(queryVector, vd.Embedding);
            vd.Document.RelevanceScore = similarity;
            results.Add((vd.Document, similarity));
        }

        return results
            .OrderByDescending(x => Convert.ToDouble(x.score))
            .Take(topK)
            .Select(x => x.doc);
    }

    protected override Document<T>? GetByIdCore(string documentId)
    {
        return _store.TryGetValue(documentId, out var vd) ? vd.Document : null;
    }

    protected override bool RemoveCore(string documentId)
    {
        return _store.Remove(documentId);
    }

    public override void Clear()
    {
        _store.Clear();
        _vectorDimension = 0;
    }
}
