using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;

namespace AiDotNet;

/// <summary>
/// A lightweight in-memory <see cref="IDocumentStore{T}"/> that adapts any <see cref="IVectorIndex{T}"/>
/// (Flat / HNSW / IVF / LSH) into a document store usable by the RAG facade.
/// </summary>
/// <remarks>
/// <para>
/// This adapter lets <c>ConfigureVectorIndex</c> surface the in-memory vector indexes and a chosen
/// similarity metric as a retriever/store without depending on any particular concrete document store.
/// The chosen index (and the metric it was constructed with) performs the actual similarity search; this
/// wrapper only keeps the document payloads keyed by id so retrieval can return full <see cref="Document{T}"/>
/// instances with their relevance scores populated.
/// </para>
/// <para><b>For Beginners:</b> A vector index only knows about ids and vectors. A document store also needs
/// to hold the actual text/metadata. This class glues them together: it forwards vectors to the index for
/// fast similarity search, then looks the matching documents back up by id.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
internal sealed class VectorIndexDocumentStore<T> : IDocumentStore<T>
{
    private readonly IVectorIndex<T> _index;
    private readonly Dictionary<string, VectorDocument<T>> _documents = new();
    private int _vectorDimension;

    /// <summary>
    /// Initializes a new instance of the <see cref="VectorIndexDocumentStore{T}"/> class.
    /// </summary>
    /// <param name="index">The vector index that performs similarity search.</param>
    /// <param name="vectorDimension">The dimensionality of stored vectors (0 = inferred on first add).</param>
    public VectorIndexDocumentStore(IVectorIndex<T> index, int vectorDimension = 0)
    {
        _index = index ?? throw new ArgumentNullException(nameof(index));
        _vectorDimension = vectorDimension;
    }

    /// <inheritdoc />
    public int DocumentCount => _documents.Count;

    /// <inheritdoc />
    public int VectorDimension => _vectorDimension;

    /// <inheritdoc />
    public void Add(VectorDocument<T> vectorDocument)
    {
        if (vectorDocument == null) throw new ArgumentNullException(nameof(vectorDocument));
        var doc = vectorDocument.Document;
        if (doc == null || string.IsNullOrEmpty(doc.Id))
            throw new ArgumentException("VectorDocument must have a Document with a non-empty Id.", nameof(vectorDocument));

        var embedding = vectorDocument.Embedding;
        if (embedding == null || embedding.Length == 0)
            throw new ArgumentException("VectorDocument must have a non-empty Embedding.", nameof(vectorDocument));

        if (_vectorDimension == 0)
            _vectorDimension = embedding.Length;

        _documents[doc.Id] = vectorDocument;
        _index.Add(doc.Id, embedding);
    }

    /// <inheritdoc />
    public void AddBatch(IEnumerable<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments == null) throw new ArgumentNullException(nameof(vectorDocuments));
        foreach (var vd in vectorDocuments)
        {
            Add(vd);
        }
    }

    /// <inheritdoc />
    public IEnumerable<Document<T>> GetSimilar(Vector<T> queryVector, int topK)
        => GetSimilarWithFilters(queryVector, topK, new Dictionary<string, object>());

    /// <inheritdoc />
    public IEnumerable<Document<T>> GetSimilarWithFilters(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        if (queryVector == null) throw new ArgumentNullException(nameof(queryVector));
        if (topK <= 0) return Enumerable.Empty<Document<T>>();

        bool hasFilters = metadataFilters != null && metadataFilters.Count > 0;
        // When filtering, over-fetch from the index so enough candidates survive the filter.
        int searchK = hasFilters ? Math.Min(_documents.Count, Math.Max(topK * 4, topK)) : topK;
        if (searchK <= 0) return Enumerable.Empty<Document<T>>();

        var hits = _index.Search(queryVector, searchK);
        var results = new List<Document<T>>(topK);

        foreach (var (id, score) in hits)
        {
            if (!_documents.TryGetValue(id, out var vd)) continue;
            if (hasFilters && !MatchesFilters(vd.Document, metadataFilters!)) continue;

            var doc = vd.Document;
            doc.RelevanceScore = score;
            results.Add(doc);
            if (results.Count >= topK) break;
        }

        return results;
    }

    /// <inheritdoc />
    public Document<T>? GetById(string documentId)
        => _documents.TryGetValue(documentId, out var vd) ? vd.Document : null;

    /// <inheritdoc />
    public bool Remove(string documentId)
    {
        if (!_documents.Remove(documentId)) return false;
        _index.Remove(documentId);
        return true;
    }

    /// <inheritdoc />
    public void Clear()
    {
        _documents.Clear();
        _index.Clear();
    }

    /// <inheritdoc />
    public IEnumerable<Document<T>> GetAll() => _documents.Values.Select(vd => vd.Document);

    /// <inheritdoc />
    public Task AddAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Add(vectorDocument);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public Task AddBatchAsync(IEnumerable<VectorDocument<T>> vectorDocuments, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        AddBatch(vectorDocuments);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public Task<IEnumerable<Document<T>>> GetSimilarAsync(Vector<T> queryVector, int topK, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(GetSimilar(queryVector, topK));
    }

    /// <inheritdoc />
    public Task<IEnumerable<Document<T>>> GetSimilarWithFiltersAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(GetSimilarWithFilters(queryVector, topK, metadataFilters));
    }

    /// <inheritdoc />
    public Task<Document<T>?> GetByIdAsync(string documentId, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(GetById(documentId));
    }

    /// <inheritdoc />
    public Task<bool> RemoveAsync(string documentId, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(Remove(documentId));
    }

    /// <inheritdoc />
    public Task ClearAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Clear();
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public Task<IEnumerable<Document<T>>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(GetAll());
    }

    private static bool MatchesFilters(Document<T> document, Dictionary<string, object> filters)
    {
        foreach (var kvp in filters)
        {
            if (!document.Metadata.TryGetValue(kvp.Key, out var value)) return false;
            if (!Equals(value, kvp.Value)) return false;
        }

        return true;
    }
}
