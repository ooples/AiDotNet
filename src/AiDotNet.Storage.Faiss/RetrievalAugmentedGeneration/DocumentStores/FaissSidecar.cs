using System;
using System.Collections.Generic;
using System.Linq;

using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// One record in the FAISS sidecar: everything FAISS itself does NOT store.
/// </summary>
/// <remarks>
/// <para>
/// FAISS stores only raw vectors keyed by an <see cref="System.Int64"/> id. It has no notion of
/// a document's string id, its text content, or its metadata, and lossy index types
/// (e.g. product quantization) cannot reconstruct the original vector. The sidecar keeps
/// all of that alongside the FAISS index so the store can map search hits back to real
/// documents, apply metadata filters, rebuild the index on removal (for index types that
/// do not support in-place deletion), and round-trip through persistence.
/// </para>
/// </remarks>
public sealed class FaissSidecarEntry
{
    /// <summary>The int64 id this document was assigned inside the FAISS index.</summary>
    public long FaissId { get; set; }

    /// <summary>The caller-supplied unique document id.</summary>
    public string DocumentId { get; set; } = string.Empty;

    /// <summary>The document's text content.</summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>The document's metadata (used for over-fetch filtering).</summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// The (already metric-adjusted, e.g. L2-normalized for cosine) embedding as it was
    /// added to FAISS. Retained so the index can be rebuilt from scratch when a document
    /// is removed from an index type that does not implement in-place deletion (HNSW).
    /// </summary>
    public float[] Embedding { get; set; } = Array.Empty<float>();
}

/// <summary>
/// Managed, native-free sidecar for <c>FaissDocumentStore</c>: owns the
/// string-id &lt;-&gt; int64-id mapping, the per-document payload, and JSON persistence.
/// </summary>
/// <remarks>
/// <para>
/// This type deliberately performs no FAISS/native calls, so its id-assignment,
/// upsert/remove, and serialization behavior are fully unit-testable without the
/// native library being loadable.
/// </para>
/// </remarks>
public sealed class FaissSidecar
{
    private readonly Dictionary<string, long> _docIdToFaissId;
    private readonly Dictionary<long, FaissSidecarEntry> _entries;

    /// <summary>
    /// The next int64 id to hand out. Monotonically increasing and never reused, even
    /// after removals, so a stale FAISS hit for a deleted id can never collide with a
    /// live document.
    /// </summary>
    public long NextId { get; private set; }

    /// <summary>Number of live documents tracked by the sidecar.</summary>
    public int Count => _entries.Count;

    /// <summary>All live entries (unordered).</summary>
    public IReadOnlyCollection<FaissSidecarEntry> Entries => _entries.Values;

    /// <summary>Creates an empty sidecar.</summary>
    public FaissSidecar()
    {
        _docIdToFaissId = new Dictionary<string, long>();
        _entries = new Dictionary<long, FaissSidecarEntry>();
        NextId = 0;
    }

    private FaissSidecar(Dictionary<string, long> docIdToFaissId, Dictionary<long, FaissSidecarEntry> entries, long nextId)
    {
        _docIdToFaissId = docIdToFaissId;
        _entries = entries;
        NextId = nextId;
    }

    /// <summary>
    /// Inserts a document, or replaces an existing one with the same <paramref name="documentId"/>.
    /// A brand-new int64 id is always allocated; when replacing, the previous id is returned via
    /// <paramref name="replacedFaissId"/> so the caller can evict it from the FAISS index.
    /// </summary>
    /// <returns>The newly allocated FAISS id for this document.</returns>
    public long Upsert(string documentId, string content, Dictionary<string, object>? metadata, float[] embedding, out long? replacedFaissId)
    {
        if (string.IsNullOrWhiteSpace(documentId))
            throw new ArgumentException("Document id cannot be null or empty", nameof(documentId));
        if (embedding == null)
            throw new ArgumentNullException(nameof(embedding));

        replacedFaissId = null;
        if (_docIdToFaissId.TryGetValue(documentId, out var existing))
        {
            replacedFaissId = existing;
            _entries.Remove(existing);
        }

        var id = NextId++;
        _entries[id] = new FaissSidecarEntry
        {
            FaissId = id,
            DocumentId = documentId,
            Content = content ?? string.Empty,
            Metadata = metadata ?? new Dictionary<string, object>(),
            Embedding = embedding
        };
        _docIdToFaissId[documentId] = id;
        return id;
    }

    /// <summary>Attempts to look up an entry by its FAISS int64 id.</summary>
    public bool TryGetByFaissId(long faissId, out FaissSidecarEntry entry) => _entries.TryGetValue(faissId, out entry!);

    /// <summary>Attempts to look up an entry by its caller-supplied document id.</summary>
    public bool TryGetByDocumentId(string documentId, out FaissSidecarEntry entry)
    {
        entry = null!;
        return documentId != null
            && _docIdToFaissId.TryGetValue(documentId, out var faissId)
            && _entries.TryGetValue(faissId, out entry!);
    }

    /// <summary>
    /// Removes the entry for <paramref name="documentId"/>. Returns true and the freed
    /// FAISS id when the document existed; false otherwise.
    /// </summary>
    public bool RemoveByDocumentId(string documentId, out long removedFaissId)
    {
        removedFaissId = -1;
        if (documentId == null || !_docIdToFaissId.TryGetValue(documentId, out var faissId))
            return false;

        removedFaissId = faissId;
        _docIdToFaissId.Remove(documentId);
        _entries.Remove(faissId);
        return true;
    }

    /// <summary>Removes all entries and resets the id counter.</summary>
    public void Clear()
    {
        _docIdToFaissId.Clear();
        _entries.Clear();
        NextId = 0;
    }

    /// <summary>Serializes the sidecar to JSON.</summary>
    public string ToJson() => JsonConvert.SerializeObject(new PersistShape
    {
        NextId = NextId,
        Entries = _entries.Values.ToList()
    });

    /// <summary>Rehydrates a sidecar from JSON produced by <see cref="ToJson"/>.</summary>
    public static FaissSidecar FromJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return new FaissSidecar();

        var shape = JsonConvert.DeserializeObject<PersistShape>(json) ?? new PersistShape();
        var byFaissId = new Dictionary<long, FaissSidecarEntry>();
        var byDocId = new Dictionary<string, long>();
        long maxId = -1;
        foreach (var e in shape.Entries)
        {
            e.Metadata ??= new Dictionary<string, object>();
            e.Embedding ??= Array.Empty<float>();
            byFaissId[e.FaissId] = e;
            byDocId[e.DocumentId] = e.FaissId;
            if (e.FaissId > maxId) maxId = e.FaissId;
        }

        // NextId must exceed every persisted id so re-added documents never reuse an id.
        var nextId = Math.Max(shape.NextId, maxId + 1);
        return new FaissSidecar(byDocId, byFaissId, nextId);
    }

    private sealed class PersistShape
    {
        public long NextId { get; set; }
        public List<FaissSidecarEntry> Entries { get; set; } = new();
    }
}

/// <summary>
/// Pure, native-free helpers for the over-fetch-then-filter retrieval plan used by
/// <c>FaissDocumentStore</c>. Extracted so the fetch-count math and the
/// filter/top-k selection can be unit-tested without touching FAISS.
/// </summary>
public static class FaissRetrievalPlanner
{
    /// <summary>
    /// Computes how many neighbors to request from FAISS. FAISS cannot filter on
    /// metadata, so when a filter is present we over-fetch <c>topK * oversample</c>
    /// candidates and filter them in managed code. The result is always clamped to the
    /// number of documents actually in the index (asking FAISS for more than it holds
    /// just yields <c>-1</c> padding ids).
    /// </summary>
    /// <param name="topK">Requested number of results.</param>
    /// <param name="oversample">Over-fetch multiplier applied when filtering (>= 1).</param>
    /// <param name="totalDocs">Number of documents currently indexed.</param>
    /// <param name="hasFilters">Whether a metadata filter will be applied.</param>
    public static int ComputeFetchCount(int topK, int oversample, int totalDocs, bool hasFilters)
    {
        if (topK <= 0 || totalDocs <= 0)
            return 0;

        long want = hasFilters
            ? (long)topK * Math.Max(1, oversample)
            : topK;

        if (want > totalDocs) want = totalDocs;
        if (want < 1) want = 1;
        return (int)want;
    }

    /// <summary>
    /// Maps ranked FAISS hits back to documents, applies the metadata filter, and takes
    /// the top <paramref name="topK"/> — preserving FAISS's (best-first) ordering.
    /// </summary>
    /// <typeparam name="T">Numeric type of the document relevance score.</typeparam>
    /// <param name="ranked">
    /// FAISS hits in best-first order: the resolved sidecar entry and the raw distance/score FAISS returned.
    /// </param>
    /// <param name="filters">Metadata filters to apply (empty = keep all).</param>
    /// <param name="topK">Maximum number of documents to return.</param>
    /// <param name="scoreConverter">Converts a raw FAISS distance/score into a T relevance score.</param>
    /// <param name="matches">
    /// Metadata evaluator (document metadata, filters) -&gt; keep. Injected so the store can
    /// reuse the shared <c>DocumentStoreBase.MatchesFilters</c> evaluator.
    /// </param>
    public static List<Document<T>> SelectTopK<T>(
        IEnumerable<(FaissSidecarEntry Entry, double RawScore)> ranked,
        Dictionary<string, object> filters,
        int topK,
        Func<double, T> scoreConverter,
        Func<Dictionary<string, object>, Dictionary<string, object>, bool> matches)
    {
        var results = new List<Document<T>>(Math.Max(0, topK));
        if (topK <= 0)
            return results;

        var hasFilters = filters != null && filters.Count > 0;
        foreach (var (entry, rawScore) in ranked)
        {
            if (entry == null)
                continue;
            if (hasFilters && !matches(entry.Metadata, filters!))
                continue;

            results.Add(new Document<T>(entry.DocumentId, entry.Content, new Dictionary<string, object>(entry.Metadata))
            {
                RelevanceScore = scoreConverter(rawScore),
                HasRelevanceScore = true
            });

            if (results.Count >= topK)
                break;
        }

        return results;
    }
}
