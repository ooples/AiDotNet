using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json;

using FaissIndex = FaissNet.Index;
using FaissMetricType = FaissNet.MetricType;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// The approximate-nearest-neighbor index structure backing a <see cref="FaissDocumentStore{T}"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This picks the trade-off between search speed, memory, and accuracy.
/// <list type="bullet">
/// <item><b>Flat</b> — exact brute-force search. Perfectly accurate, simplest, no training. Best for small/medium collections.</item>
/// <item><b>IVFFlat</b> — clusters vectors into cells and only searches the closest cells. Much faster on large collections; requires a one-time training step and returns approximate results.</item>
/// <item><b>HNSW</b> — a navigable small-world graph. Very fast queries with high recall, no training, but higher memory and no in-place deletion.</item>
/// <item><b>IVFPQ</b> — IVF plus product quantization, which compresses vectors so billions fit in memory. Smallest footprint; requires training and is the most approximate.</item>
/// </list>
/// </para>
/// </remarks>
public enum FaissIndexType
{
    /// <summary>Exact brute-force search (<c>IDMap2,Flat</c>). No training required.</summary>
    Flat = 0,

    /// <summary>Inverted-file index with flat storage (<c>IVF{nlist},Flat</c>). Requires training.</summary>
    IVFFlat = 1,

    /// <summary>Hierarchical Navigable Small World graph (<c>IDMap2,HNSW{M}</c>). No training; no in-place deletion.</summary>
    HNSW = 2,

    /// <summary>Inverted-file index with product quantization (<c>IVF{nlist},PQ{m}</c>). Requires training; compresses vectors.</summary>
    IVFPQ = 3
}

/// <summary>
/// The distance metric used for similarity search.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is how "closeness" between two vectors is measured.
/// <list type="bullet">
/// <item><b>Cosine</b> — angle between vectors, ignoring length. Implemented as inner product over L2-normalized vectors. The usual choice for text embeddings.</item>
/// <item><b>InnerProduct</b> — raw dot product (no normalization). Higher is more similar.</item>
/// <item><b>L2</b> — Euclidean distance. Lower is more similar.</item>
/// </list>
/// </para>
/// </remarks>
public enum FaissDistanceMetric
{
    /// <summary>Cosine similarity via inner product over L2-normalized vectors.</summary>
    Cosine = 0,

    /// <summary>Raw inner (dot) product.</summary>
    InnerProduct = 1,

    /// <summary>Euclidean (L2) distance.</summary>
    L2 = 2
}

/// <summary>
/// A real, native FAISS-backed vector store (Facebook AI Similarity Search) exposing IVF / HNSW / PQ
/// approximate-nearest-neighbor indexes through the managed <c>FaissNet</c> wrapper.
/// </summary>
/// <typeparam name="T">The numeric data type used for vectors and relevance scores (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Unlike the in-memory <see cref="FAISSDocumentStore{T}"/> (a brute-force simulation kept for
/// dependency-free scenarios), this store delegates indexing and search to the native FAISS
/// library and therefore matches FAISS on raw ANN throughput. It lives in the opt-in
/// <c>AiDotNet.Storage.Faiss</c> package so the core AiDotNet package takes no native dependency.
/// </para>
/// <para>
/// FAISS stores only vectors keyed by an int64 id, so this store keeps a <see cref="FaissSidecar"/>
/// mapping each int64 id to the document's string id, content, metadata, and (metric-adjusted)
/// embedding. The sidecar drives id round-tripping, metadata filtering, index rebuilds on deletion
/// for index types that lack in-place removal (HNSW), and persistence.
/// </para>
/// <para><b>Native requirements:</b> FaissNet ships a win-x64 native runtime. On platforms/TFMs where
/// the native library cannot be loaded, constructing this store throws (the wrapper raises a
/// <see cref="DllNotFoundException"/> / type-initialization error); callers that need graceful
/// degradation should catch that and fall back to another <c>IDocumentStore</c> implementation.</para>
/// </remarks>
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public sealed class FaissDocumentStore<T> : DocumentStoreBase<T>, IDisposable
{
    private const int PersistenceVersion = 1;

    private readonly FaissIndexType _indexType;
    private readonly FaissDistanceMetric _metric;
    private readonly FaissMetricType _faissMetric;
    private readonly int _vectorDimension;
    private readonly int _nlist;
    private readonly int _hnswM;
    private readonly int _pqM;
    private readonly int _searchOversample;
    private readonly bool _normalize;
    private readonly bool _requiresTraining;

    private FaissIndex _index;
    private FaissSidecar _sidecar;
    private bool _isTrained;
    private bool _disposed;

    /// <inheritdoc/>
    public override int DocumentCount => _sidecar.Count;

    /// <inheritdoc/>
    public override int VectorDimension => _vectorDimension;

    /// <summary>The ANN index structure this store was created with.</summary>
    public FaissIndexType IndexType => _indexType;

    /// <summary>The distance metric this store was created with.</summary>
    public FaissDistanceMetric Metric => _metric;

    /// <summary>
    /// Creates a native FAISS-backed document store.
    /// </summary>
    /// <param name="vectorDimension">Dimensionality of the embeddings (must be positive).</param>
    /// <param name="indexType">Which ANN index structure to build. Defaults to exact <see cref="FaissIndexType.Flat"/>.</param>
    /// <param name="metric">Distance metric. Defaults to <see cref="FaissDistanceMetric.Cosine"/>.</param>
    /// <param name="nlist">Number of IVF cells (only used by IVF-based index types).</param>
    /// <param name="hnswM">Number of neighbor links per node (only used by <see cref="FaissIndexType.HNSW"/>).</param>
    /// <param name="pqM">Number of PQ sub-quantizers (only used by <see cref="FaissIndexType.IVFPQ"/>; must divide <paramref name="vectorDimension"/>).</param>
    /// <param name="searchOversample">
    /// Over-fetch multiplier for metadata-filtered searches. FAISS cannot filter on metadata, so
    /// filtered queries request <c>topK * searchOversample</c> candidates and filter them in managed code.
    /// </param>
    public FaissDocumentStore(
        int vectorDimension,
        FaissIndexType indexType = FaissIndexType.Flat,
        FaissDistanceMetric metric = FaissDistanceMetric.Cosine,
        int nlist = 100,
        int hnswM = 32,
        int pqM = 8,
        int searchOversample = 4)
    {
        if (vectorDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(vectorDimension), "Vector dimension must be positive");
        if (nlist <= 0)
            throw new ArgumentOutOfRangeException(nameof(nlist), "nlist must be positive");
        if (hnswM <= 0)
            throw new ArgumentOutOfRangeException(nameof(hnswM), "hnswM must be positive");
        if (pqM <= 0)
            throw new ArgumentOutOfRangeException(nameof(pqM), "pqM must be positive");
        if (indexType == FaissIndexType.IVFPQ && vectorDimension % pqM != 0)
            throw new ArgumentException($"For IVFPQ, vectorDimension ({vectorDimension}) must be divisible by pqM ({pqM}).", nameof(pqM));
        if (searchOversample < 1)
            throw new ArgumentOutOfRangeException(nameof(searchOversample), "searchOversample must be at least 1");

        _vectorDimension = vectorDimension;
        _indexType = indexType;
        _metric = metric;
        _nlist = nlist;
        _hnswM = hnswM;
        _pqM = pqM;
        _searchOversample = searchOversample;
        _faissMetric = ToFaissMetric(metric);
        _normalize = metric == FaissDistanceMetric.Cosine;
        _requiresTraining = indexType is FaissIndexType.IVFFlat or FaissIndexType.IVFPQ;

        _sidecar = new FaissSidecar();
        _index = CreateIndex();
        _isTrained = !_requiresTraining;
    }

    // Private ctor used by Load: adopts an already-populated index + sidecar.
    private FaissDocumentStore(
        FaissIndex index,
        FaissSidecar sidecar,
        int vectorDimension,
        FaissIndexType indexType,
        FaissDistanceMetric metric,
        int nlist,
        int hnswM,
        int pqM,
        int searchOversample,
        bool isTrained)
    {
        _index = index;
        _sidecar = sidecar;
        _vectorDimension = vectorDimension;
        _indexType = indexType;
        _metric = metric;
        _nlist = nlist;
        _hnswM = hnswM;
        _pqM = pqM;
        _searchOversample = searchOversample;
        _faissMetric = ToFaissMetric(metric);
        _normalize = metric == FaissDistanceMetric.Cosine;
        _requiresTraining = indexType is FaissIndexType.IVFFlat or FaissIndexType.IVFPQ;
        _isTrained = isTrained;
    }

    private static FaissMetricType ToFaissMetric(FaissDistanceMetric metric) => metric switch
    {
        FaissDistanceMetric.Cosine => FaissMetricType.METRIC_INNER_PRODUCT,
        FaissDistanceMetric.InnerProduct => FaissMetricType.METRIC_INNER_PRODUCT,
        FaissDistanceMetric.L2 => FaissMetricType.METRIC_L2,
        _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, "Unsupported FAISS metric")
    };

    /// <summary>
    /// Builds the FAISS index-factory string for the configured index type.
    /// See https://github.com/facebookresearch/faiss/wiki/The-index-factory.
    /// </summary>
    private string BuildFactoryString() => _indexType switch
    {
        // IDMap2 lets Flat/HNSW carry arbitrary int64 ids and reconstruct vectors.
        // IVF-based indexes carry ids natively, so no IDMap wrapper is used there.
        FaissIndexType.Flat => "IDMap2,Flat",
        FaissIndexType.HNSW => $"IDMap2,HNSW{_hnswM}",
        FaissIndexType.IVFFlat => $"IVF{_nlist},Flat",
        FaissIndexType.IVFPQ => $"IVF{_nlist},PQ{_pqM}",
        _ => throw new ArgumentOutOfRangeException(nameof(_indexType), _indexType, "Unsupported FAISS index type")
    };

    private FaissIndex CreateIndex() => FaissIndex.Create(_vectorDimension, BuildFactoryString(), _faissMetric);

    /// <inheritdoc/>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        AddBatchCore(new[] { vectorDocument });
    }

    /// <inheritdoc/>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0)
            return;

        var n = vectorDocuments.Count;
        var vectors = new float[n][];
        var ids = new long[n];
        var replaced = new List<long>();

        for (int i = 0; i < n; i++)
        {
            var vd = vectorDocuments[i];
            if (vd.Embedding.Length != _vectorDimension)
                throw new ArgumentException(
                    $"Vector dimension mismatch. Expected {_vectorDimension}, got {vd.Embedding.Length}", nameof(vectorDocuments));

            var vec = ToFloatVector(vd.Embedding);
            var id = _sidecar.Upsert(
                vd.Document.Id,
                vd.Document.Content,
                vd.Document.Metadata,
                vec,
                out var replacedId);

            vectors[i] = vec;
            ids[i] = id;
            if (replacedId.HasValue)
                replaced.Add(replacedId.Value);
        }

        // Evict any documents that were replaced by an id-collision (re-add / update).
        if (replaced.Count > 0)
            EvictFromIndex(replaced.ToArray());

        EnsureTrained(vectors);

        var flat = Flatten(vectors);
        _index.AddWithIdsFlat(n, flat, ids);
    }

    /// <inheritdoc/>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        if (_sidecar.Count == 0)
            return Array.Empty<Document<T>>();

        var hasFilters = metadataFilters != null && metadataFilters.Count > 0;
        var fetch = FaissRetrievalPlanner.ComputeFetchCount(topK, _searchOversample, _sidecar.Count, hasFilters);
        if (fetch <= 0)
            return Array.Empty<Document<T>>();

        var query = ToFloatVector(queryVector);
        var (distances, resultIds) = _index.SearchFlat(1, query, fetch);

        // FAISS returns hits best-first, padding with id == -1 when fewer than k exist.
        var ranked = new List<(FaissSidecarEntry Entry, double RawScore)>(resultIds.Length);
        for (int i = 0; i < resultIds.Length; i++)
        {
            var id = resultIds[i];
            if (id < 0)
                continue;
            if (_sidecar.TryGetByFaissId(id, out var entry))
                ranked.Add((entry, distances[i]));
        }

        return FaissRetrievalPlanner.SelectTopK<T>(
            ranked,
            metadataFilters ?? new Dictionary<string, object>(),
            topK,
            RawScoreToRelevance,
            MetadataMatches);
    }

    // Reuses the shared DocumentStoreBase.MatchesFilters evaluator against just the metadata.
    private bool MetadataMatches(Dictionary<string, object> metadata, Dictionary<string, object> filters)
        => MatchesFilters(new Document<T> { Metadata = metadata }, filters);

    /// <summary>
    /// Converts a raw FAISS distance/score into a relevance score where higher always means more relevant.
    /// Inner-product/cosine scores are used directly; L2 (squared) distances map to 1/(1+d).
    /// </summary>
    private T RawScoreToRelevance(double raw) => _metric == FaissDistanceMetric.L2
        ? NumOps.FromDouble(1.0 / (1.0 + raw))
        : NumOps.FromDouble(raw);

    /// <inheritdoc/>
    protected override Document<T>? GetByIdCore(string documentId)
        => _sidecar.TryGetByDocumentId(documentId, out var entry)
            ? new Document<T>(entry.DocumentId, entry.Content, new Dictionary<string, object>(entry.Metadata))
            : null;

    /// <inheritdoc/>
    protected override bool RemoveCore(string documentId)
    {
        if (!_sidecar.RemoveByDocumentId(documentId, out var removedFaissId))
            return false;

        EvictFromIndex(new[] { removedFaissId });
        return true;
    }

    /// <inheritdoc/>
    protected override IEnumerable<Document<T>> GetAllCore()
        => _sidecar.Entries
            .Select(e => new Document<T>(e.DocumentId, e.Content, new Dictionary<string, object>(e.Metadata)))
            .ToList();

    /// <inheritdoc/>
    public override void Clear()
    {
        _sidecar.Clear();
        var old = _index;
        _index = CreateIndex();
        old.Dispose();
        _isTrained = !_requiresTraining;
    }

    /// <summary>
    /// Removes the given int64 ids from the FAISS index. Index types that implement in-place
    /// deletion (Flat, IVF*) use <c>RemoveIds</c>; those that do not (HNSW) trigger a full
    /// rebuild from the sidecar, which is the source of truth for live documents.
    /// </summary>
    private void EvictFromIndex(long[] faissIds)
    {
        if (faissIds.Length == 0)
            return;

        if (_indexType == FaissIndexType.HNSW)
        {
            RebuildIndex();
            return;
        }

        try
        {
            _index.RemoveIds(faissIds);
        }
        catch
        {
            // Fall back to a rebuild if this index type cannot delete in place.
            RebuildIndex();
        }
    }

    /// <summary>
    /// Rebuilds the FAISS index from scratch using the sidecar's live embeddings. Used when a
    /// deletion cannot be applied in place, so the index and sidecar stay consistent.
    /// </summary>
    private void RebuildIndex()
    {
        var fresh = CreateIndex();
        var entries = _sidecar.Entries.ToList();
        _isTrained = !_requiresTraining;

        if (entries.Count > 0)
        {
            var vectors = entries.Select(e => e.Embedding).ToArray();
            var ids = entries.Select(e => e.FaissId).ToArray();

            if (_requiresTraining && !_isTrained)
            {
                fresh.Train(vectors.Length, Flatten(vectors));
                _isTrained = true;
            }

            fresh.AddWithIdsFlat(vectors.Length, Flatten(vectors), ids);
        }

        var old = _index;
        _index = fresh;
        old.Dispose();
    }

    /// <summary>Trains IVF/PQ indexes on first population; a no-op for Flat/HNSW or once trained.</summary>
    private void EnsureTrained(float[][] vectors)
    {
        if (!_requiresTraining || _isTrained || vectors.Length == 0)
            return;

        _index.Train(vectors.Length, Flatten(vectors));
        _isTrained = true;
    }

    private float[] ToFloatVector(Vector<T> vector)
    {
        var raw = vector.ToArray();
        var result = new float[raw.Length];
        for (int i = 0; i < raw.Length; i++)
            result[i] = (float)Convert.ToDouble(raw[i], CultureInfo.InvariantCulture);

        if (_normalize)
            NormalizeInPlace(result);
        return result;
    }

    private static void NormalizeInPlace(float[] vector)
    {
        double sumSq = 0.0;
        for (int i = 0; i < vector.Length; i++)
            sumSq += (double)vector[i] * vector[i];

        if (sumSq <= 0.0)
            return;

        var inv = 1.0 / Math.Sqrt(sumSq);
        for (int i = 0; i < vector.Length; i++)
            vector[i] = (float)(vector[i] * inv);
    }

    private static float[] Flatten(float[][] vectors)
    {
        if (vectors.Length == 0)
            return Array.Empty<float>();

        var dim = vectors[0].Length;
        var flat = new float[vectors.Length * dim];
        for (int i = 0; i < vectors.Length; i++)
            Buffer.BlockCopy(vectors[i], 0, flat, i * dim * sizeof(float), dim * sizeof(float));
        return flat;
    }

    /// <summary>
    /// Persists the store to disk: the native FAISS index to <c>{path}.faissindex</c> and the
    /// sidecar + configuration to <c>{path}.faissmeta.json</c>.
    /// </summary>
    /// <param name="path">Base path (without extension) for the two output files.</param>
    public void Save(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be null or empty", nameof(path));

        _index.Save(IndexPath(path));

        var meta = new PersistedMeta
        {
            Version = PersistenceVersion,
            IndexType = _indexType,
            Metric = _metric,
            VectorDimension = _vectorDimension,
            Nlist = _nlist,
            HnswM = _hnswM,
            PqM = _pqM,
            SearchOversample = _searchOversample,
            IsTrained = _isTrained,
            SidecarJson = _sidecar.ToJson()
        };
        File.WriteAllText(MetaPath(path), JsonConvert.SerializeObject(meta));
    }

    /// <summary>
    /// Loads a store previously written by <see cref="Save"/>.
    /// </summary>
    /// <param name="path">The same base path passed to <see cref="Save"/>.</param>
    public static FaissDocumentStore<T> Load(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be null or empty", nameof(path));

        var metaJson = File.ReadAllText(MetaPath(path));
        var meta = JsonConvert.DeserializeObject<PersistedMeta>(metaJson)
                   ?? throw new InvalidOperationException("Failed to deserialize FAISS store metadata.");

        var index = FaissIndex.Load(IndexPath(path));
        var sidecar = FaissSidecar.FromJson(meta.SidecarJson);

        return new FaissDocumentStore<T>(
            index,
            sidecar,
            meta.VectorDimension,
            meta.IndexType,
            meta.Metric,
            meta.Nlist,
            meta.HnswM,
            meta.PqM,
            meta.SearchOversample,
            meta.IsTrained);
    }

    private static string IndexPath(string basePath) => basePath + ".faissindex";
    private static string MetaPath(string basePath) => basePath + ".faissmeta.json";

    /// <summary>Releases the native FAISS index.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _index.Dispose();
    }

    private sealed class PersistedMeta
    {
        public int Version { get; set; }
        public FaissIndexType IndexType { get; set; }
        public FaissDistanceMetric Metric { get; set; }
        public int VectorDimension { get; set; }
        public int Nlist { get; set; }
        public int HnswM { get; set; }
        public int PqM { get; set; }
        public int SearchOversample { get; set; }
        public bool IsTrained { get; set; }
        public string SidecarJson { get; set; } = string.Empty;
    }
}
