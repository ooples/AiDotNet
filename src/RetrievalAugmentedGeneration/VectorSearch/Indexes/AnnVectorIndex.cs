using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Ann;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// The approximate-nearest-neighbour structure backing an <see cref="AnnVectorIndex{T}"/>.
    /// Mirrors the FAISS index families but runs entirely on the AiDotNet Tensors stack.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the speed/memory/accuracy trade-off.
    /// <list type="bullet">
    /// <item><b>Flat</b> — exact brute-force. Perfectly accurate, no training. Best for small/medium sets.</item>
    /// <item><b>Ivf</b> — clusters vectors and only scans the closest clusters. Much faster on large sets; approximate; trained.</item>
    /// <item><b>Pq</b> — product quantization: compressed codes searched with asymmetric distance. Tiny footprint; trained.</item>
    /// <item><b>IvfPq</b> — IVF over PQ-compressed residuals (FAISS's workhorse). Smallest + fast; most approximate; trained.</item>
    /// </list></para>
    /// </remarks>
    public enum AnnVectorIndexType
    {
        /// <summary>Exact brute-force scan. No training.</summary>
        Flat = 0,
        /// <summary>Inverted file (coarse k-means) — probe the nearest lists only.</summary>
        Ivf = 1,
        /// <summary>Product quantization — compressed codes + ADC search.</summary>
        Pq = 2,
        /// <summary>IVF over PQ-compressed residuals.</summary>
        IvfPq = 3
    }

    /// <summary>The distance metric used by an <see cref="AnnVectorIndex{T}"/>.</summary>
    public enum AnnVectorMetric
    {
        /// <summary>Cosine similarity via inner product over L2-normalized vectors (the usual choice for text embeddings).</summary>
        Cosine = 0,
        /// <summary>Raw inner (dot) product — higher is more similar.</summary>
        InnerProduct = 1,
        /// <summary>Euclidean (L2) distance — lower is more similar.</summary>
        L2 = 2
    }

    /// <summary>
    /// A dependency-free approximate-nearest-neighbour index (Flat / IVF / PQ / IVFPQ) implemented on the
    /// AiDotNet Tensors stack via <see cref="AnnIndex"/>. This is the native replacement for the external
    /// FaissNet-backed index whose IVF/PQ path is blocked by an incomplete MKL redistribution: it takes no
    /// external native dependency and, when a GPU backend is attached, dispatches its distance / assignment /
    /// PQ scans to the fused ANN kernels (<see cref="IAnnBackend"/>) across all supported backends, falling
    /// back to the managed CPU reference otherwise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The underlying <see cref="AnnIndex"/> trains its coarse quantizer / PQ codebooks once on a representative
    /// sample and does not support in-place deletion, so this adapter keeps the live vectors as the source of
    /// truth and lazily (re)builds the index on the next <see cref="Search"/> after any mutation. That keeps the
    /// simple incremental <see cref="IVectorIndex{T}"/> contract (Add / Remove / Search) correct over an index
    /// that is fundamentally train-then-populate.
    /// </para>
    /// <para>
    /// <b>Scale hardening — incremental add.</b> Discarding and retraining the whole ANN index on every mutation
    /// is O(n) per query after any change. When the index is already built and either <see cref="AnnVectorIndexType.Flat"/>
    /// or a trained IVF/PQ/IVFPQ index, and only <em>additions</em> (no removals, overwrites, GPU re-attach, or
    /// dimension changes) have happened since the last build, the newly-added vectors are appended to the live
    /// <see cref="AnnIndex"/> via <see cref="AnnIndex.Add(long, float[])"/> incrementally rather than rebuilt from
    /// scratch — the trained quantizer / codebooks are reused. A removal (or the first build, or a not-yet-trained
    /// index) forces a full rebuild. Ordinals stay contiguous and in insertion order, so the ordinal-to-id map and
    /// the search results are equivalent to the original full-rebuild path.
    /// </para>
    /// <para>
    /// <b>Scale hardening — persistence.</b> <see cref="Save(string)"/> / <see cref="Load(string)"/> serialize the
    /// configuration plus the live vectors (id -&gt; float[]) and insertion order to JSON (Newtonsoft). Because
    /// <see cref="AnnIndex"/> is not itself serializable, <see cref="Load(string)"/> reconstructs the adapter from
    /// the persisted vectors and parameters; the deterministic seeded k-means makes the rebuilt index reproducible,
    /// so a loaded index returns the same search results as the saved one.
    /// </para>
    /// <para><b>For Beginners:</b> You just add and remove vectors by id and search — the training and rebuilding
    /// that IVF/PQ need happen automatically behind the scenes the first time you query after a change.</para>
    /// </remarks>
    /// <typeparam name="T">The numeric data type used for vectors and scores (typically float or double).</typeparam>
    [ComponentType(ComponentType.VectorIndex)]
    [PipelineStage(PipelineStage.Retrieval)]
    public sealed class AnnVectorIndex<T> : IVectorIndex<T>
    {
        private const int PersistenceVersion = 1;

        private readonly AnnVectorIndexType _type;
        private readonly AnnVectorMetric _metric;
        private readonly int _annMetric;      // AnnPrimitives.Metric* code
        private readonly bool _normalize;     // cosine => normalize + inner product
        private readonly int _nlist;
        private readonly int _nprobe;
        private readonly int _m;
        private readonly int _ksub;
        private readonly int _seed;
        private readonly bool _requiresTraining;
        private readonly INumericOperations<T> _numOps;

        // Live vectors are the source of truth for lazy (re)builds; insertion order gives deterministic ordinals.
        private readonly Dictionary<string, float[]> _vectors = new();
        private readonly List<string> _order = new();

        private int _dim;
        private AnnIndex? _index;
        private List<string>? _ordinalToId;
        private bool _dirty = true;
        // True when the next EnsureBuilt must discard and rebuild the AnnIndex from scratch (first build, a removal,
        // a same-id overwrite, a GPU re-attach, or a dimension change). When false and the index is already built,
        // pure additions are appended incrementally instead.
        private bool _needsRebuild = true;
        private IDirectGpuBackend? _gpu;

        /// <summary>Number of full (re)builds of the underlying <see cref="AnnIndex"/> performed so far. Test/telemetry hook.</summary>
        internal int RebuildCount { get; private set; }

        /// <summary>Number of vectors appended incrementally (without a full rebuild) so far. Test/telemetry hook.</summary>
        internal int IncrementalAddCount { get; private set; }

        /// <inheritdoc/>
        public int Count => _vectors.Count;

        /// <summary>The ANN index structure this index was created with.</summary>
        public AnnVectorIndexType IndexType => _type;

        /// <summary>The distance metric this index was created with.</summary>
        public AnnVectorMetric Metric => _metric;

        /// <summary>
        /// Creates a native ANN vector index.
        /// </summary>
        /// <param name="indexType">Which ANN structure to build. Defaults to exact <see cref="AnnVectorIndexType.Flat"/>.</param>
        /// <param name="dimension">Vector dimensionality (0 = inferred from the first added vector).</param>
        /// <param name="metric">Distance metric. Defaults to <see cref="AnnVectorMetric.Cosine"/>.</param>
        /// <param name="nlist">Number of IVF coarse lists (IVF/IVFPQ only).</param>
        /// <param name="nprobe">Number of IVF lists probed per query (IVF/IVFPQ only).</param>
        /// <param name="m">Number of PQ subspaces (PQ/IVFPQ only; must divide <paramref name="dimension"/> when known).</param>
        /// <param name="ksub">PQ sub-centroids per subspace (PQ/IVFPQ only; typically 256).</param>
        /// <param name="seed">Deterministic k-means seed.</param>
        public AnnVectorIndex(
            AnnVectorIndexType indexType = AnnVectorIndexType.Flat,
            int dimension = 0,
            AnnVectorMetric metric = AnnVectorMetric.Cosine,
            int nlist = 64,
            int nprobe = 8,
            int m = 8,
            int ksub = 256,
            int seed = 42)
        {
            if (dimension < 0) throw new ArgumentOutOfRangeException(nameof(dimension));
            if (nlist <= 0) throw new ArgumentOutOfRangeException(nameof(nlist));
            if (nprobe <= 0) throw new ArgumentOutOfRangeException(nameof(nprobe));
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
            if (ksub <= 0) throw new ArgumentOutOfRangeException(nameof(ksub));

            _type = indexType;
            _metric = metric;
            _dim = dimension;
            _normalize = metric == AnnVectorMetric.Cosine;
            // Cosine maps to L2 over L2-normalized vectors, NOT inner product: for unit vectors
            // ||q-x||^2 = 2 - 2<q,x>, so L2-nearest is exactly cosine-nearest, and — critically — L2 keeps
            // the PQ/IVF Lloyd's k-means CONSISTENT (assignment metric matches the Euclidean-mean centroid
            // update). Training PQ codebooks under inner product (assign by max-IP, update by mean) produces
            // near-useless sub-quantizers, which collapsed PQ/IVFPQ recall (benchmarked 0.04-0.45). Only a
            // raw (un-normalized) inner-product metric actually uses the IP path.
            _annMetric = metric == AnnVectorMetric.InnerProduct ? AnnPrimitives.MetricInnerProduct : AnnPrimitives.MetricL2;
            _nlist = nlist;
            _nprobe = nprobe;
            _m = m;
            _ksub = ksub;
            _seed = seed;
            _requiresTraining = indexType != AnnVectorIndexType.Flat;
            _numOps = MathHelper.GetNumericOperations<T>();
        }

        /// <summary>
        /// Attaches a GPU backend so search/training dispatch to the fused ANN kernels when available. Pass a
        /// backend from <c>DirectGpuBackendFactory.Create()</c>, or <c>null</c> to force the managed CPU path.
        /// Takes effect on the next (re)build (forces a full rebuild).
        /// </summary>
        public void AttachGpu(IDirectGpuBackend? backend)
        {
            _gpu = backend;
            _dirty = true;
            _needsRebuild = true;
        }

        /// <inheritdoc/>
        public void Add(string id, Vector<T> vector)
        {
            if (string.IsNullOrEmpty(id)) throw new ArgumentException("Id must be non-empty.", nameof(id));
            if (vector == null) throw new ArgumentNullException(nameof(vector));
            if (_dim == 0) _dim = vector.Length;
            if (vector.Length != _dim)
                throw new ArgumentException($"Vector dimension mismatch. Expected {_dim}, got {vector.Length}.", nameof(vector));

            if (_vectors.ContainsKey(id))
            {
                // Overwriting an existing (possibly already-indexed) vector: AnnIndex cannot update in place,
                // so this must force a full rebuild to stay correct.
                _needsRebuild = true;
            }
            else
            {
                _order.Add(id);
            }

            _vectors[id] = ToFloat(vector);
            _dirty = true;
        }

        /// <inheritdoc/>
        public void AddBatch(Dictionary<string, Vector<T>> vectors)
        {
            if (vectors == null) throw new ArgumentNullException(nameof(vectors));
            foreach (var kvp in vectors)
                Add(kvp.Key, kvp.Value);
        }

        /// <inheritdoc/>
        public bool Remove(string id)
        {
            if (id == null || !_vectors.Remove(id)) return false;
            _order.Remove(id);
            _dirty = true;
            _needsRebuild = true; // deletion is unsupported by AnnIndex; force a full rebuild.
            return true;
        }

        /// <inheritdoc/>
        public void Clear()
        {
            _vectors.Clear();
            _order.Clear();
            _index = null;
            _ordinalToId = null;
            _dirty = true;
            _needsRebuild = true;
        }

        /// <inheritdoc/>
        public List<(string Id, T Score)> Search(Vector<T> query, int k)
        {
            if (query == null) throw new ArgumentNullException(nameof(query));
            if (k <= 0 || _vectors.Count == 0) return new List<(string Id, T Score)>();

            EnsureBuilt();
            if (_index == null || _ordinalToId == null) return new List<(string Id, T Score)>();

            var q = ToFloat(query);
            var (ids, distances) = _index.Search(q, k);

            var results = new List<(string Id, T Score)>(ids.Length);
            for (int i = 0; i < ids.Length; i++)
            {
                int ordinal = (int)ids[i];
                if (ordinal < 0 || ordinal >= _ordinalToId.Count) continue;
                results.Add((_ordinalToId[ordinal], ToScore(distances[i])));
            }
            return results;
        }

        /// <summary>
        /// Lazily (re)builds — or incrementally extends — the underlying <see cref="AnnIndex"/> from the live
        /// vectors after any mutation.
        /// </summary>
        private void EnsureBuilt()
        {
            if (_index != null && !_dirty) return;

            // Fast path: the index is already built and trained and only additions happened since the last build.
            // Append the new tail of the insertion order to the live AnnIndex, reusing the trained quantizer /
            // codebooks instead of discarding and rebuilding from scratch. Because no removals have occurred, the
            // already-indexed prefix of _order matches _ordinalToId one-to-one, so the new vectors take contiguous
            // ordinals continuing from _ordinalToId.Count — exactly the ordinals a full rebuild would assign.
            if (_index != null && _ordinalToId != null && !_needsRebuild)
            {
                for (int i = _ordinalToId.Count; i < _order.Count; i++)
                {
                    var id = _order[i];
                    _index.Add(i, _vectors[id]);
                    _ordinalToId.Add(id);
                    IncrementalAddCount++;
                }
                _dirty = false;
                return;
            }

            var live = _order.Where(_vectors.ContainsKey).ToList();
            if (live.Count == 0 || _dim == 0)
            {
                _index = null;
                _ordinalToId = null;
                _dirty = false;
                _needsRebuild = true;
                return;
            }

            var idx = new AnnIndex((AnnIndexType)_type, _dim, _annMetric, _nlist, _nprobe, _m, _ksub, _seed);
            idx.AttachGpu(_gpu);

            if (_requiresTraining)
            {
                var flat = new float[(long)live.Count * _dim];
                for (int i = 0; i < live.Count; i++)
                    Array.Copy(_vectors[live[i]], 0, flat, (long)i * _dim, _dim);
                idx.Train(flat, live.Count);
            }

            var ordinalToId = new List<string>(live.Count);
            for (int i = 0; i < live.Count; i++)
            {
                idx.Add(i, _vectors[live[i]]);
                ordinalToId.Add(live[i]);
            }

            _index = idx;
            _ordinalToId = ordinalToId;
            _dirty = false;
            _needsRebuild = false;
            RebuildCount++;
        }

        /// <summary>
        /// Persists the index configuration, the live vectors (id -&gt; stored float[]) and their insertion order to
        /// <paramref name="path"/> as JSON. The underlying <see cref="AnnIndex"/> is not serialized — it is
        /// reconstructed deterministically by <see cref="Load(string)"/> from the persisted vectors and parameters.
        /// </summary>
        /// <param name="path">Destination file path.</param>
        public void Save(string path)
        {
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path must be non-empty.", nameof(path));

            // Canonicalize to a full-rebuild state so the persisted index and any freshly-loaded index share the
            // exact same (fully retrained) ANN structure — guaranteeing Save/Load search-result equivalence even
            // when the live index currently reflects an incremental-add history.
            _needsRebuild = true;
            _dirty = true;
            EnsureBuilt();

            var state = new PersistedState
            {
                Version = PersistenceVersion,
                IndexType = (int)_type,
                Metric = (int)_metric,
                Dimension = _dim,
                Nlist = _nlist,
                Nprobe = _nprobe,
                M = _m,
                Ksub = _ksub,
                Seed = _seed,
                Order = new List<string>(_order),
                Vectors = _order.Where(_vectors.ContainsKey).ToDictionary(id => id, id => _vectors[id])
            };

            var json = JsonConvert.SerializeObject(state, Formatting.Indented);
            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Reconstructs an <see cref="AnnVectorIndex{T}"/> previously written by <see cref="Save(string)"/>. The
        /// live vectors and configuration are restored and the underlying <see cref="AnnIndex"/> is rebuilt
        /// deterministically on the first query, so the loaded index returns the same search results as the saved one.
        /// </summary>
        /// <param name="path">Source file path.</param>
        public static AnnVectorIndex<T> Load(string path)
        {
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path must be non-empty.", nameof(path));

            var json = File.ReadAllText(path);
            var state = JsonConvert.DeserializeObject<PersistedState>(json);
            if (state == null) throw new InvalidOperationException($"Failed to deserialize AnnVectorIndex from '{path}'.");

            var index = new AnnVectorIndex<T>(
                (AnnVectorIndexType)state.IndexType,
                state.Dimension,
                (AnnVectorMetric)state.Metric,
                state.Nlist,
                state.Nprobe,
                state.M,
                state.Ksub,
                state.Seed);

            // Restore the live vectors directly (they are the already-processed stored float[], so do NOT re-run
            // ToFloat/normalization) and the insertion order, then mark for a deterministic first-query rebuild.
            var order = state.Order ?? new List<string>();
            var vectors = state.Vectors ?? new Dictionary<string, float[]>();
            foreach (var id in order)
            {
                if (!vectors.TryGetValue(id, out var vec) || vec == null) continue;
                if (!index._vectors.ContainsKey(id)) index._order.Add(id);
                index._vectors[id] = vec;
            }

            index._dirty = true;
            index._needsRebuild = true;
            return index;
        }

        private float[] ToFloat(Vector<T> vector)
        {
            var result = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
                result[i] = (float)Convert.ToDouble(vector[i], CultureInfo.InvariantCulture);
            if (_normalize) NormalizeInPlace(result);
            return result;
        }

        private static void NormalizeInPlace(float[] vector)
        {
            double sumSq = 0.0;
            for (int i = 0; i < vector.Length; i++)
                sumSq += (double)vector[i] * vector[i];
            if (sumSq <= 0.0) return;
            var inv = 1.0 / Math.Sqrt(sumSq);
            for (int i = 0; i < vector.Length; i++)
                vector[i] = (float)(vector[i] * inv);
        }

        // AnnIndex returns metric-native distances (L2: smaller nearer; inner product: larger nearer) already
        // ordered best-first. Map to a relevance score where higher always means more similar.
        private T ToScore(float distance) => _annMetric == AnnPrimitives.MetricL2
            ? _numOps.FromDouble(1.0 / (1.0 + distance))
            : _numOps.FromDouble(distance);

        /// <summary>JSON DTO for <see cref="Save(string)"/> / <see cref="Load(string)"/>.</summary>
        private sealed class PersistedState
        {
            public int Version { get; set; }
            public int IndexType { get; set; }
            public int Metric { get; set; }
            public int Dimension { get; set; }
            public int Nlist { get; set; }
            public int Nprobe { get; set; }
            public int M { get; set; }
            public int Ksub { get; set; }
            public int Seed { get; set; }
            public List<string>? Order { get; set; }
            public Dictionary<string, float[]>? Vectors { get; set; }
        }
    }
}
