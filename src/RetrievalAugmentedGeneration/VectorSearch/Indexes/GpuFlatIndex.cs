using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// Flat (brute-force) index that offloads the query-vs-all similarity computation to the GPU
    /// when a device is available, falling back to an exact CPU scan otherwise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Like <see cref="FlatIndex{T}"/> this index compares the query against every stored vector and
    /// returns exact nearest neighbors, but the scoring step is executed as a single batched GPU
    /// matrix multiply (see <see cref="GpuVectorScorer"/>), matching how FAISS-GPU / Milvus-GPU
    /// accelerate flat search. This makes large brute-force scans (tens of thousands to millions of
    /// vectors) dramatically faster while preserving exact results.
    /// </para>
    /// <para>
    /// <b>Fallback.</b> The GPU path is used only when: the GPU is available and not disabled via
    /// <c>AIDOTNET_DISABLE_GPU</c>; the number of stored vectors is at or above
    /// <see cref="GpuThreshold"/>; and the metric has a GPU closed form (dot-product, cosine,
    /// Euclidean). In every other case — and if any GPU error occurs — the index performs the exact
    /// same CPU brute-force scan as <see cref="FlatIndex{T}"/>, producing identical top-k ordering.
    /// </para>
    /// <para>
    /// Because the GPU path falls back to CPU when no device is present, this type is safe to use in
    /// CI and on all target frameworks; it never throws when the GPU is absent.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.VectorIndex)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class GpuFlatIndex<T> : IVectorIndex<T>
    {
        private readonly Dictionary<string, Vector<T>> _vectors;
        private readonly ISimilarityMetric<T> _metric;

        /// <summary>
        /// The minimum number of stored vectors before the GPU path is attempted. Below this the
        /// CPU scan is used (the host/device transfer overhead is not worth it for small sets).
        /// </summary>
        public int GpuThreshold { get; }

        /// <inheritdoc/>
        public int Count => _vectors.Count;

        /// <summary>
        /// Initializes a new instance of the <see cref="GpuFlatIndex{T}"/> class.
        /// </summary>
        /// <param name="metric">The similarity metric to use for search.</param>
        /// <param name="gpuThreshold">
        /// Minimum number of vectors before the GPU path is used. Defaults to
        /// <see cref="GpuVectorScorer.DefaultGpuThreshold"/>.
        /// </param>
        public GpuFlatIndex(ISimilarityMetric<T> metric, int gpuThreshold = GpuVectorScorer.DefaultGpuThreshold)
        {
            Guard.NotNull(metric);
            if (gpuThreshold < 1)
                throw new ArgumentException("GPU threshold must be positive", nameof(gpuThreshold));

            _metric = metric;
            GpuThreshold = gpuThreshold;
            _vectors = new Dictionary<string, Vector<T>>();
        }

        /// <inheritdoc/>
        public void Add(string id, Vector<T> vector)
        {
            if (string.IsNullOrEmpty(id))
                throw new ArgumentException("ID cannot be null or empty", nameof(id));
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));

            _vectors[id] = vector;
        }

        /// <inheritdoc/>
        public void AddBatch(Dictionary<string, Vector<T>> vectors)
        {
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));

            foreach (var kvp in vectors)
            {
                Add(kvp.Key, kvp.Value);
            }
        }

        /// <inheritdoc/>
        public List<(string Id, T Score)> Search(Vector<T> query, int k)
        {
            if (query == null)
                throw new ArgumentNullException(nameof(query));
            if (k <= 0)
                throw new ArgumentException("k must be positive", nameof(k));

            // Snapshot ids and vectors in a single consistent enumeration order so the GPU and CPU
            // branches produce identically-ordered (id, score) pairs.
            int n = _vectors.Count;
            var ids = new List<string>(n);
            var vecs = new List<Vector<T>>(n);
            foreach (var kvp in _vectors)
            {
                ids.Add(kvp.Key);
                vecs.Add(kvp.Value);
            }

            var scores = new List<(string Id, T Score)>(n);

            // Attempt the GPU batched scoring; on any signal to fall back, use the exact CPU scan.
            if (GpuVectorScorer.TryScoreBatch(_metric, query, vecs, GpuThreshold, out var gpuScores)
                && gpuScores.Length == n)
            {
                for (int i = 0; i < n; i++)
                {
                    scores.Add((ids[i], gpuScores[i]));
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var score = _metric.Calculate(query, vecs[i]);
                    scores.Add((ids[i], score));
                }
            }

            // Sort based on whether higher or lower is better (identical to FlatIndex).
            var sorted = _metric.HigherIsBetter
                ? scores.OrderByDescending(x => x.Score)
                : scores.OrderBy(x => x.Score);

            return sorted.Take(Math.Min(k, scores.Count)).ToList();
        }

        /// <inheritdoc/>
        public bool Remove(string id)
        {
            return _vectors.Remove(id);
        }

        /// <inheritdoc/>
        public void Clear()
        {
            _vectors.Clear();
        }
    }
}
