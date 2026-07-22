using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// A flat (brute-force) index that stores compressed quantized codes instead of full
    /// vectors, trading a small amount of accuracy for large memory savings and faster
    /// scanning. This mirrors FAISS <c>IndexPQ</c>/<c>IndexScalarQuantizer</c> and Qdrant's
    /// on-disk quantized collections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Vectors added before the underlying quantizer is trained are held in raw form and
    /// searched with an exact fallback (equivalent to a <see cref="FlatIndex{T}"/>). Once
    /// <see cref="Train"/> is called, those vectors are encoded to compact codes, the raw
    /// copies are released, and subsequent additions are encoded immediately.
    /// </para>
    /// <para>
    /// Ranking after training uses the quantizer to score candidates: for a
    /// <see cref="ProductQuantizer{T}"/> paired with a distance metric it uses fast
    /// Asymmetric Distance Computation (ADC); otherwise it reconstructs each candidate with
    /// <see cref="IVectorQuantizer{T}.Decode"/> and applies the supplied metric.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This index keeps a tiny "summary" of each vector instead of the
    /// whole thing, so it uses far less memory. You still call Add and Search the same way;
    /// just remember to call Train once (on a representative sample) so it can learn how to
    /// summarize. Until then it behaves like an ordinary exact index.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.VectorIndex)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class QuantizedFlatIndex<T> : IVectorIndex<T>
    {
        private readonly ISimilarityMetric<T> _metric;
        private readonly IVectorQuantizer<T> _quantizer;
        private readonly INumericOperations<T> _numOps;

        // Raw vectors kept only until the quantizer is trained (then cleared to save memory).
        private readonly Dictionary<string, Vector<T>> _raw;
        // Compressed codes used once trained.
        private readonly Dictionary<string, byte[]> _codes;

        /// <inheritdoc/>
        public int Count => _raw.Count + _codes.Count;

        /// <summary>
        /// Gets a value indicating whether the underlying quantizer has been trained.
        /// </summary>
        public bool IsTrained => _quantizer.IsTrained;

        /// <summary>
        /// Gets the quantizer used to compress vectors.
        /// </summary>
        public IVectorQuantizer<T> Quantizer => _quantizer;

        /// <summary>
        /// Initializes a new instance of the <see cref="QuantizedFlatIndex{T}"/> class.
        /// </summary>
        /// <param name="metric">The similarity metric used for ranking (and for the untrained fallback).</param>
        /// <param name="quantizer">The quantizer that compresses stored vectors.</param>
        public QuantizedFlatIndex(ISimilarityMetric<T> metric, IVectorQuantizer<T> quantizer)
        {
            Guard.NotNull(metric);
            Guard.NotNull(quantizer);
            _metric = metric;
            _quantizer = quantizer;
            _numOps = MathHelper.GetNumericOperations<T>();
            _raw = new Dictionary<string, Vector<T>>();
            _codes = new Dictionary<string, byte[]>();
        }

        /// <summary>
        /// Trains the underlying quantizer, then encodes and compacts any vectors that were
        /// added beforehand.
        /// </summary>
        /// <param name="vectors">
        /// Optional training vectors. If null, the quantizer is trained on the vectors already
        /// added to this index.
        /// </param>
        public void Train(IEnumerable<Vector<T>>? vectors = null)
        {
            var trainingData = vectors != null ? vectors.ToList() : _raw.Values.ToList();
            if (trainingData.Count == 0)
                throw new InvalidOperationException("Cannot train: no training vectors provided and the index is empty.");

            _quantizer.Train(trainingData);

            // Encode and release any pending raw vectors.
            foreach (var kvp in _raw)
                _codes[kvp.Key] = _quantizer.Encode(kvp.Value);
            _raw.Clear();
        }

        /// <inheritdoc/>
        public void Add(string id, Vector<T> vector)
        {
            if (string.IsNullOrEmpty(id))
                throw new ArgumentException("ID cannot be null or empty", nameof(id));
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));

            if (_quantizer.IsTrained)
            {
                _raw.Remove(id);
                _codes[id] = _quantizer.Encode(vector);
            }
            else
            {
                _codes.Remove(id);
                _raw[id] = vector;
            }
        }

        /// <inheritdoc/>
        public void AddBatch(Dictionary<string, Vector<T>> vectors)
        {
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));

            foreach (var kvp in vectors)
                Add(kvp.Key, kvp.Value);
        }

        /// <inheritdoc/>
        public List<(string Id, T Score)> Search(Vector<T> query, int k)
        {
            if (query == null)
                throw new ArgumentNullException(nameof(query));
            if (k <= 0)
                throw new ArgumentException("k must be positive", nameof(k));

            // Untrained: exact fallback over any raw vectors (behaves like FlatIndex).
            if (!_quantizer.IsTrained)
                return RankRaw(query, k);

            if (_codes.Count == 0)
                return new List<(string Id, T Score)>();

            // Fast path: Product Quantizer + distance metric -> Asymmetric Distance Computation.
            if (_quantizer is ProductQuantizer<T> pq && !_metric.HigherIsBetter)
                return RankAdc(pq, query, k);

            // General path: reconstruct each candidate and apply the metric.
            return RankReconstructed(query, k);
        }

        /// <inheritdoc/>
        public bool Remove(string id)
        {
            bool removed = _raw.Remove(id);
            removed |= _codes.Remove(id);
            return removed;
        }

        /// <inheritdoc/>
        public void Clear()
        {
            _raw.Clear();
            _codes.Clear();
        }

        private List<(string Id, T Score)> RankRaw(Vector<T> query, int k)
        {
            if (_raw.Count == 0)
                return new List<(string Id, T Score)>();

            var scores = _raw
                .Select(kvp => (Id: kvp.Key, Score: _metric.Calculate(query, kvp.Value)))
                .ToList();

            var sorted = _metric.HigherIsBetter
                ? scores.OrderByDescending(x => x.Score)
                : scores.OrderBy(x => x.Score);

            return sorted.Take(Math.Min(k, scores.Count)).ToList();
        }

        private List<(string Id, T Score)> RankReconstructed(Vector<T> query, int k)
        {
            var scores = _codes
                .Select(kvp => (Id: kvp.Key, Score: _metric.Calculate(query, _quantizer.Decode(kvp.Value))))
                .ToList();

            var sorted = _metric.HigherIsBetter
                ? scores.OrderByDescending(x => x.Score)
                : scores.OrderBy(x => x.Score);

            return sorted.Take(Math.Min(k, scores.Count)).ToList();
        }

        private List<(string Id, T Score)> RankAdc(ProductQuantizer<T> pq, Vector<T> query, int k)
        {
            var table = pq.BuildDistanceTable(query);

            var scores = _codes
                .Select(kvp => (Id: kvp.Key, Distance: pq.ComputeAsymmetricDistance(table, kvp.Value)))
                .OrderBy(x => x.Distance)
                .Take(Math.Min(k, _codes.Count))
                .Select(x => (x.Id, Score: _numOps.FromDouble(x.Distance)))
                .ToList();

            return scores;
        }
    }
}
