using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch
{
    /// <summary>
    /// GPU-accelerated batched similarity scoring for vector search.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This helper computes the similarity/distance between a single query vector and a large
    /// batch of stored vectors in a single GPU matrix multiply, mirroring how FAISS-GPU /
    /// Milvus-GPU accelerate brute-force ("flat") search. All stored vectors are packed into an
    /// <c>[N, dim]</c> row-major matrix, the query is uploaded as a <c>[dim, 1]</c> column, and a
    /// single <c>Gemm</c> produces the <c>[N, 1]</c> dot-product vector. Metric-specific closed
    /// forms then derive the final score from the dot products and precomputed L2 norms.
    /// </para>
    /// <para>
    /// The GPU abstraction reused is the same one the neural-network layers use:
    /// <see cref="AiDotNetEngine.Current"/> cast to
    /// <see cref="DirectGpuTensorEngine"/>, its <c>IsGpuAvailable</c> probe, and the
    /// <see cref="IDirectGpuBackend"/> returned by <c>GetBackend()</c> (<c>AllocateBuffer</c>,
    /// <c>Gemm</c>, <c>DownloadBuffer</c>). No raw CUDA/OpenCL is written here.
    /// </para>
    /// <para><b>Fallback contract.</b> <see cref="TryScoreBatch{T}"/> returns <c>false</c> — signalling
    /// the caller to use its existing CPU path — whenever any of the following hold:
    /// the <c>AIDOTNET_DISABLE_GPU</c> environment variable is set; no GPU engine/backend is
    /// available; the batch is smaller than the GPU threshold (small batches are faster on the
    /// CPU than the upload/compute/download round-trip); the metric has no GPU closed form
    /// (only dot-product, cosine and Euclidean are accelerated); vector dimensions are
    /// inconsistent; or any GPU error occurs. In every fallback case the caller's CPU brute-force
    /// path runs, so results are always correct and CI (which sets <c>AIDOTNET_DISABLE_GPU</c>)
    /// exercises the CPU path deterministically.</para>
    /// <para>
    /// The types used here (<see cref="DirectGpuTensorEngine"/>, <see cref="IDirectGpuBackend"/>)
    /// ship in the AiDotNet.Tensors package for every target framework (net10.0, net8.0, net471),
    /// so this file compiles on all TFMs; the GPU path simply stays dormant at runtime when no
    /// device is present.
    /// </para>
    /// </remarks>
    public static class GpuVectorScorer
    {
        /// <summary>
        /// Default minimum number of stored vectors before the GPU path is attempted. Below this
        /// the CPU brute-force path is faster (no host/device transfer overhead).
        /// </summary>
        public const int DefaultGpuThreshold = 1024;

        private enum MetricKind
        {
            Unsupported,
            DotProduct,
            Cosine,
            Euclidean
        }

        /// <summary>
        /// Returns true when the <c>AIDOTNET_DISABLE_GPU</c> environment variable is set to a
        /// non-empty, non-"0"/"false" value. Mirrors the gate honored by the rest of the codebase.
        /// </summary>
        public static bool IsGpuDisabledByEnvironment
        {
            get
            {
                var value = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU");
                if (string.IsNullOrEmpty(value))
                    return false;
                return !string.Equals(value, "0", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(value, "false", StringComparison.OrdinalIgnoreCase);
            }
        }

        /// <summary>
        /// Returns true when a GPU tensor engine is present and enabled (and not disabled via the
        /// environment variable). This is the same probe used by the neural-network / diffusion code.
        /// </summary>
        public static bool IsGpuAvailable()
        {
            if (IsGpuDisabledByEnvironment)
                return false;

            try
            {
                var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
                return engine != null && engine.IsGpuAvailable;
            }
            catch
            {
                // Any failure probing the engine is treated as "no GPU" so callers fall back to CPU.
                return false;
            }
        }

        private static MetricKind GetKind<T>(ISimilarityMetric<T> metric)
        {
            switch (metric)
            {
                case DotProductMetric<T>:
                    return MetricKind.DotProduct;
                case CosineSimilarityMetric<T>:
                    return MetricKind.Cosine;
                case EuclideanDistanceMetric<T>:
                    return MetricKind.Euclidean;
                default:
                    return MetricKind.Unsupported;
            }
        }

        /// <summary>
        /// Returns true when a batch of the given size with the given metric would be routed to the
        /// GPU (metric supported, batch at/above threshold, and a GPU device available). Useful for
        /// callers that want to decide before materializing a candidate list.
        /// </summary>
        public static bool CanAccelerate<T>(ISimilarityMetric<T> metric, int count, int gpuThreshold)
        {
            if (metric == null)
                return false;
            if (count < gpuThreshold)
                return false;
            if (GetKind(metric) == MetricKind.Unsupported)
                return false;
            return IsGpuAvailable();
        }

        /// <summary>
        /// Scores <paramref name="query"/> against every vector in <paramref name="docs"/> on the GPU
        /// in a single batched matrix multiply, producing one score per document in the same order as
        /// <paramref name="docs"/> and using the same semantics as the CPU metric.
        /// </summary>
        /// <typeparam name="T">The numeric type.</typeparam>
        /// <param name="metric">The similarity metric (only dot-product, cosine, Euclidean are accelerated).</param>
        /// <param name="query">The query vector.</param>
        /// <param name="docs">The stored vectors to score, in caller order.</param>
        /// <param name="gpuThreshold">Minimum batch size before the GPU path is attempted.</param>
        /// <param name="scores">On success, one score per doc aligned to <paramref name="docs"/>; otherwise empty.</param>
        /// <returns>
        /// True when the scores were produced on the GPU; false when the caller must fall back to its
        /// CPU path (GPU disabled/absent, metric unsupported, batch too small, or a GPU error).
        /// </returns>
        public static bool TryScoreBatch<T>(
            ISimilarityMetric<T> metric,
            Vector<T> query,
            IReadOnlyList<Vector<T>> docs,
            int gpuThreshold,
            out T[] scores)
        {
            scores = Array.Empty<T>();

            if (metric == null || query == null || docs == null)
                return false;

            var kind = GetKind(metric);
            if (kind == MetricKind.Unsupported)
                return false;

            int n = docs.Count;
            if (n < gpuThreshold || n == 0)
                return false;

            int dim = query.Length;
            if (dim == 0)
                return false;

            // Guard against int overflow of the flat matrix size; fall back to CPU for extreme sizes.
            long totalLong = (long)n * dim;
            if (totalLong > int.MaxValue)
                return false;

            if (!IsGpuAvailable())
                return false;

            for (int i = 0; i < n; i++)
            {
                if (docs[i] == null || docs[i].Length != dim)
                    return false; // inconsistent dimensions -> CPU fallback
            }

            var engine = AiDotNetEngine.Current;
            var matrix = engine.TensorStack(docs.Select(v => Tensor<T>.FromVector(v)).ToArray(), 0);
            var queryRow = Tensor<T>.FromVector(query);

            Tensor<T> result;
            switch (kind)
            {
                case MetricKind.DotProduct:
                    result = engine.TensorMatMul(matrix, engine.Reshape(queryRow, [dim, 1]));
                    break;

                case MetricKind.Cosine:
                    // Unit rows (zero rows stay zero, giving similarity 0).
                    result = engine.TensorMatMul(UnitRows(engine, matrix), engine.Reshape(UnitRows(engine, engine.Reshape(queryRow, [1, dim])), [dim, 1]));
                    break;

                case MetricKind.Euclidean:
                    // ||q - d|| directly (no ||q||^2 + ||d||^2 - 2 q.d cancellation), scaled by the largest
                    // magnitude before squaring.
                    result = RowNorms(engine, engine.TensorSubtract(matrix, engine.TensorBroadcastTo(engine.Reshape(queryRow, [1, dim]), [n, dim])));
                    break;

                default:
                    return false;
            }

            scores = result.ToArray();
            return true;
        }

        /// <summary>Euclidean norm of every row, computed on rows pre-divided by their largest magnitude.</summary>
        private static Tensor<T> RowNorms<T>(IEngine engine, Tensor<T> matrix)
        {
            var scale = NonZero(engine, engine.ReduceMax(engine.TensorAbs(matrix), [1], true));
            var scaled = engine.TensorDivide(matrix, engine.TensorBroadcastTo(scale, matrix._shape));
            var norms = engine.TensorMultiply(
                engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(scaled), [1], true)), scale);
            return engine.Reshape(norms, [matrix._shape[0]]);
        }

        /// <summary>Each row divided by its Euclidean norm; zero rows stay zero.</summary>
        private static Tensor<T> UnitRows<T>(IEngine engine, Tensor<T> matrix)
        {
            var norms = NonZero(engine, engine.Reshape(RowNorms(engine, matrix), [matrix._shape[0], 1]));
            return engine.TensorDivide(matrix, engine.TensorBroadcastTo(norms, matrix._shape));
        }

        /// <summary>Replaces zero entries with one so dividing a zero row by them leaves it zero.</summary>
        private static Tensor<T> NonZero<T>(IEngine engine, Tensor<T> values)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            return engine.TensorWhere(
                engine.TensorGreaterThan(values, numOps.Zero), values, Tensor<T>.CreateDefault(values._shape, numOps.One));
        }
    }
}
