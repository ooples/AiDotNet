using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// Tests for <see cref="GpuFlatIndex{T}"/>, the batched-GPU brute-force index, and the
    /// <see cref="GpuVectorScorer"/> helper it relies on.
    /// </summary>
    /// <remarks>
    /// In CI the GPU is disabled (the test assembly sets <c>AIDOTNET_DISABLE_GPU=1</c>), so the GPU
    /// path deterministically falls back to the CPU brute-force scan. These tests therefore assert
    /// two things: (1) the GPU index produces the exact same top-k as the reference CPU
    /// <see cref="FlatIndex{T}"/> on identical data, and (2) the code never throws when no GPU is
    /// present. Both are the properties that matter for correctness on any machine — on a real GPU
    /// the scores match within floating-point tolerance and the ordering is preserved.
    /// </remarks>
    public class GpuFlatIndexTests
    {
        private const int Seed = 12345;

        private static List<(string Id, Vector<double>)> GenerateVectors(int count, int dim, int seed)
        {
            var rng = new Random(seed);
            var list = new List<(string, Vector<double>)>(count);
            for (int i = 0; i < count; i++)
            {
                var data = new double[dim];
                for (int j = 0; j < dim; j++)
                {
                    // Non-zero, well-separated values so norms are never degenerate.
                    data[j] = rng.NextDouble() * 2.0 - 1.0 + 0.001;
                }
                list.Add(($"vec{i}", new Vector<double>(data)));
            }
            return list;
        }

        private static void PopulateBoth(
            IVectorIndex<double> a,
            IVectorIndex<double> b,
            IReadOnlyList<(string Id, Vector<double> Vec)> data)
        {
            foreach (var (id, vec) in data)
            {
                a.Add(id, vec);
                b.Add(id, vec);
            }
        }

        private static void AssertTopKEqual(
            List<(string Id, double Score)> expected,
            List<(string Id, double Score)> actual)
        {
            Assert.Equal(expected.Count, actual.Count);
            for (int i = 0; i < expected.Count; i++)
            {
                Assert.Equal(expected[i].Id, actual[i].Id);
                Assert.Equal(expected[i].Score, actual[i].Score, 9);
            }
        }

        // ---- Constructor validation ----

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullMetric_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new GpuFlatIndex<double>(null!));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNonPositiveThreshold_Throws()
        {
            var metric = new CosineSimilarityMetric<double>();
            Assert.Throws<ArgumentException>(() => new GpuFlatIndex<double>(metric, gpuThreshold: 0));
        }

        // ---- Correctness vs CPU brute force, across supported metrics ----

        [Theory(Timeout = 120000)]
        [InlineData("cosine")]
        [InlineData("dot")]
        [InlineData("euclidean")]
        public async Task Search_MatchesCpuFlatIndex_AboveThreshold(string metricName)
        {
            // Large batch that crosses the default GPU threshold so the GPU path is attempted
            // (and, in CI with no GPU, falls back to CPU).
            var data = GenerateVectors(count: 2500, dim: 16, seed: Seed);

            var cpu = new FlatIndex<double>(MakeMetric(metricName));
            var gpu = new GpuFlatIndex<double>(MakeMetric(metricName));
            PopulateBoth(cpu, gpu, data);

            Assert.True(gpu.Count >= GpuVectorScorer.DefaultGpuThreshold);

            var query = data[7].Item2;
            var expected = cpu.Search(query, 10);
            var actual = gpu.Search(query, 10);

            AssertTopKEqual(expected, actual);
        }

        [Theory(Timeout = 60000)]
        [InlineData("cosine")]
        [InlineData("dot")]
        [InlineData("euclidean")]
        public async Task Search_MatchesCpuFlatIndex_BelowThreshold(string metricName)
        {
            // Small batch: GPU path is skipped by the threshold, CPU scan is used directly.
            var data = GenerateVectors(count: 50, dim: 8, seed: Seed + 1);

            var cpu = new FlatIndex<double>(MakeMetric(metricName));
            var gpu = new GpuFlatIndex<double>(MakeMetric(metricName));
            PopulateBoth(cpu, gpu, data);

            var query = new Vector<double>(new double[] { 0.5, -0.2, 0.9, 0.1, -0.7, 0.3, 0.6, -0.4 });
            var expected = cpu.Search(query, 5);
            var actual = gpu.Search(query, 5);

            AssertTopKEqual(expected, actual);
        }

        [Theory(Timeout = 120000)]
        [InlineData("cosine")]
        [InlineData("dot")]
        [InlineData("euclidean")]
        public async Task Search_MatchesCpuFlatIndex_ForcedGpuThreshold(string metricName)
        {
            // Threshold of 1 forces the GPU attempt for every query; with GPU disabled the scorer
            // signals fallback and the CPU scan runs — results must still equal the reference.
            var data = GenerateVectors(count: 300, dim: 12, seed: Seed + 2);

            var cpu = new FlatIndex<double>(MakeMetric(metricName));
            var gpu = new GpuFlatIndex<double>(MakeMetric(metricName), gpuThreshold: 1);
            PopulateBoth(cpu, gpu, data);

            var query = data[42].Item2;
            var expected = cpu.Search(query, 20);
            var actual = gpu.Search(query, 20);

            AssertTopKEqual(expected, actual);
        }

        [Fact(Timeout = 60000)]
        public async Task Search_WithUnsupportedMetric_FallsBackAndMatchesCpu()
        {
            // Manhattan has no GPU closed form -> scorer signals fallback regardless of batch size.
            var data = GenerateVectors(count: 1500, dim: 10, seed: Seed + 3);

            var cpu = new FlatIndex<double>(new ManhattanDistanceMetric<double>());
            var gpu = new GpuFlatIndex<double>(new ManhattanDistanceMetric<double>(), gpuThreshold: 1);
            PopulateBoth(cpu, gpu, data);

            var query = data[3].Item2;
            var expected = cpu.Search(query, 7);
            var actual = gpu.Search(query, 7);

            AssertTopKEqual(expected, actual);
        }

        // ---- Robustness / edge cases: must never throw when GPU absent ----

        [Fact(Timeout = 60000)]
        public async Task Search_OnEmptyIndex_ReturnsEmpty()
        {
            var gpu = new GpuFlatIndex<double>(new CosineSimilarityMetric<double>(), gpuThreshold: 1);
            var result = gpu.Search(new Vector<double>(new double[] { 1.0, 2.0, 3.0 }), 5);
            Assert.Empty(result);
        }

        [Fact(Timeout = 60000)]
        public async Task Search_WithKLargerThanCount_ReturnsAll()
        {
            var data = GenerateVectors(count: 2000, dim: 8, seed: Seed + 4);
            var gpu = new GpuFlatIndex<double>(new DotProductMetric<double>());
            foreach (var (id, vec) in data) gpu.Add(id, vec);

            var result = gpu.Search(data[0].Item2, 5000);
            Assert.Equal(2000, result.Count);
        }

        [Fact(Timeout = 60000)]
        public async Task Search_NullQuery_Throws()
        {
            var gpu = new GpuFlatIndex<double>(new CosineSimilarityMetric<double>());
            Assert.Throws<ArgumentNullException>(() => gpu.Search(null!, 5));
        }

        [Fact(Timeout = 60000)]
        public async Task Search_NonPositiveK_Throws()
        {
            var gpu = new GpuFlatIndex<double>(new CosineSimilarityMetric<double>());
            gpu.Add("a", new Vector<double>(new double[] { 1.0, 2.0 }));
            Assert.Throws<ArgumentException>(() => gpu.Search(new Vector<double>(new double[] { 1.0, 2.0 }), 0));
        }

        [Fact(Timeout = 60000)]
        public async Task Remove_And_Clear_Work()
        {
            var gpu = new GpuFlatIndex<double>(new CosineSimilarityMetric<double>());
            gpu.Add("a", new Vector<double>(new double[] { 1.0, 2.0 }));
            gpu.Add("b", new Vector<double>(new double[] { 3.0, 4.0 }));

            Assert.True(gpu.Remove("a"));
            Assert.False(gpu.Remove("missing"));
            Assert.Equal(1, gpu.Count);

            gpu.Clear();
            Assert.Equal(0, gpu.Count);
        }

        // ---- GpuVectorScorer helper contract ----

        [Fact(Timeout = 60000)]
        public async Task Scorer_IsGpuDisabledByEnvironment_TrueInCi()
        {
            // The test assembly sets AIDOTNET_DISABLE_GPU=1, so the gate reports disabled and
            // IsGpuAvailable is false. This documents the environment gate the rest of the codebase uses.
            Assert.True(GpuVectorScorer.IsGpuDisabledByEnvironment);
            Assert.False(GpuVectorScorer.IsGpuAvailable());
        }

        [Fact(Timeout = 60000)]
        public async Task Scorer_TryScoreBatch_ReturnsFalseWhenGpuUnavailable()
        {
            var metric = new DotProductMetric<double>();
            var query = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var docs = GenerateVectors(2000, 3, Seed + 5).Select(x => x.Item2).ToList();

            // GPU disabled in CI -> must signal fallback (false) without throwing.
            bool ok = GpuVectorScorer.TryScoreBatch(metric, query, docs, gpuThreshold: 1, out var scores);
            Assert.False(ok);
            Assert.Empty(scores);
        }

        [Fact(Timeout = 60000)]
        public async Task Scorer_CanAccelerate_FalseWhenGpuDisabled()
        {
            var metric = new CosineSimilarityMetric<double>();
            Assert.False(GpuVectorScorer.CanAccelerate(metric, count: 100000, gpuThreshold: 1));
        }

        // ---- HNSW batched candidate scoring helper ----

        [Fact(Timeout = 120000)]
        public async Task Hnsw_ScoreCandidatesGpu_MatchesCpuMetricOrdering()
        {
            var metric = new CosineSimilarityMetric<double>();
            var data = GenerateVectors(count: 1500, dim: 16, seed: Seed + 6);

            var hnsw = new HNSWIndex<double>(metric);
            foreach (var (id, vec) in data) hnsw.Add(id, vec);

            var query = data[10].Item2;
            var candidateIds = data.Select(x => x.Item1).ToList();

            // GPU-batched (falls back to CPU metric in CI) ...
            var scored = hnsw.ScoreCandidatesGpu(query, candidateIds, gpuThreshold: 1);

            // ... must equal an independent CPU brute-force ranking over the same candidates.
            var expected = candidateIds
                .Select(id => (Id: id, Score: metric.Calculate(query, data.First(d => d.Item1 == id).Item2)))
                .OrderByDescending(x => x.Score)
                .ToList();

            Assert.Equal(expected.Count, scored.Count);
            for (int i = 0; i < expected.Count; i++)
            {
                Assert.Equal(expected[i].Id, scored[i].Id);
                Assert.Equal(expected[i].Score, scored[i].Score, 9);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task Hnsw_ScoreCandidatesGpu_SkipsMissingIds()
        {
            var metric = new DotProductMetric<double>();
            var hnsw = new HNSWIndex<double>(metric);
            hnsw.Add("a", new Vector<double>(new double[] { 1.0, 0.0 }));
            hnsw.Add("b", new Vector<double>(new double[] { 0.0, 1.0 }));

            var result = hnsw.ScoreCandidatesGpu(
                new Vector<double>(new double[] { 1.0, 1.0 }),
                new List<string> { "a", "missing", "b" },
                gpuThreshold: 1);

            Assert.Equal(2, result.Count);
            Assert.Contains(result, r => r.Id == "a");
            Assert.Contains(result, r => r.Id == "b");
        }

        private static ISimilarityMetric<double> MakeMetric(string name) => name switch
        {
            "cosine" => new CosineSimilarityMetric<double>(),
            "dot" => new DotProductMetric<double>(),
            "euclidean" => new EuclideanDistanceMetric<double>(),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown metric")
        };
    }
}
