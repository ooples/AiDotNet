using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    /// <summary>
    /// Tests for <see cref="AnnVectorIndex{T}"/> — the dependency-free native ANN index (Flat/IVF/PQ/IVFPQ) on the
    /// AiDotNet Tensors stack that replaces the external FaissNet backend. Validated on separable synthetic
    /// clusters so the expected nearest neighbour is unambiguous, plus the incremental Add/Remove/Clear contract.
    /// </summary>
    public class AnnVectorIndexTests
    {
        private const int Dim = 16;
        private const int Clusters = 8;
        private const int PerCluster = 40;

        // Deterministic well-separated clusters: cluster c centred at a far-apart lattice point.
        private static (List<(string id, Vector<double> v)> data, Vector<double>[] centers) MakeData(int seed)
        {
            var rng = new Random(seed);
            var data = new List<(string, Vector<double>)>();
            var centers = new Vector<double>[Clusters];
            for (int c = 0; c < Clusters; c++)
            {
                var center = new double[Dim];
                for (int d = 0; d < Dim; d++) center[d] = c * 50.0 + d;
                centers[c] = new Vector<double>(center);
                for (int p = 0; p < PerCluster; p++)
                {
                    var v = new double[Dim];
                    for (int d = 0; d < Dim; d++) v[d] = center[d] + (rng.NextDouble() - 0.5) * 2.0;
                    data.Add(($"c{c}_p{p}", new Vector<double>(v)));
                }
            }
            return (data, centers);
        }

        private static int ClusterOf(string id) => int.Parse(id.Substring(1, id.IndexOf('_') - 1));

        [Theory]
        [InlineData(AnnVectorIndexType.Flat)]
        [InlineData(AnnVectorIndexType.Ivf)]
        [InlineData(AnnVectorIndexType.Pq)]
        [InlineData(AnnVectorIndexType.IvfPq)]
        public void Search_ReturnsNeighboursFromTheQueryCluster(AnnVectorIndexType type)
        {
            var (data, centers) = MakeData(seed: 7);
            var index = new AnnVectorIndex<double>(type, Dim, AnnVectorMetric.L2, nlist: Clusters, nprobe: 3, m: 4, ksub: 32);
            foreach (var (id, v) in data) index.Add(id, v);

            Assert.Equal(data.Count, index.Count);

            int target = 5;
            var hits = index.Search(centers[target], 5);
            Assert.NotEmpty(hits);
            Assert.Equal(target, ClusterOf(hits[0].Id));                       // exact/approx top-1 in the query cluster
            Assert.True(hits.Count(h => ClusterOf(h.Id) == target) >= 3);      // majority of top-5 in-cluster
        }

        [Fact]
        public void Flat_ReturnsStoredVectorAsItsOwnNearestNeighbour()
        {
            var (data, _) = MakeData(seed: 11);
            var index = new AnnVectorIndex<double>(AnnVectorIndexType.Flat, Dim, AnnVectorMetric.L2);
            foreach (var (id, v) in data) index.Add(id, v);

            var (probeId, probeVec) = data[123];
            var hits = index.Search(probeVec, 1);
            Assert.Single(hits);
            Assert.Equal(probeId, hits[0].Id);
        }

        [Fact]
        public void Remove_ExcludesTheDocumentAndDecrementsCount()
        {
            var (data, _) = MakeData(seed: 3);
            var index = new AnnVectorIndex<double>(AnnVectorIndexType.Flat, Dim, AnnVectorMetric.L2);
            foreach (var (id, v) in data) index.Add(id, v);

            var (probeId, probeVec) = data[200];
            Assert.True(index.Remove(probeId));
            Assert.Equal(data.Count - 1, index.Count);
            Assert.False(index.Remove(probeId)); // idempotent

            var hits = index.Search(probeVec, 5);
            Assert.DoesNotContain(hits, h => h.Id == probeId);
        }

        [Fact]
        public void Clear_EmptiesTheIndex()
        {
            var (data, _) = MakeData(seed: 1);
            var index = new AnnVectorIndex<double>(AnnVectorIndexType.Ivf, Dim, AnnVectorMetric.L2, nlist: Clusters);
            foreach (var (id, v) in data) index.Add(id, v);
            Assert.True(index.Count > 0);

            index.Clear();
            Assert.Equal(0, index.Count);
            Assert.Empty(index.Search(data[0].v, 5));
        }

        [Fact]
        public void Cosine_NormalizesSoDirectionWinsOverMagnitude()
        {
            var index = new AnnVectorIndex<double>(AnnVectorIndexType.Flat, dimension: 3, metric: AnnVectorMetric.Cosine);
            index.Add("a", new Vector<double>(new[] { 1.0, 0.0, 0.0 }));
            index.Add("b", new Vector<double>(new[] { 0.0, 1.0, 0.0 }));

            // Query points along +x with a large magnitude; cosine must pick "a" regardless of length.
            var hits = index.Search(new Vector<double>(new[] { 100.0, 1.0, 0.0 }), 1);
            Assert.Single(hits);
            Assert.Equal("a", hits[0].Id);
        }

        [Fact]
        public void AddBatch_AddsAllVectors()
        {
            var (data, _) = MakeData(seed: 9);
            var index = new AnnVectorIndex<double>(AnnVectorIndexType.Flat, Dim, AnnVectorMetric.L2);
            index.AddBatch(data.ToDictionary(d => d.id, d => d.v));
            Assert.Equal(data.Count, index.Count);
        }
    }
}
