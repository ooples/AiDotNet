using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    public class QuantizedFlatIndexTests
    {
        private const int Dim = 8;

        private static readonly double[][] Centers =
        {
            new double[] { 10, 10, 10, 10,  0,  0,  0,  0 },
            new double[] {  0,  0,  0,  0, 10, 10, 10, 10 },
            new double[] {-10,-10,-10,-10, 10, 10, 10, 10 },
            new double[] { 10, 10, 10, 10, 10, 10, 10, 10 },
        };

        private static Dictionary<string, Vector<double>> BuildClusteredData(int perCluster, int seed)
        {
            var rng = RandomHelper.CreateSeededRandom(seed);
            var data = new Dictionary<string, Vector<double>>();
            int id = 0;
            foreach (var center in Centers)
            {
                for (int n = 0; n < perCluster; n++)
                {
                    var arr = new double[Dim];
                    for (int i = 0; i < Dim; i++)
                        arr[i] = center[i] + (rng.NextDouble() - 0.5);
                    data[$"v{id++}"] = new Vector<double>(arr);
                }
            }

            return data;
        }

        [Fact]
        public void Constructor_WithNullArgs_Throws()
        {
            var metric = new EuclideanDistanceMetric<double>();
            var quantizer = new ScalarQuantizer<double>();
            Assert.Throws<ArgumentNullException>(() => new QuantizedFlatIndex<double>(null!, quantizer));
            Assert.Throws<ArgumentNullException>(() => new QuantizedFlatIndex<double>(metric, null!));
        }

        [Fact]
        public void UntrainedIndex_FallsBackToExactSearch()
        {
            var metric = new EuclideanDistanceMetric<double>();
            var index = new QuantizedFlatIndex<double>(metric, new ProductQuantizer<double>(subspaceCount: 2));

            var data = BuildClusteredData(5, 1);
            index.AddBatch(data);

            Assert.False(index.IsTrained);
            Assert.Equal(20, index.Count);

            // Query exactly equal to an existing vector -> that vector is the top-1 (distance 0).
            var target = data["v3"];
            var results = index.Search(target, 1);
            Assert.Single(results);
            Assert.Equal("v3", results[0].Id);
        }

        [Fact]
        public void Trained_ProductQuantizer_MatchesFlatIndexTopK()
        {
            var data = BuildClusteredData(15, 2);

            var flat = new FlatIndex<double>(new EuclideanDistanceMetric<double>());
            flat.AddBatch(data);

            var quantized = new QuantizedFlatIndex<double>(
                new EuclideanDistanceMetric<double>(),
                new ProductQuantizer<double>(subspaceCount: 2, centroidsPerSubspace: 64, maxIterations: 25, seed: 42));
            quantized.AddBatch(data);
            quantized.Train();

            Assert.True(quantized.IsTrained);
            Assert.Equal(data.Count, quantized.Count);

            var rng = RandomHelper.CreateSeededRandom(555);
            for (int trial = 0; trial < 10; trial++)
            {
                var center = Centers[trial % Centers.Length];
                var q = new double[Dim];
                for (int i = 0; i < Dim; i++)
                    q[i] = center[i] + (rng.NextDouble() - 0.5);
                var query = new Vector<double>(q);

                var flatTop = flat.Search(query, 3).Select(r => r.Id).ToList();
                var quantTop = quantized.Search(query, 3).Select(r => r.Id).ToList();

                // Top-1 must agree on well-separated clusters.
                Assert.Equal(flatTop[0], quantTop[0]);
            }
        }

        [Fact]
        public void Trained_ScalarQuantizer_ReturnsCorrectTopKViaReconstruction()
        {
            var data = BuildClusteredData(15, 4);

            var flat = new FlatIndex<double>(new CosineSimilarityMetric<double>());
            flat.AddBatch(data);

            var quantized = new QuantizedFlatIndex<double>(
                new CosineSimilarityMetric<double>(),
                new ScalarQuantizer<double>());
            quantized.AddBatch(data);
            quantized.Train();

            var query = data["v0"];
            var flatTop = flat.Search(query, 3).Select(r => r.Id).ToList();
            var quantTop = quantized.Search(query, 3).Select(r => r.Id).ToList();

            Assert.Equal(flatTop[0], quantTop[0]);
        }

        [Fact]
        public void AddAfterTraining_EncodesImmediately()
        {
            var data = BuildClusteredData(10, 6);
            var index = new QuantizedFlatIndex<double>(
                new EuclideanDistanceMetric<double>(),
                new ScalarQuantizer<double>());
            index.AddBatch(data);
            index.Train();

            int before = index.Count;
            var extra = new double[Dim];
            Array.Copy(Centers[0], extra, Dim);
            index.Add("extra", new Vector<double>(extra));

            Assert.Equal(before + 1, index.Count);
            var results = index.Search(new Vector<double>(extra), 1);
            Assert.Equal("extra", results[0].Id);
        }

        [Fact]
        public void RemoveAndClear_Work()
        {
            var data = BuildClusteredData(5, 8);
            var index = new QuantizedFlatIndex<double>(
                new EuclideanDistanceMetric<double>(),
                new ScalarQuantizer<double>());
            index.AddBatch(data);

            Assert.True(index.Remove("v0"));
            Assert.False(index.Remove("does-not-exist"));
            Assert.Equal(19, index.Count);

            index.Train();
            Assert.True(index.Remove("v1"));
            Assert.Equal(18, index.Count);

            index.Clear();
            Assert.Equal(0, index.Count);
        }

        [Fact]
        public void Train_OnEmptyIndex_Throws()
        {
            var index = new QuantizedFlatIndex<double>(
                new EuclideanDistanceMetric<double>(),
                new ScalarQuantizer<double>());
            Assert.Throws<InvalidOperationException>(() => index.Train());
        }
    }
}
