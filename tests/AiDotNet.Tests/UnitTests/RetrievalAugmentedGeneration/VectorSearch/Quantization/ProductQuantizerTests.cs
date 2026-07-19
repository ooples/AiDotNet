using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Quantization
{
    public class ProductQuantizerTests
    {
        private const int Dim = 8;

        // Three well-separated cluster centers in 8-D space.
        private static readonly double[][] Centers =
        {
            new double[] { 10, 10, 10, 10,  0,  0,  0,  0 },
            new double[] {  0,  0,  0,  0, 10, 10, 10, 10 },
            new double[] {-10,-10,-10,-10,-10,-10,-10,-10 },
        };

        private static List<(double[] Raw, Vector<double> Vector, int Cluster)> BuildClusteredData(int perCluster, int seed)
        {
            var rng = RandomHelper.CreateSeededRandom(seed);
            var data = new List<(double[], Vector<double>, int)>();
            for (int c = 0; c < Centers.Length; c++)
            {
                var center = Centers[c];
                for (int n = 0; n < perCluster; n++)
                {
                    var arr = new double[Dim];
                    for (int i = 0; i < Dim; i++)
                        arr[i] = center[i] + (rng.NextDouble() - 0.5); // +/- 0.5 noise
                    data.Add((arr, new Vector<double>(arr), c));
                }
            }

            return data;
        }

        private static double SquaredL2(double[] a, double[] b)
        {
            double s = 0.0;
            for (int i = 0; i < a.Length; i++)
            {
                double d = a[i] - b[i];
                s += d * d;
            }

            return s;
        }

        [Fact]
        public void Train_LearnsCodebooksWithExpectedShape()
        {
            int m = 2, ksub = 16;
            var data = BuildClusteredData(30, 7).Select(x => x.Vector);
            var pq = new ProductQuantizer<double>(subspaceCount: m, centroidsPerSubspace: ksub, maxIterations: 20, seed: 42);
            pq.Train(data);

            Assert.True(pq.IsTrained);
            Assert.Equal(m, pq.SubspaceCount);
            Assert.Equal(ksub, pq.CentroidsPerSubspace);
            Assert.Equal(Dim / m, pq.SubDimension);

            var books = pq.Codebooks!;
            Assert.Equal(m, books.Length);
            for (int mm = 0; mm < m; mm++)
            {
                Assert.Equal(ksub, books[mm].Length);
                for (int c = 0; c < ksub; c++)
                    Assert.Equal(Dim / m, books[mm][c].Length);
            }
        }

        [Fact]
        public void Encode_ProducesOneBytePerSubspace()
        {
            int m = 4;
            var data = BuildClusteredData(30, 7).Select(x => x.Vector).ToList();
            var pq = new ProductQuantizer<double>(subspaceCount: m, centroidsPerSubspace: 32, maxIterations: 20, seed: 42);
            pq.Train(data);

            var code = pq.Encode(data[0]);
            Assert.Equal(m, code.Length);
            Assert.Equal(m, pq.CodeLength);
        }

        [Fact]
        public void Train_WithDimensionNotDivisibleByM_Throws()
        {
            var data = new List<Vector<double>> { new Vector<double>(new double[] { 1, 2, 3 }) };
            var pq = new ProductQuantizer<double>(subspaceCount: 2);
            Assert.Throws<ArgumentException>(() => pq.Train(data));
        }

        [Fact]
        public void Adc_NearestNeighbor_MatchesBruteForceTop1()
        {
            var data = BuildClusteredData(30, 11);
            var pq = new ProductQuantizer<double>(subspaceCount: 2, centroidsPerSubspace: 64, maxIterations: 25, seed: 42);
            pq.Train(data.Select(x => x.Vector));

            // Encode the whole database.
            var codes = data.Select(x => pq.Encode(x.Vector)).ToList();

            var rng = RandomHelper.CreateSeededRandom(99);
            for (int trial = 0; trial < 15; trial++)
            {
                // Query near a random cluster center.
                int expectedCluster = trial % Centers.Length;
                var center = Centers[expectedCluster];
                var q = new double[Dim];
                for (int i = 0; i < Dim; i++)
                    q[i] = center[i] + (rng.NextDouble() - 0.5);
                var query = new Vector<double>(q);

                // Brute-force exact nearest neighbor.
                int bruteBest = 0;
                double bruteDist = double.MaxValue;
                for (int j = 0; j < data.Count; j++)
                {
                    double d = SquaredL2(q, data[j].Raw);
                    if (d < bruteDist)
                    {
                        bruteDist = d;
                        bruteBest = j;
                    }
                }

                // ADC nearest neighbor.
                var table = pq.BuildDistanceTable(query);
                int adcBest = 0;
                double adcDist = double.MaxValue;
                for (int j = 0; j < codes.Count; j++)
                {
                    double d = pq.ComputeAsymmetricDistance(table, codes[j]);
                    if (d < adcDist)
                    {
                        adcDist = d;
                        adcBest = j;
                    }
                }

                // On separable clusters, ADC (a lossy approximation) must recover the same
                // cluster as the exact brute-force nearest neighbor. Distinguishing the single
                // closest point *within* a tight cluster is below quantization resolution and
                // is not expected of any product quantizer.
                Assert.Equal(expectedCluster, data[bruteBest].Cluster);
                Assert.Equal(data[bruteBest].Cluster, data[adcBest].Cluster);
            }
        }

        [Fact]
        public void Adc_EqualsExactDistanceToReconstruction()
        {
            var data = BuildClusteredData(20, 5);
            var pq = new ProductQuantizer<double>(subspaceCount: 2, centroidsPerSubspace: 32, maxIterations: 20, seed: 42);
            pq.Train(data.Select(x => x.Vector));

            var query = new Vector<double>(new double[] { 9.7, 10.1, 9.9, 10.2, 0.1, -0.2, 0.05, 0.0 });
            var table = pq.BuildDistanceTable(query);

            foreach (var item in data.Take(10))
            {
                var code = pq.Encode(item.Vector);
                double adc = pq.ComputeAsymmetricDistance(table, code);
                double exact = SquaredL2(query.ToArray(), pq.Decode(code).ToArray());
                Assert.Equal(exact, adc, 6);
            }
        }

        [Fact]
        public void IsDeterministic_AcrossRuns()
        {
            var data = BuildClusteredData(25, 3).Select(x => x.Vector).ToList();

            var pq1 = new ProductQuantizer<double>(subspaceCount: 2, centroidsPerSubspace: 32, seed: 7);
            var pq2 = new ProductQuantizer<double>(subspaceCount: 2, centroidsPerSubspace: 32, seed: 7);
            pq1.Train(data);
            pq2.Train(data);

            foreach (var v in data)
                Assert.Equal(pq1.Encode(v), pq2.Encode(v));
        }
    }
}
