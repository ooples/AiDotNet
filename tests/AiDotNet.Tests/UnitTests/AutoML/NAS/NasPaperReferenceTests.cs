using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Reference tests that pin NAS derivation and normalisation rules to their papers.
    /// </summary>
    public class NasPaperReferenceTests
    {
        private static int ArgMax(Matrix<double> m, int row, out double best)
        {
            int idx = 0;
            best = m[row, 0];
            for (int c = 1; c < m.Columns; c++)
            {
                if (m[row, c] > best)
                {
                    best = m[row, c];
                    idx = c;
                }
            }

            return idx;
        }

        private static List<(int ToNode, int FromNode, string Operation)> ExpectedTopTwo(
            IReadOnlyList<Matrix<double>> scores, IReadOnlyList<string> operations)
        {
            var expected = new List<(int, int, string)>();
            for (int node = 0; node < scores.Count; node++)
            {
                var edges = new List<(int From, int Op, double Score)>();
                for (int prev = 0; prev < scores[node].Rows; prev++)
                {
                    int op = ArgMax(scores[node], prev, out double best);
                    edges.Add((prev, op, best));
                }

                foreach (var edge in edges.OrderByDescending(e => e.Score).Take(2))
                {
                    expected.Add((node + 1, edge.From, operations[edge.Op]));
                }
            }

            return expected;
        }

        private static Matrix<double> SoftmaxRows(Matrix<double> m)
        {
            var result = new Matrix<double>(m.Rows, m.Columns);
            for (int r = 0; r < m.Rows; r++)
            {
                double sum = 0;
                for (int c = 0; c < m.Columns; c++)
                {
                    sum += Math.Exp(m[r, c]);
                }

                for (int c = 0; c < m.Columns; c++)
                {
                    result[r, c] = Math.Exp(m[r, c]) / sum;
                }
            }

            return result;
        }

        [Fact]
        public void GDAS_DeriveArchitecture_TakesArgMaxOfAlphaAndTopTwoEdges()
        {
            // GDAS (Dong and Yang 2019, Sec. 3.3) derives the final cell like DARTS: the most probable operation per
            // edge and the two strongest incoming edges per node. Master drew a fresh hard Gumbel sample per call
            // and kept every edge.
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4);
            var alphas = gdas.GetArchitectureParameters();

            var first = gdas.DeriveArchitecture();
            var second = gdas.DeriveArchitecture();

            var expected = ExpectedTopTwo(alphas.Select(SoftmaxRows).ToList(), searchSpace.Operations);
            Assert.Equal(expected.OrderBy(e => e.ToNode).ThenBy(e => e.FromNode),
                first.Operations.OrderBy(e => e.ToNode).ThenBy(e => e.FromNode));
            Assert.Equal(first.Operations, second.Operations);
        }

        [Fact]
        public void PCDARTS_EdgeNormalization_WeightsEdgesBySoftmaxOfBeta()
        {
            // PC-DARTS (Xu et al. 2020, Eq. 4): edge (i, j) is weighted by softmax_i(beta_ij), so every node's
            // weights sum to 1 and each edge row sums to its beta weight. beta starts at 1e-3 * N(0, 1), so the
            // edge weights are 1 / k up to ~1e-3. Master scaled by 1 / sqrt(k) instead.
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4, useEdgeNormalization: true);

            foreach (var alpha in pcdarts.GetArchitectureParameters())
            {
                var normalized = pcdarts.ApplyEdgeNormalization(alpha);
                double total = 0;
                for (int r = 0; r < normalized.Rows; r++)
                {
                    double rowSum = 0;
                    for (int c = 0; c < normalized.Columns; c++)
                    {
                        rowSum += normalized[r, c];
                    }

                    Assert.Equal(1.0 / alpha.Rows, rowSum, 5e-3);
                    total += rowSum;
                }

                Assert.Equal(1.0, total, 1e-12);
            }
        }

        [Fact]
        public void PCDARTS_EdgeNormalization_KeepsOperationMixturePerEdge()
        {
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 3, useEdgeNormalization: true);
            var alpha = pcdarts.GetArchitectureParameters()[2];

            var normalized = pcdarts.ApplyEdgeNormalization(alpha);
            var opSoftmax = SoftmaxRows(alpha);
            for (int r = 0; r < alpha.Rows; r++)
            {
                double rowSum = 0;
                for (int c = 0; c < alpha.Columns; c++)
                {
                    rowSum += normalized[r, c];
                }

                for (int c = 0; c < alpha.Columns; c++)
                {
                    Assert.Equal(opSoftmax[r, c], normalized[r, c] / rowSum, 1e-12);
                }
            }
        }

        [Fact]
        public void ProxylessNAS_ExpectedLatency_IsProbabilityWeightedOperationLatency()
        {
            // ProxylessNAS (Cai et al. 2019, Eq. 5): E[latency] = sum_edges sum_ops p(op) * F(op).
            var searchSpace = new SearchSpaceBase<double>();
            var nas = new ProxylessNAS<double>(searchSpace, numNodes: 3);
            var costModel = new HardwareCostModel<double>(HardwarePlatform.Mobile);

            double expected = 0;
            foreach (var alpha in nas.GetArchitectureParameters())
            {
                var probs = SoftmaxRows(alpha);
                for (int r = 0; r < probs.Rows; r++)
                {
                    for (int c = 0; c < probs.Columns; c++)
                    {
                        expected += probs[r, c] * costModel.EstimateOperationCost(searchSpace.Operations[c], 16, 16, 32).Latency;
                    }
                }
            }

            Assert.Equal(expected, nas.ComputeExpectedLatency(16, 32), 1e-9);
        }

        [Fact]
        public void FBNet_DeriveArchitecture_PicksFirstMaximumLogitPerLayer()
        {
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 6);
            var thetas = fbnet.GetArchitectureParameters();
            var architecture = fbnet.DeriveArchitecture();

            Assert.Equal(thetas.Count, architecture.Operations.Count);
            for (int layer = 0; layer < thetas.Count; layer++)
            {
                int best = 0;
                for (int i = 1; i < thetas[layer].Length; i++)
                {
                    if (thetas[layer][i] > thetas[layer][best])
                    {
                        best = i;
                    }
                }

                Assert.Equal((layer + 1, layer, searchSpace.Operations[best]), architecture.Operations[layer]);
            }
        }

        [Fact]
        public void HardwareCostModel_ArchitectureCost_SumsOperationCosts()
        {
            var model = new HardwareCostModel<double>(HardwarePlatform.Mobile);
            var architecture = new Architecture<double>();
            architecture.AddOperation(1, 0, "conv3x3");
            architecture.AddOperation(2, 1, "maxpool3x3");
            architecture.AddOperation(2, 0, "conv5x5");

            var total = model.EstimateArchitectureCost(architecture, 8, 16);
            double latency = 0, energy = 0, memory = 0;
            foreach (var (_, _, op) in architecture.Operations)
            {
                var cost = model.EstimateOperationCost(op, 8, 8, 16);
                latency += cost.Latency;
                energy += cost.Energy;
                memory += cost.Memory;
            }

            Assert.Equal(latency, total.Latency, 1e-9);
            Assert.Equal(energy, total.Energy, 1e-9);
            Assert.Equal(memory, total.Memory, 1e-9);
        }
    }
}
