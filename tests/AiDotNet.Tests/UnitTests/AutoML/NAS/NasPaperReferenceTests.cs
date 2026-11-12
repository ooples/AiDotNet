using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.NeuralNetworks;
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
        public void ENAS_ControllerLogProbabilityGradient_MatchesFiniteDifferences()
        {
            // ENAS (Pham et al. 2018, Sec. 2.1) samples each decision from an LSTM controller; REINFORCE needs
            // d log pi(a) / d theta through the whole LSTM, attention and embeddings. Check it numerically.
            var space = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(space, numNodes: 3, controllerHiddenSize: 4);
            enas.SampleArchitecture();
            var decisions = enas.LastDecisions.ToList();
            Assert.Equal(2 * 3, decisions.Count);

            var analytic = enas.LogProbabilityGradient(decisions);
            var parameters = enas.GetControllerParameters();
            Assert.Equal(parameters.Count, analytic.Count);

            const double h = 1e-6;
            bool anyNonZero = false;
            for (int i = 0; i < parameters.Count; i++)
            {
                Assert.Equal(parameters[i].Length, analytic[i].Length);
                for (int k = 0; k < parameters[i].Length; k += Math.Max(1, parameters[i].Length / 7))
                {
                    double original = parameters[i][k];
                    parameters[i][k] = original + h;
                    double up = enas.ReplayLogProbability(decisions);
                    parameters[i][k] = original - h;
                    double down = enas.ReplayLogProbability(decisions);
                    parameters[i][k] = original;

                    double numeric = (up - down) / (2 * h);
                    Assert.Equal(numeric, analytic[i][k], 1e-6);
                    anyNonZero |= Math.Abs(analytic[i][k]) > 1e-9;
                }
            }

            Assert.True(anyNonZero);
        }

        [Fact]
        public void ENAS_UpdateController_TakesAdamStepOnReinforceGradientWithBaseline()
        {
            // ENAS: R' = R + lambda * H, b <- decay * b + (1 - decay) * R', gradient -(R' - b) d log pi(a) / d theta,
            // and the controller is trained with Adam at 0.00035. The first Adam step (bias-corrected) moves every
            // parameter by -lr * g / (|g| + eps).
            const double decay = 0.9, lambda = 0.1, lr = 3.5e-4, eps = 1e-8;
            var space = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(space, numNodes: 3, controllerHiddenSize: 4, baselineDecay: decay, entropyWeight: lambda);
            var (_, logProb, entropy) = enas.SampleArchitecture();
            var decisions = enas.LastDecisions.ToList();
            Assert.Equal(logProb, enas.ReplayLogProbability(decisions), 12);

            var scoreFunction = enas.LogProbabilityGradient(decisions);
            var before = enas.GetControllerParameters().Select(v => v.ToArray()).ToList();
            enas.UpdateController(2.0, logProb, entropy);

            double shaped = 2.0 + lambda * entropy;
            double baseline = (1 - decay) * shaped;
            Assert.Equal(baseline, enas.GetBaseline(), 12);

            var gradients = enas.GetControllerGradients();
            var after = enas.GetControllerParameters();
            for (int i = 0; i < gradients.Count; i++)
            {
                for (int k = 0; k < gradients[i].Length; k++)
                {
                    double g = -(shaped - baseline) * scoreFunction[i][k];
                    Assert.Equal(g, gradients[i][k], 12);
                    Assert.Equal(before[i][k] - lr * g / (Math.Abs(g) + eps), after[i][k], 12);
                }
            }
        }

        [Fact]
        public void DartsOptimizers_MatchPyTorchAdamAndSgdWithWeightDecay()
        {
            // DARTS trains alpha with Adam(lr 3e-4, betas (0.5, 0.999), wd 1e-3) and weights with SGD(momentum 0.9,
            // wd 3e-4) under a cosine schedule from 0.025 to 0.001; master used Adam(0.9, 0.999) without decay
            // for both. These pin the per-step formulas of torch.optim.Adam / torch.optim.SGD.
            var theta = new Vector<double>(new[] { 0.5, -1.0, 2.0 });
            var grad = new Vector<double>(new[] { 0.1, 0.3, -0.2 });
            var m = new Vector<double>(new[] { 0.01, -0.02, 0.03 });
            var v = new Vector<double>(new[] { 0.001, 0.002, 0.003 });
            double[] th0 = theta.ToArray(), m0 = m.ToArray(), v0 = v.ToArray();
            const double lr = 3e-4, b1 = 0.5, b2 = 0.999, eps = 1e-8, wd = 1e-3;
            const int t = 3;
            NeuralArchitectureSearch<double>.AdamStep(theta, grad, m, v, lr, b1, b2, eps, wd, t);
            for (int i = 0; i < 3; i++)
            {
                double g = grad[i] + wd * th0[i];
                double mi = b1 * m0[i] + (1 - b1) * g;
                double vi = b2 * v0[i] + (1 - b2) * g * g;
                double mHat = mi / (1 - Math.Pow(b1, t));
                double vHat = vi / (1 - Math.Pow(b2, t));
                Assert.Equal(mi, m[i], 1e-15);
                Assert.Equal(vi, v[i], 1e-15);
                Assert.Equal(th0[i] - lr * mHat / (Math.Sqrt(vHat) + eps), theta[i], 1e-15);
            }

            var w = new Vector<double>(new[] { 1.0, -2.0 });
            var wg = new Vector<double>(new[] { 3.0, 4.0 });
            var buf = new Vector<double>(new[] { 0.5, 0.25 });
            double scale = NeuralArchitectureSearch<double>.GradientClipScale(new[] { wg }, 2.5);
            Assert.Equal(2.5 / (5.0 + 1e-6), scale, 1e-15);
            Assert.Equal(1.0, NeuralArchitectureSearch<double>.GradientClipScale(new[] { wg }, 10.0));

            NeuralArchitectureSearch<double>.SgdMomentumStep(w, wg, buf, 0.025, 0.9, 3e-4, scale);
            double[] w0 = { 1.0, -2.0 }, g0 = { 3.0, 4.0 }, b0 = { 0.5, 0.25 };
            for (int i = 0; i < 2; i++)
            {
                double b = 0.9 * b0[i] + (scale * g0[i] + 3e-4 * w0[i]);
                Assert.Equal(b, buf[i], 1e-15);
                Assert.Equal(w0[i] - 0.025 * b, w[i], 1e-15);
            }

            var schedule = NeuralArchitectureSearch<double>.CosineAnnealedLearningRates(50, 0.025, 0.001);
            Assert.Equal(51, schedule.Length);
            Assert.Equal(0.025, schedule[0], 1e-15);
            Assert.Equal(0.013, schedule[25], 1e-15);
            Assert.Equal(0.001, schedule[50], 1e-15);
            Assert.Equal(0.001 + 0.024 * 0.5 * (1 + Math.Cos(Math.PI * 10 / 50)), schedule[10], 1e-15);
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

        /// <summary>
        /// Random search with weight sharing (Li &amp; Talwalkar 2019, Sec. 3) scores each sampled architecture by
        /// applying it to the shared-weight supernet. Before the fix every trial scored a fresh, untouched supernet,
        /// so the loss was independent of the architecture it was attributed to.
        /// </summary>
        [Fact]
        public void RandomSearch_ScoresEachCandidateWithItsOwnArchitectureOnTheSharedSupernet()
        {
            var searchSpace = new SearchSpaceBase<double> { Operations = new List<string> { "identity", "conv3x3", "avgpool3x3" } };
            var ops = searchSpace.Operations;
            var supernet = new SuperNet<double>(searchSpace, numNodes: 3);
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
            var x = new Tensor<double>(new[] { 8, 6 });
            var y = new Tensor<double>(new[] { 8, 6 });
            for (int i = 0; i < x.Length; i++)
            {
                x.SetFlat(i, random.NextDouble() * 2.0 - 1.0);
                y.SetFlat(i, random.NextDouble());
            }

            supernet.Predict(x);

            var candidates = new List<Architecture<double>>
            {
                AllEdges(3, "identity"),
                AllEdges(3, "conv3x3"),
            };
            candidates.AddRange(Enumerable.Range(0, 4).Select(_ => NeuralArchitectureSearch<double>.SampleEdgeArchitecture(random, 3, ops)));

            // Each sampled architecture covers every edge with one search-space operation.
            Assert.All(candidates, c => Assert.Equal(1 + 2 + 3, c.Operations.Count));

            // Reference: apply each candidate by hand and take its validation loss.
            var alphas = supernet.GetArchitectureParameters();
            var expected = candidates.Select(c =>
            {
                NeuralArchitectureSearch<double>.ApplyEdgeArchitecture(alphas, ops, c);

                // The applied alphas encode exactly this architecture.
                var derived = supernet.DeriveArchitecture().Operations.OrderBy(o => o.ToNode).ThenBy(o => o.FromNode);
                Assert.Equal(c.Operations.OrderBy(o => o.ToNode).ThenBy(o => o.FromNode), derived);
                return supernet.ComputeValidationLoss(x, y);
            }).ToList();
            Assert.NotEqual(expected[0], expected[1], 6);

            var best = NeuralArchitectureSearch<double>.SelectByValidationLoss(supernet, ops, candidates, x, y, out double bestLoss);

            int bestIndex = expected.IndexOf(expected.Min());
            Assert.Same(candidates[bestIndex], best);
            Assert.Equal(expected[bestIndex], bestLoss, 12);
        }

        private static Architecture<double> AllEdges(int nodes, string op)
        {
            var arch = new Architecture<double>();
            for (int node = 0; node < nodes; node++)
            {
                for (int from = 0; from <= node; from++)
                {
                    arch.AddOperation(node, from, op);
                }
            }

            return arch;
        }
    }
}
