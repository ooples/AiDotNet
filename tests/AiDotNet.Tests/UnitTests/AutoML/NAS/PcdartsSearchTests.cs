using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// The PC-DARTS search loop (Xu et al. 2020, Sec. 3 and 4.2): weight steps on the training split alternate with
    /// architecture steps (alpha and beta) on the validation split after a weights-only warm-up, and the derived
    /// architecture takes the strongest non-zero operation of each edge, with edges ranked by
    /// softmax(beta)_i * softmax(alpha_i)_o.
    /// </summary>
    public class PcdartsSearchTests
    {
        private const int Features = 8;

        /// <summary>A tiny regression task: the target is a smoothed copy of the input.</summary>
        private static (Tensor<double> Inputs, Tensor<double> Targets) SyntheticTask(int samples, int seed)
        {
            var random = RandomHelper.CreateSeededRandom(seed);
            var inputs = new Tensor<double>(new[] { samples, Features });
            var targets = new Tensor<double>(new[] { samples, Features });
            for (int s = 0; s < samples; s++)
            {
                for (int f = 0; f < Features; f++)
                {
                    inputs[s, f] = random.NextDouble() * 2.0 - 1.0;
                }

                for (int f = 0; f < Features; f++)
                {
                    double left = f > 0 ? inputs[s, f - 1] : 0.0;
                    double right = f < Features - 1 ? inputs[s, f + 1] : 0.0;
                    targets[s, f] = 0.5 * inputs[s, f] + 0.25 * (left + right);
                }
            }

            return (inputs, targets);
        }

        private static List<double[]> Snapshot(PCDARTS<double> pcdarts)
            => pcdarts.GetArchitectureParameters().Select(alpha => alpha.AsSpan().ToArray()).ToList();

        private static List<double[]> SnapshotBeta(PCDARTS<double> pcdarts)
            => pcdarts.GetEdgeParameters().Select(beta => beta.ToArray()).ToList();

        private static double MaxChange(List<double[]> before, List<double[]> after)
            => before.Zip(after, (b, a) => b.Zip(a, (x, y) => Math.Abs(x - y)).DefaultIfEmpty(0.0).Max()).Max();

        private static PCDARTS<double> Create(int epochs, int warmupEpochs)
        {
            // No architecture weight decay: Adam would move alpha and beta on the decay term alone, so any change
            // seen here comes from the validation-loss gradient.
            var options = new PCDARTSOptions
            {
                Epochs = epochs,
                WarmupEpochs = warmupEpochs,
                BatchSize = 4,
                PartialChannelK = 4,
                ArchitectureWeightDecay = 0.0,
            };
            return new PCDARTS<double>(new SearchSpaceBase<double>(), options, numNodes: 2, useEdgeNormalization: true);
        }

        [Fact]
        public void Options_DefaultToThePaperSettings()
        {
            var options = new PCDARTSOptions();

            Assert.Equal(50, options.Epochs);
            Assert.Equal(15, options.WarmupEpochs);
            Assert.Equal(256, options.BatchSize);
            Assert.Equal(4, options.PartialChannelK);
            Assert.Equal(0.1, options.WeightMaxLearningRate);
            Assert.Equal(0.0, options.WeightMinLearningRate);
            Assert.Equal(0.9, options.WeightMomentum);
            Assert.Equal(3e-4, options.WeightDecay);
            Assert.Equal(6e-4, options.ArchitectureLearningRate);
            Assert.Equal(0.5, options.ArchitectureBeta1);
            Assert.Equal(0.999, options.ArchitectureBeta2);
            Assert.Equal(1e-3, options.ArchitectureWeightDecay);
        }

        [Fact(Timeout = 120000)]
        public async Task RunSearch_TrainsAlphaAndBeta()
        {
            var pcdarts = Create(epochs: 3, warmupEpochs: 1);
            var alphaBefore = Snapshot(pcdarts);
            var betaBefore = SnapshotBeta(pcdarts);
            var (inputs, targets) = SyntheticTask(8, seed: 11);
            var (validationInputs, validationTargets) = SyntheticTask(8, seed: 23);

            var architecture = pcdarts.RunSearch(inputs, targets, validationInputs, validationTargets,
                TimeSpan.Zero, CancellationToken.None);

            Assert.NotNull(architecture);
            Assert.True(MaxChange(alphaBefore, Snapshot(pcdarts)) > 1e-5, "alpha did not change");
            Assert.True(MaxChange(betaBefore, SnapshotBeta(pcdarts)) > 1e-5, "beta did not change");
        }

        [Fact(Timeout = 120000)]
        public async Task RunSearch_WarmupEpochs_LeaveAlphaAndBetaUnchanged()
        {
            var pcdarts = Create(epochs: 2, warmupEpochs: 2);
            var alphaBefore = Snapshot(pcdarts);
            var betaBefore = SnapshotBeta(pcdarts);
            var (inputs, targets) = SyntheticTask(8, seed: 11);
            var (validationInputs, validationTargets) = SyntheticTask(8, seed: 23);

            pcdarts.RunSearch(inputs, targets, validationInputs, validationTargets, TimeSpan.Zero, CancellationToken.None);

            Assert.Equal(0.0, MaxChange(alphaBefore, Snapshot(pcdarts)));
            Assert.Equal(0.0, MaxChange(betaBefore, SnapshotBeta(pcdarts)));
        }

        private static double[] Softmax(IReadOnlyList<double> logits)
        {
            double max = logits.Max();
            var exp = logits.Select(l => Math.Exp(l - max)).ToArray();
            double sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }

        [Fact]
        public void DeriveArchitecture_PicksTheArgmaxOperationPerEdge_WithEdgesRankedByBeta()
        {
            // Node index 2 has three incoming edges and keeps two. By softmax(alpha) alone, edges 0 and 1 are the
            // strongest; beta = (-3, 0, 3) moves the weight to edges 2 and 1.
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 3, useEdgeNormalization: true);
            var alpha = pcdarts.GetArchitectureParameters()[2];
            var beta = pcdarts.GetEdgeParameters()[2];
            for (int r = 0; r < alpha.Rows; r++)
            {
                for (int c = 0; c < alpha.Columns; c++)
                {
                    alpha[r, c] = 0.0;
                }
            }

            alpha[0, 1] = 3.0;
            alpha[1, 2] = 2.0;
            alpha[2, 3] = 1.0;
            beta[0] = -3.0;
            beta[1] = 0.0;
            beta[2] = 3.0;

            // The expected choice, computed from W = softmax(beta)_i * softmax(alpha_i)_o.
            var edgeWeights = Softmax(new[] { beta[0], beta[1], beta[2] });
            var expected = Enumerable.Range(0, 3)
                .Select(edge =>
                {
                    var row = Softmax(Enumerable.Range(0, alpha.Columns).Select(c => alpha[edge, c]).ToArray());
                    int op = Array.IndexOf(row, row.Max());
                    return (Edge: edge, Op: op, Score: edgeWeights[edge] * row[op]);
                })
                .OrderByDescending(e => e.Score)
                .Take(2)
                .Select(e => (3, e.Edge, searchSpace.Operations[e.Op]))
                .ToList();

            Assert.Equal(new[] { 2, 1 }, expected.Select(e => e.Item2).ToArray());

            var architecture = pcdarts.DeriveArchitecture();
            var node = architecture.Operations.Where(o => o.ToNode == 3).ToList();

            Assert.Equal(expected, node);
            Assert.Equal(searchSpace.Operations[3], node[0].Operation);
            Assert.Equal(searchSpace.Operations[2], node[1].Operation);
        }

        [Fact]
        public void DeriveArchitecture_NeverChoosesTheZeroOperation()
        {
            // The zero operation marks an absent edge (DARTS Sec. 2.4; PC-DARTS' genotype excludes 'none'), so even
            // the largest weight on it does not select it: the strongest other operation is chosen.
            var searchSpace = new SearchSpaceBase<double>
            {
                Operations = new List<string> { "none", "identity", "conv3x3" },
            };
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 1, useEdgeNormalization: true);
            var alpha = pcdarts.GetArchitectureParameters()[0];
            alpha[0, 0] = 5.0;
            alpha[0, 1] = 0.0;
            alpha[0, 2] = 1.0;

            var architecture = pcdarts.DeriveArchitecture();

            var operation = Assert.Single(architecture.Operations);
            Assert.Equal((1, 0, "conv3x3"), operation);
        }
    }
}
