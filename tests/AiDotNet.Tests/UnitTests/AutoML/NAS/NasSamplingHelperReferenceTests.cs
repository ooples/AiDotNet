using System;
using AiDotNet.AutoML.NAS;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Reference tests for the NAS sampling helpers against closed-form softmax and the Gumbel-max theorem.
    /// </summary>
    public class NasSamplingHelperReferenceTests
    {
        private const double Tolerance = 1e-12;
        private static readonly DoubleOperations Ops = new DoubleOperations();

        [Fact]
        public void Softmax_MatchesClosedForm()
        {
            var logits = new Vector<double>(new[] { 1.0, -2.0, 0.5, 3.0 });
            var result = NasSamplingHelper.Softmax(logits, Ops);

            double max = 3.0;
            double sum = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                sum += Math.Exp(logits[i] - max);
            }

            for (int i = 0; i < logits.Length; i++)
            {
                Assert.Equal(Math.Exp(logits[i] - max) / sum, result[i], Tolerance);
            }
        }

        [Fact]
        public void SoftmaxRowsWithTemperature_MatchesClosedFormPerRow()
        {
            var logits = new Matrix<double>(new double[,] { { 1.0, 2.0, 3.0 }, { -1.0, 0.0, 4.0 } });
            const double temperature = 0.5;
            var result = NasSamplingHelper.SoftmaxRowsWithTemperature(logits, temperature, Ops);
            var plain = NasSamplingHelper.SoftmaxRows(logits, Ops);

            for (int r = 0; r < 2; r++)
            {
                double sumT = 0, sum = 0;
                for (int c = 0; c < 3; c++)
                {
                    sumT += Math.Exp(logits[r, c] / temperature);
                    sum += Math.Exp(logits[r, c]);
                }

                for (int c = 0; c < 3; c++)
                {
                    Assert.Equal(Math.Exp(logits[r, c] / temperature) / sumT, result[r, c], 1e-10);
                    Assert.Equal(Math.Exp(logits[r, c]) / sum, plain[r, c], 1e-10);
                }
            }
        }

        [Fact]
        public void GumbelSoftmax_SoftSample_UsesUnclippedGumbelNoise()
        {
            // Jang et al. 2017, Eq. 1: y = softmax((logits + g) / tau), g = -log(-log(u)), u ~ U(0, 1).
            var logits = new Vector<double>(new[] { 0.3, -1.2, 2.0 });
            const double temperature = 0.7;
            var result = NasSamplingHelper.GumbelSoftmax(logits, temperature, Ops, RandomHelper.CreateSeededRandom(17));

            var reference = RandomHelper.CreateSeededRandom(17);
            var z = new double[3];
            for (int i = 0; i < 3; i++)
            {
                double u = reference.NextDouble() * (1.0 - 2e-10) + 1e-10;
                z[i] = (logits[i] - Math.Log(-Math.Log(u))) / temperature;
            }

            double max = Math.Max(z[0], Math.Max(z[1], z[2]));
            double sum = Math.Exp(z[0] - max) + Math.Exp(z[1] - max) + Math.Exp(z[2] - max);
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(Math.Exp(z[i] - max) / sum, result[i], 1e-10);
            }
        }

        [Fact]
        public void GumbelSoftmax_HardSamples_FollowCategoricalDistribution()
        {
            // Gumbel-max theorem: argmax(logits + g) ~ Categorical(softmax(logits)). Clipping u to
            // [0.005, 0.995] (master) truncated the noise and under-sampled the low-probability class.
            var logits = new Vector<double>(new[] { 0.0, Math.Log(4.0), Math.Log(0.25) });
            var random = RandomHelper.CreateSeededRandom(1234);
            const int draws = 200000;
            var counts = new int[3];
            for (int n = 0; n < draws; n++)
            {
                var oneHot = NasSamplingHelper.GumbelSoftmax(logits, 1.0, Ops, random, hard: true);
                for (int i = 0; i < 3; i++)
                {
                    if (oneHot[i] == 1.0)
                    {
                        counts[i]++;
                    }
                }
            }

            double z = 1.0 + 4.0 + 0.25;
            double[] expected = { 1.0 / z, 4.0 / z, 0.25 / z };
            for (int i = 0; i < 3; i++)
            {
                // 4 standard errors of a binomial proportion with 200k draws is below 0.0019; the clipped
                // noise gave 0.767 and 0.044 for the last two classes (off by 0.005 and 0.003).
                Assert.Equal(expected[i], counts[i] / (double)draws, 0.002);
            }
        }

        [Fact]
        public void GumbelSoftmaxRows_DrawsRowsInOrderFromTheSameStream()
        {
            var logits = new Matrix<double>(new double[,] { { 0.1, 0.2 }, { 1.0, -1.0 }, { 0.0, 0.0 } });
            var rows = NasSamplingHelper.GumbelSoftmaxRows(logits, 0.9, Ops, RandomHelper.CreateSeededRandom(5));

            var random = RandomHelper.CreateSeededRandom(5);
            for (int r = 0; r < 3; r++)
            {
                var expected = NasSamplingHelper.GumbelSoftmax(logits.GetRow(r), 0.9, Ops, random);
                for (int c = 0; c < 2; c++)
                {
                    Assert.Equal(expected[c], rows[r, c], Tolerance);
                }
            }
        }
    }
}
