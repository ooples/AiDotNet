using System;
using System.Linq;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// PC-DARTS (Xu et al. 2020, Sec. 3.3, Eq. 4) learns the edge-normalization parameters beta jointly with alpha:
    /// W = softmax(beta)_i * softmax(alpha_i)_o, and the architecture optimizer updates both.
    /// </summary>
    public class PcdartsEdgeTrainingTests
    {
        private static double Surrogate(PCDARTS<double> pcdarts, int node, Matrix<double> upstream)
        {
            var weights = pcdarts.ApplyEdgeNormalization(pcdarts.GetArchitectureParameters()[node]);
            double sum = 0;
            for (int r = 0; r < weights.Rows; r++)
            {
                for (int c = 0; c < weights.Columns; c++)
                {
                    sum += weights[r, c] * upstream[r, c];
                }
            }

            return sum;
        }

        [Fact]
        public void AccumulateArchitectureGradients_MatchesFiniteDifferencesForAlphaAndBeta()
        {
            const int node = 2;
            var pcdarts = new PCDARTS<double>(new SearchSpaceBase<double>(), numNodes: 3, useEdgeNormalization: true);
            var alpha = pcdarts.GetArchitectureParameters()[node];
            var beta = pcdarts.GetEdgeParameters()[node];

            // Spread beta out so the edge weights are far from uniform and its gradient is not negligible.
            for (int i = 0; i < beta.Length; i++)
            {
                beta[i] = 0.4 * i - 0.3;
            }

            var upstream = new Matrix<double>(alpha.Rows, alpha.Columns);
            for (int r = 0; r < alpha.Rows; r++)
            {
                for (int c = 0; c < alpha.Columns; c++)
                {
                    upstream[r, c] = Math.Sin(1.3 * r + 0.7 * c + 0.2);
                }
            }

            pcdarts.AccumulateArchitectureGradients(node, upstream);
            var alphaGradient = pcdarts.GetArchitectureGradients()[node];
            var betaGradient = pcdarts.GetEdgeGradients()[node];

            const double h = 1e-6;
            bool betaMoves = false;
            for (int i = 0; i < beta.Length; i++)
            {
                double original = beta[i];
                beta[i] = original + h;
                double up = Surrogate(pcdarts, node, upstream);
                beta[i] = original - h;
                double down = Surrogate(pcdarts, node, upstream);
                beta[i] = original;
                Assert.Equal((up - down) / (2 * h), betaGradient[i], 1e-6);
                betaMoves |= Math.Abs(betaGradient[i]) > 1e-6;
            }

            Assert.True(betaMoves);

            for (int r = 0; r < alpha.Rows; r++)
            {
                for (int c = 0; c < alpha.Columns; c++)
                {
                    double original = alpha[r, c];
                    alpha[r, c] = original + h;
                    double up = Surrogate(pcdarts, node, upstream);
                    alpha[r, c] = original - h;
                    double down = Surrogate(pcdarts, node, upstream);
                    alpha[r, c] = original;
                    Assert.Equal((up - down) / (2 * h), alphaGradient[r, c], 1e-6);
                }
            }
        }

        [Fact]
        public void UpdateArchitectureParameters_TakesAnAdamStepOnBeta_AndClearsTheGradients()
        {
            // First bias-corrected Adam step with L2 decay: theta -= lr * g' / (|g'| + eps), g' = g + wd * theta.
            const int node = 1;
            const double lr = 6e-4, wd = 1e-3, eps = 1e-8;
            var pcdarts = new PCDARTS<double>(new SearchSpaceBase<double>(), numNodes: 3, useEdgeNormalization: true);
            var alpha = pcdarts.GetArchitectureParameters()[node];
            var upstream = new Matrix<double>(alpha.Rows, alpha.Columns);
            for (int c = 0; c < alpha.Columns; c++)
            {
                upstream[0, c] = 1.0;
                upstream[1, c] = -1.0;
            }

            pcdarts.AccumulateArchitectureGradients(node, upstream);
            var beta = pcdarts.GetEdgeParameters()[node];
            var before = beta.ToArray();
            var gradient = pcdarts.GetEdgeGradients()[node].ToArray();
            Assert.True(gradient.Any(g => Math.Abs(g) > 1e-6));

            pcdarts.UpdateArchitectureParameters();

            for (int i = 0; i < beta.Length; i++)
            {
                double g = gradient[i] + wd * before[i];
                Assert.Equal(before[i] - lr * g / (Math.Abs(g) + eps), beta[i], 12);
                Assert.Equal(0.0, pcdarts.GetEdgeGradients()[node][i]);
            }
        }
    }
}
