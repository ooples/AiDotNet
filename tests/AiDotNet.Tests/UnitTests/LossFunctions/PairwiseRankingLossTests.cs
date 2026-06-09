using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Metrics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.LossFunctions
{
    public class PairwiseRankingLossTests
    {
        private const double Tol = 1e-9;

        [Fact(Timeout = 60000)]
        public async Task PerfectlyOrderedPredictions_GiveZeroLossInTheLimit()
        {
            await Task.CompletedTask;
            // Predictions perfectly ordered with a huge margin => log(1+exp(-large)) -> 0.
            var loss = new PairwiseRankingLoss<double>();
            var predicted = new Vector<double>(new double[] { 100.0, 50.0, 0.0, -50.0 });
            var actual = new Vector<double>(new double[] { 4.0, 3.0, 2.0, 1.0 });

            var result = loss.CalculateLoss(predicted, actual);

            Assert.True(result >= 0.0);
            Assert.True(result < 1e-12, $"Expected near-zero loss for perfectly ordered preds, got {result}");
        }

        [Fact(Timeout = 60000)]
        public async Task EqualScores_GiveLog2_StandardRankNetReference()
        {
            await Task.CompletedTask;
            // All predicted scores equal => every pair loss = log(1+exp(0)) = log(2).
            var loss = new PairwiseRankingLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });
            var actual = new Vector<double>(new double[] { 3.0, 2.0, 1.0 });

            var result = loss.CalculateLoss(predicted, actual);

            Assert.Equal(Math.Log(2.0), result, 9);
        }

        [Fact(Timeout = 60000)]
        public async Task MisorderedPair_GivesPositiveLoss()
        {
            await Task.CompletedTask;
            var loss = new PairwiseRankingLoss<double>();
            // True order says item0 > item1, but predictions say item0 < item1 => positive loss.
            var predicted = new Vector<double>(new double[] { 0.0, 2.0 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0 });

            var result = loss.CalculateLoss(predicted, actual);

            // Single pair (0,1): log(1+exp(-(0-2))) = log(1+exp(2)).
            Assert.Equal(Math.Log(1.0 + Math.Exp(2.0)), result, 9);
            Assert.True(result > 0.0);
        }

        [Fact(Timeout = 60000)]
        public async Task Gradient_OnMisorderedPair_HasCorrectSign()
        {
            await Task.CompletedTask;
            var loss = new PairwiseRankingLoss<double>();
            // item0 should rank above item1 (actual 1 > 0) but predicted score is lower.
            var predicted = new Vector<double>(new double[] { 0.0, 2.0 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0 });

            var grad = loss.CalculateDerivative(predicted, actual);

            // To DECREASE loss we must raise s0 and lower s1. Gradient descent steps along
            // -grad, so grad[0] must be negative (push s0 up) and grad[1] positive (push s1 down).
            Assert.True(grad[0] < 0.0, $"grad[0] should be negative, got {grad[0]}");
            Assert.True(grad[1] > 0.0, $"grad[1] should be positive, got {grad[1]}");
            // Gradient is anti-symmetric for a single pair.
            Assert.Equal(-grad[0], grad[1], 9);
        }

        [Fact(Timeout = 60000)]
        public async Task Gradient_MatchesFiniteDifference()
        {
            await Task.CompletedTask;
            var loss = new PairwiseRankingLoss<double>(tailWeightPower: 1.0);
            var predicted = new Vector<double>(new double[] { 0.3, -0.1, 1.2, 0.7 });
            var actual = new Vector<double>(new double[] { 0.05, -0.02, 0.10, -0.08 });

            var analytic = loss.CalculateDerivative(predicted, actual);

            double eps = 1e-6;
            for (int k = 0; k < predicted.Length; k++)
            {
                var plus = new double[predicted.Length];
                var minus = new double[predicted.Length];
                for (int i = 0; i < predicted.Length; i++) { plus[i] = predicted[i]; minus[i] = predicted[i]; }
                plus[k] += eps;
                minus[k] -= eps;

                double lp = loss.CalculateLoss(new Vector<double>(plus), actual);
                double lm = loss.CalculateLoss(new Vector<double>(minus), actual);
                double fd = (lp - lm) / (2.0 * eps);

                Assert.Equal(fd, analytic[k], 5);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task TailWeighting_IncreasesContributionOfExtremePairs()
        {
            await Task.CompletedTask;
            // Two mis-ordered pairs of equal predicted-margin error, but one pair sits at the
            // extreme tails of the actual distribution and the other near the median.
            // Construct so that the extreme misorder dominates under tail weighting.
            // actuals: index0 = top, index4 = bottom (extremes); index2 = median.
            var actual = new Vector<double>(new double[] { 10.0, 1.0, 0.0, -1.0, -10.0 });

            // Mis-order ONLY the extreme pair (0 vs 4): predict bottom above top.
            var predExtremeBad = new Vector<double>(new double[] { 0.0, 5.0, 4.0, 3.0, 8.0 });
            // Mis-order ONLY a near-median pair (1 vs 3) by the same predicted margin.
            var predMedianBad = new Vector<double>(new double[] { 8.0, 0.0, 4.0, 3.0, -8.0 });

            var plain = new PairwiseRankingLoss<double>(tailWeightPower: 0.0);
            var tailed = new PairwiseRankingLoss<double>(tailWeightPower: 3.0);

            // Under PLAIN RankNet, a single mis-ordered pair of equal margin costs the same
            // regardless of which pair it is (weights all 1, and the averaging normalizer is
            // by total pairs which is identical between the two prediction vectors).
            double plainExtreme = plain.CalculateLoss(predExtremeBad, actual);
            double plainMedian = plain.CalculateLoss(predMedianBad, actual);

            // Under TAIL weighting, mis-ordering the extreme pair must cost strictly MORE than
            // mis-ordering the median pair.
            double tailedExtreme = tailed.CalculateLoss(predExtremeBad, actual);
            double tailedMedian = tailed.CalculateLoss(predMedianBad, actual);

            Assert.True(tailedExtreme > tailedMedian,
                $"Tail weighting should penalize the extreme misorder more: extreme={tailedExtreme}, median={tailedMedian}");

            // Sanity: tail weighting changed the relative ranking of the two errors compared to plain.
            // (Plain treats them comparably; tailed strongly prefers fixing the extreme.)
            Assert.True(tailedExtreme / Math.Max(tailedMedian, 1e-12) > plainExtreme / Math.Max(plainMedian, 1e-12));
        }

        [Fact(Timeout = 60000)]
        public async Task TailWeightPower_Zero_EqualsStandardRankNet()
        {
            await Task.CompletedTask;
            var plain = new PairwiseRankingLoss<double>(tailWeightPower: 0.0);
            var explicitDefault = new PairwiseRankingLoss<double>();

            var predicted = new Vector<double>(new double[] { 0.5, -0.3, 1.1, 0.2 });
            var actual = new Vector<double>(new double[] { 0.02, -0.05, 0.09, 0.01 });

            Assert.Equal(plain.CalculateLoss(predicted, actual), explicitDefault.CalculateLoss(predicted, actual), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_RejectsNegativePower()
        {
            await Task.CompletedTask;
            Assert.Throws<ArgumentOutOfRangeException>(() => new PairwiseRankingLoss<double>(-0.5));
        }

        [Fact(Timeout = 60000)]
        public async Task AllTiedActuals_GiveZeroLossAndZeroGradient()
        {
            await Task.CompletedTask;
            var loss = new PairwiseRankingLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 5.0, 5.0, 5.0 });

            Assert.Equal(0.0, loss.CalculateLoss(predicted, actual), 12);
            var grad = loss.CalculateDerivative(predicted, actual);
            for (int i = 0; i < grad.Length; i++) Assert.Equal(0.0, grad[i], 12);
        }

        // ---------- NDCG metric ----------

        [Fact(Timeout = 60000)]
        public async Task Ndcg_PerfectRanking_IsOne()
        {
            await Task.CompletedTask;
            var predicted = new Vector<double>(new double[] { 3.0, 2.0, 1.0, 0.0 });
            var relevance = new Vector<double>(new double[] { 3.0, 2.0, 1.0, 0.0 });

            var ndcg = RankingMetrics<double>.NdcgAtK(predicted, relevance, 4);

            Assert.Equal(1.0, ndcg, 9);
        }

        [Fact(Timeout = 60000)]
        public async Task Ndcg_SwapDegradesScoreBelowOne()
        {
            await Task.CompletedTask;
            var relevance = new Vector<double>(new double[] { 3.0, 2.0, 1.0, 0.0 });
            // Swap top two predicted scores so the #2 item is ranked first.
            var predicted = new Vector<double>(new double[] { 2.0, 3.0, 1.0, 0.0 });

            var ndcg = RankingMetrics<double>.NdcgAtK(predicted, relevance, 4);

            Assert.True(ndcg < 1.0, $"A swap should give NDCG < 1, got {ndcg}");
            Assert.True(ndcg > 0.0);
        }

        [Fact(Timeout = 60000)]
        public async Task Ndcg_AtK_OnlyConsidersTopK()
        {
            await Task.CompletedTask;
            // Mistakes beyond the cutoff should not affect NDCG@1.
            var relevance = new Vector<double>(new double[] { 5.0, 0.0, 4.0, 3.0 });
            var predicted = new Vector<double>(new double[] { 9.0, 8.0, 1.0, 0.0 }); // top item correct, rest wrong

            var ndcgAt1 = RankingMetrics<double>.NdcgAtK(predicted, relevance, 1);

            // Top predicted item is the most relevant => NDCG@1 = 1 regardless of lower ranks.
            Assert.Equal(1.0, ndcgAt1, 9);
        }

        [Fact(Timeout = 60000)]
        public async Task Ndcg_LinearGain_HandlesSignedReturns()
        {
            await Task.CompletedTask;
            // Signed forward returns: linear gain ranks the perfect order at its own ideal => 1.0.
            var relevance = new Vector<double>(new double[] { 0.08, 0.01, -0.02, -0.07 });
            var predicted = new Vector<double>(new double[] { 4.0, 3.0, 2.0, 1.0 });

            var ndcg = RankingMetrics<double>.NdcgAtK(predicted, relevance, 4, useExponentialGain: false);

            Assert.Equal(1.0, ndcg, 9);
        }
    }
}
