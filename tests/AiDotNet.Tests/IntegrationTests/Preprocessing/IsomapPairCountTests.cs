using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Manifold;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the Isomap selectors' connected-pair count, which is kept in int.
/// </summary>
public class IsomapPairCountTests
{
    // 260 points on a line (feature 1 = 0.001 i) with a constant feature 0: 260 * 259 = 67340 connected pairs. The
    // former count was a TensorSum of the pair mask in Half, which is +infinity past 65504, so every mean score was
    // finite / infinity = 0 and the tie went to feature 0. Feature 1 carries all the geodesic structure.
    [Fact(Timeout = 120000)]
    public async Task IsomapBasedSelector_Half_ScoresPastHalfsLargestPairCount()
    {
        await Task.CompletedTask;
        const int n = 260;
        var data = new Half[n, 2];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = (Half)0;
            data[i, 1] = (Half)(0.001 * i);
        }

        var selector = new IsomapBasedSelector<Half>(nFeaturesToSelect: 1, nNeighbors: 5);
        selector.Fit(new Matrix<Half>(data));

        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
        var scores = selector.FeatureScores;
        Assert.NotNull(scores);
        if (scores is not null)
        {
            Assert.Equal(0.0, (double)scores[0]);
            Assert.Equal(1.0, (double)scores[1]);
        }
    }

    // 300 points (89700 ordered pairs): feature 1 a line of step 2^-8 that sets the geodesics, feature 0 a 0/1
    // pattern times 2^-9 unrelated to the position. Feature 0's score is the Pearson correlation, over the
    // connected pairs, of |x_i - x_j| with the geodesic; the expectation recomputes it in double from the same
    // Half geodesics. The former count summed in Half was +infinity, so the mean geodesic was 0 and the variations
    // uncentred: the score became the cosine sum(d g) / sqrt(sum d^2 sum g^2), about 0.58 instead of about 0.
    [Fact(Timeout = 120000)]
    public async Task IsomapSelector_Half_CentresTheCorrelationPastHalfsLargestPairCount()
    {
        await Task.CompletedTask;
        const int n = 300;
        const int neighbours = 20;
        var data = new Half[n, 2];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = (Half)(((i * 37) % 11) % 2 / 512.0);
            data[i, 1] = (Half)((i - n / 2) / 256.0);
        }

        var selector = new IsomapSelector<Half>(nFeaturesToSelect: 1, nNeighbors: neighbours);
        selector.Fit(new Matrix<Half>(data));

        var engine = AiDotNetEngine.Current;
        var x = Tensor<Half>.FromMatrix(new Matrix<Half>(data));
        var euclidean = FeatureSelectionTensorOps.PairwiseDistances(x);
        var nearest = FeatureSelectionTensorOps.KNearestMask(euclidean, neighbours);
        var geodesic = FeatureSelectionTensorOps.ShortestPaths(
            euclidean, engine.TensorMax(nearest, engine.TensorTranspose(nearest)), out var reachable);

        int count = 0;
        double sumD = 0, sumG = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j || (double)reachable[i, j] == 0)
                    continue;
                count++;
                sumD += Math.Abs((double)data[i, 0] - (double)data[j, 0]);
                sumG += (double)geodesic[i, j];
            }
        }

        Assert.True(count > 65504, $"only {count} connected pairs");
        double meanD = sumD / count, meanG = sumG / count;
        double sdg = 0, sdd = 0, sgg = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j || (double)reachable[i, j] == 0)
                    continue;
                double d = Math.Abs((double)data[i, 0] - (double)data[j, 0]) - meanD;
                double g = (double)geodesic[i, j] - meanG;
                sdg += d * g;
                sdd += d * d;
                sgg += g * g;
            }
        }

        double expected = sdg / Math.Sqrt(sdd * sgg);
        var scores = selector.ManifoldScores;
        Assert.NotNull(scores);
        if (scores is not null)
            Assert.InRange((double)scores[0], expected - 0.05, expected + 0.05);
    }
}
