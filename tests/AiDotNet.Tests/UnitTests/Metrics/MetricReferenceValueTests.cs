using AiDotNet.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.Metrics;

/// <summary>
/// Reference values (scikit-learn / SciPy) for metric formulas corrected in the element-wise sweep.
/// </summary>
public class MetricReferenceValueTests
{
    [Fact]
    public void NdcgAtK_TiedScores_MatchesScikitLearnTieAveraging()
    {
        // sklearn.metrics.ndcg_score([[1, 0, 2]], [[1, 1, 0]]) = 0.6900468833579672
        var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0 });
        var relevance = new Vector<double>(new[] { 1.0, 0.0, 2.0 });

        double ndcg = RankingMetrics<double>.NdcgAtK(predicted, relevance, 0, useExponentialGain: false);

        Assert.Equal(0.6900468833579672, ndcg, 12);
    }

    [Fact]
    public void NdcgAtK_TiedScoresWithCutoff_MatchesScikitLearn()
    {
        // sklearn.metrics.ndcg_score([[3, 2, 3, 0, 1, 2]], [[0.9, 0.9, 0.5, 0.5, 0.1, 0.3]], k=4) = 0.8103677588719043
        var predicted = new Vector<double>(new[] { 0.9, 0.9, 0.5, 0.5, 0.1, 0.3 });
        var relevance = new Vector<double>(new[] { 3.0, 2.0, 3.0, 0.0, 1.0, 2.0 });

        double ndcg = RankingMetrics<double>.NdcgAtK(predicted, relevance, 4, useExponentialGain: false);

        Assert.Equal(0.8103677588719043, ndcg, 12);
    }

    [Fact]
    public void FrechetDistance_NonCommutingCovariances_MatchesScipySqrtm()
    {
        // ||m1 - m2||^2 + Tr(S1) + Tr(S2) - 2 Tr(scipy.linalg.sqrtm(S1 @ S2)) = 5.77122044765434
        var mean1 = new Vector<double>(new[] { 0.0, 1.0 });
        var mean2 = new Vector<double>(new[] { 1.0, 3.0 });
        var cov1 = new Matrix<double>(new double[,] { { 1.0, 0.0 }, { 0.0, 4.0 } });
        var cov2 = new Matrix<double>(new double[,] { { 2.0, 1.0 }, { 1.0, 2.0 } });

        double distance = FrechetDistanceHelper<double>.FrechetDistance(mean1, cov1, mean2, cov2);

        Assert.Equal(5.77122044765434, distance, 9);
    }
}
