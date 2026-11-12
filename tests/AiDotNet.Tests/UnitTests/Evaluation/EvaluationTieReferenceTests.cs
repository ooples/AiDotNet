using AiDotNet.Evaluation.Metrics.Classification;
using AiDotNet.Evaluation.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evaluation;

/// <summary>
/// Reference values for metrics whose result depends on how tied scores and pooled ranks are handled.
/// Expected values come from scikit-learn (roc_auc_score, average_precision_score) and the textbook
/// Dunn (1964) post-hoc statistic.
/// </summary>
public class EvaluationTieReferenceTests
{
    [Fact]
    public void AucRoc_TiedScoresAcrossClasses_CountHalf()
    {
        // sklearn.metrics.roc_auc_score([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == 0.875
        var scores = new[] { 0.1, 0.5, 0.5, 0.9 };
        var labels = new[] { 0.0, 0.0, 1.0, 1.0 };

        double auc = new AUCROCMetric<double>().Compute(scores, labels);

        Assert.Equal(0.875, auc, 12);
    }

    [Fact]
    public void AucRoc_AllScoresTied_IsOneHalf()
    {
        // sklearn.metrics.roc_auc_score([0, 1], [0.5, 0.5]) == 0.5
        double auc = new AUCROCMetric<double>().Compute(new[] { 0.5, 0.5 }, new[] { 0.0, 1.0 });

        Assert.Equal(0.5, auc, 12);
    }

    [Fact]
    public void AveragePrecision_TiedScoresAcrossClasses_TreatsTieAsOneThreshold()
    {
        // sklearn.metrics.average_precision_score([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == 0.5 * 1 + 0.5 * 2/3
        var scores = new[] { 0.1, 0.5, 0.5, 0.9 };
        var labels = new[] { 0.0, 0.0, 1.0, 1.0 };

        double ap = new AUCPRMetric<double>().Compute(scores, labels);

        Assert.Equal(5.0 / 6.0, ap, 12);
    }

    [Fact]
    public void DunnPostHoc_GroupsNotInSortedOrder_UsesEachObservationsOwnRank()
    {
        // Pooled ranks: A = {7, 8, 9}, B = {1, 2, 3}, C = {4, 5, 6}; mean ranks 8, 2, 5.
        // SE = sqrt(N (N + 1) / 12 * (1/3 + 1/3)) = sqrt(5); p = 2 * (1 - Phi(|R_i - R_j| / SE)).
        var groups = new[]
        {
            new[] { 7.0, 8.0, 9.0 },
            new[] { 1.0, 2.0, 3.0 },
            new[] { 4.0, 5.0, 6.0 }
        };

        var results = new KruskalWallisTest<double>().DunnPostHoc(groups);

        Assert.Equal(0.007290358091535644, results[(0, 1)], 9);
        Assert.Equal(0.17971249487899985, results[(0, 2)], 9);
        Assert.Equal(0.17971249487899985, results[(1, 2)], 9);
        Assert.Equal(3, results.Count);
    }
}
