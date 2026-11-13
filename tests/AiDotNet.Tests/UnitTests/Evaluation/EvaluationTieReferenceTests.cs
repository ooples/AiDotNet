using AiDotNet.Evaluation.Engines;
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
    public void MulticlassMcc_MatchesBalancedConfusionReferenceIncludingNaNLabels()
    {
        // Three correct predictions, equal marginals of two per class: (3*6-12)/(36-12) = 1/4.
        Assert.Equal(0.25, new MatthewsCorrelationCoefficientMetric<double>().Compute(
            new[] { 0.0, 1, 1, double.NaN, double.NaN, 0 },
            new[] { 0.0, 0, 1, 1, double.NaN, double.NaN }), 12);
        decimal origin = 10000000000000000m;
        Assert.Equal(-0.5m, new MatthewsCorrelationCoefficientMetric<decimal>().Compute(
            new[] { origin + 1, origin + 2, origin }, new[] { origin, origin + 1, origin + 2 }));
        Assert.Equal(-1.0, new MatthewsCorrelationCoefficientMetric<double>(3).Compute(
            new[] { 2.0, 3 }, new[] { 3.0, 2 }));
        Assert.Equal(0.0, new MatthewsCorrelationCoefficientMetric<double>().Compute(
            new[] { 2.0, 3 }, new[] { 3.0, 2 }));
    }

    [Fact]
    public void MulticlassMcc_LargePermutationKeepsLinearHistograms()
    {
        const int classes = 10000;
        var actual = Enumerable.Range(0, classes).Select(x => (double)x).ToArray();
        var predicted = Enumerable.Range(0, classes).Select(x => (double)((x + 1) % classes)).ToArray();
        Assert.Equal(-1.0 / (classes - 1),
            new MatthewsCorrelationCoefficientMetric<double>().Compute(predicted, actual), 12);
    }

    private sealed class NoFullUniqueEngine : AiDotNet.Tensors.Engines.CpuEngine
    {
        public override (Tensor<T> Values, Tensor<int>? Inverse, Tensor<int>? Counts) TensorUniqueWithInfo<T>(
            Tensor<T> input, bool sorted = true, bool returnInverse = false, bool returnCounts = false)
            => throw new InvalidOperationException("Target inference must bound distinct-label tracking.");
    }

    [Fact]
    public void TargetInference_StopsAtClassLimitWithoutFullUniqueMaterialization()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new NoFullUniqueEngine();
            foreach (int classes in new[] { 2, 511, 512, 100000 })
            {
                var values = Enumerable.Range(0, classes * 4).Select(x => (double)(x % classes)).ToArray();
                var expected = classes == 2 ? AiDotNet.Enums.PredictionType.BinaryClassification
                    : classes < 512 ? AiDotNet.Enums.PredictionType.MultiClass : AiDotNet.Enums.PredictionType.Regression;
                Assert.Equal(expected, AiDotNet.Evaluation.PredictionTypeInference.Infer(new Vector<double>(values)));
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }


    [Theory]
    [InlineData(2, 12.706204736432095)]
    [InlineData(3, 4.302652729696142)]
    [InlineData(5, 2.7764451051977987)]
    [InlineData(10, 2.2621571628540993)]
    [InlineData(30, 2.045229642132703)]
    [InlineData(100, 1.9842169515086827)]
    public void CrossValidationIntervals_UseStudentTReferenceQuantiles(int folds, double criticalValue)
    {
        // NIST two-sided 95% Student-t critical values, with df = folds - 1.
        var summarize = typeof(CrossValidationEngine<double>).GetMethod("Summarize",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;
        var values = Enumerable.Range(0, folds).Select(x => (double)x).ToList();
        var (_, standardDeviation, margin) = Assert.IsType<(double, double, double)>(
            summarize.Invoke(null, new object[] { values }));
        Assert.Equal(criticalValue, margin * Math.Sqrt(folds) / standardDeviation, 6);
        var (_, singletonDeviation, singletonMargin) = Assert.IsType<(double, double, double)>(
            summarize.Invoke(null, new object[] { new List<double> { 1 } }));
        Assert.Equal(0, singletonDeviation);
        Assert.Equal(0, singletonMargin);
    }


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

    [Fact]
    public void IsotonicCalibration_RunOfThreeViolators_MatchesSklearnPav()
    {
        // scikit-learn 1.9.1:
        // IsotonicRegression().fit_transform([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 0, 0, 1, 0, 1])
        // == [1/3, 1/3, 1/3, 0.5, 0.5, 1.0]. The first three samples form one pooled block of three.
        var probabilities = new[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
        var actuals = new[] { 1.0, 0.0, 0.0, 1.0, 0.0, 1.0 };

        var calibrated = new CalibrationEngine<double>().IsotonicCalibration(probabilities, actuals);

        var expected = new[] { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5, 1.0 };
        Assert.Equal(expected.Length, calibrated.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], calibrated[i], 12);
    }

    [Fact]
    public void IsotonicCalibration_UnsortedInput_ReturnsFitInInputOrder()
    {
        // scikit-learn 1.9.1:
        // IsotonicRegression().fit_transform([0.6, 0.1, 0.4, 0.2, 0.5, 0.3], [1, 0, 0, 1, 0, 1])
        // == [1.0, 0.0, 0.5, 0.5, 0.5, 0.5] (sorted labels 0, 1, 1, 0, 0, 1 pool the middle four to 0.5).
        var probabilities = new[] { 0.6, 0.1, 0.4, 0.2, 0.5, 0.3 };
        var actuals = new[] { 1.0, 0.0, 0.0, 1.0, 0.0, 1.0 };

        var calibrated = new CalibrationEngine<double>().IsotonicCalibration(probabilities, actuals);

        var expected = new[] { 1.0, 0.0, 0.5, 0.5, 0.5, 0.5 };
        Assert.Equal(expected.Length, calibrated.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], calibrated[i], 12);
    }
}
