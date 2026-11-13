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
    [Theory]
    [InlineData(0.0, 1, 1.0)]
    [InlineData(1.0, 1, 0.31731050786291415)]
    [InlineData(3.841458820694124, 1, 0.05)]
    [InlineData(1000.0, 1, 1.7958327848006388e-219)]
    [InlineData(100.0, 2, 1.9287498479639178e-22)]
    [InlineData(100.0, 4, 9.836624224615988e-21)]
    [InlineData(1000.0, 1000, 0.49405285382923964)]
    [InlineData(10.0, 10, 0.44049328506521257)]
    [InlineData(double.PositiveInfinity, 1, 0.0)]
    public void KruskalChiSquareTail_MatchesClosedFormAndReferenceValues(double statistic, int degrees, double expected)
    {
        var survival = typeof(KruskalWallisTest<double>).GetMethod("ChiSquareSurvival",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;
        double actual = Assert.IsType<double>(survival.Invoke(null, new object[] { statistic, degrees }));
        if (expected == 0) Assert.Equal(0, actual);
        else Assert.InRange(actual / expected, 1 - 1e-10, 1 + 1e-10);
    }

    [Fact]
    public void KruskalOrderedGroups_ReturnATinyPositiveTailWithoutFullUniqueScanning()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new NoFullUniqueEngine();
            double[][] groups =
            {
                Enumerable.Range(0, 667).Select(x => (double)x).ToArray(),
                Enumerable.Range(667, 667).Select(x => (double)x).ToArray()
            };
            var result = new KruskalWallisTest<double>().Test(groups);
            Assert.Equal(3.0 * 667 * 667 / 1335, result.Statistic, 9);
            // df=1 survival is erfc(sqrt(H/2)); independently evaluated with Python's math.erfc.
            Assert.InRange(result.PValue / 2.034626876138803e-219, 1 - 1e-9, 1 + 1e-9);
            Assert.True(result.IsSignificant);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void Dunn_TiedObservationsUseTheCorrectedVariance()
    {
        double[][] groups = { new[] { 1.0, 1, 2 }, new[] { 2.0, 3, 3 } };
        // Ranks are [1.5,1.5,3.5] and [3.5,5.5,5.5]; tie sum=18, variance=3.2, z^2=10/3.
        const double expected = 0.06788915486182903;
        var test = new KruskalWallisTest<double>();
        Assert.Equal(expected, test.DunnPostHoc(groups)[(0, 1)], 9);
        Assert.Equal(10.0 / 3, test.Test(groups).Statistic, 12);
        Assert.Equal(expected, test.Test(groups).PValue, 12);
        decimal origin = 10000000000000000m;
        var nativeGroups = groups.Select(group => group.Select(value => origin + (decimal)value).ToArray()).ToArray();
        Assert.Equal(expected, new KruskalWallisTest<decimal>().DunnPostHoc(nativeGroups)[(0, 1)], 9);
        Assert.Equal(1, test.DunnPostHoc(new[] { new[] { 1.0, 1 }, new[] { 1.0, 1 } })[(0, 1)]);
    }

    [Fact]
    public void Dunn_RejectsMissingOrEmptyGroupsBeforeDividingByTheirSizes()
    {
        var test = new KruskalWallisTest<double>();
        Assert.Throws<ArgumentException>(() => test.DunnPostHoc(new[] { new[] { 1.0 } }));
        Assert.Throws<ArgumentException>(() => test.DunnPostHoc(new[] { Array.Empty<double>(), new[] { 1.0 } }));
    }


    [Fact]
    public void RankMetrics_KeepAdjacentDecimalAndInt64ValuesDistinct()
    {
        decimal origin = 10000000000000000m;
        var decimalScore = new AiDotNet.Evaluation.Metrics.Regression.SpearmanCorrelationMetric<decimal>().Compute(
            new[] { origin + 1, origin, origin + 3, origin + 1 },
            new[] { origin + 3, origin + 1, origin, origin + 1 });
        Assert.Equal(-0.5m, decimalScore);
        long large = 9007199254740992L;
        var integerRanks = new NemenyiPostHocTest<long>().Test(new[]
        {
            new[] { large, large + 1, large + 2 }, new[] { large + 1, large + 2, large + 3 }
        });
        Assert.Equal(new[] { 2.0, 1.0 }, integerRanks.AverageRanks);
    }

    [Fact]
    public void Nemenyi_ReportsFractionalDoubleRanksForIntegralInputs()
    {
        var result = new NemenyiPostHocTest<int>().Test(new[] { new[] { 1, 2, 3 }, new[] { 1, 2, 3 } });
        Assert.Equal(new[] { 1.5, 1.5 }, Assert.IsType<double[]>(result.AverageRanks));
    }

    [Fact]
    public void DatasetRanks_PreserveDecimalOrderingAndTheNativeTieTolerance()
    {
        decimal origin = 10000000000000000m;
        decimal[][] samples =
        {
            new[] { origin + 2, origin, origin + 1 },
            new[] { origin + 1, origin + 2, origin + 1 },
            new[] { origin, origin + 1, origin + 2 }
        };
        var result = new NemenyiPostHocTest<decimal>().Test(samples);
        Assert.Equal(6.5 / 3, result.AverageRanks[0], 12);
        Assert.Equal(5.5 / 3, result.AverageRanks[1], 12);
        Assert.Equal(2, result.AverageRanks[2], 12);
        Assert.Equal(2.0 / 35, (double)new FriedmanTest<decimal>().Test(samples).Statistic, 12);
        var near = new NemenyiPostHocTest<decimal>().Test(new[]
        {
            Enumerable.Repeat(origin, 3).ToArray(),
            Enumerable.Repeat(origin + 0.00000000001m, 3).ToArray()
        });
        Assert.Equal(new[] { 1.5, 1.5 }, near.AverageRanks);
        var extremes = new NemenyiPostHocTest<decimal>().Test(new[]
        {
            Enumerable.Repeat(decimal.MinValue, 3).ToArray(),
            Enumerable.Repeat(decimal.MaxValue, 3).ToArray()
        });
        Assert.Equal(new[] { 2.0, 1.0 }, extremes.AverageRanks);
    }

    [Fact]
    public void KruskalAndDunn_PreserveDecimalOrderingAndDoubleProbabilityContracts()
    {
        decimal origin = 10000000000000000m;
        decimal[][] groups = { new[] { origin, origin + 1, origin + 2 }, new[] { origin + 3, origin + 4, origin + 5 } };
        var test = new KruskalWallisTest<decimal>();
        Assert.Equal(27.0 / 7, (double)test.Test(groups).Statistic, 12);
        System.Collections.Generic.Dictionary<(int, int), double> probabilities = test.DunnPostHoc(groups);
        Assert.Equal(0.04953461343562674, probabilities[(0, 1)], 9);
        var integer = new KruskalWallisTest<int>().DunnPostHoc(new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } });
        Assert.Equal(0.04953461343562674, integer[(0, 1)], 9);
    }

    [Fact]
    public void Wilcoxon_KeepsNativeDifferencesBeforeRanking()
    {
        decimal origin = 10000000000000000m;
        Assert.Equal(5m, new WilcoxonSignedRankTest<decimal>().Test(
            new[] { origin + 1, origin + 2, origin + 3, origin + 4, origin + 5 },
            new[] { origin, origin, origin, origin, origin + 10 }).Statistic);
        long large = 9007199254740992L;
        Assert.Equal(5L, new WilcoxonSignedRankTest<long>().Test(
            new[] { large + 1, large + 2, large + 3, large + 4, large + 5 },
            new[] { large, large, large, large, large + 10 }).Statistic);
    }

    [Fact]
    public void Wilcoxon_HandlesFullDecimalAndIntegerDifferenceRangesExactly()
    {
        decimal high = decimal.MaxValue, low = decimal.MinValue;
        Assert.Equal(7m, new WilcoxonSignedRankTest<decimal>().Test(
            new[] { high, high - 1, high - 2, low, low + 1 },
            new[] { low, low, low, high, high }).Statistic);
        long maximum = long.MaxValue, minimum = long.MinValue;
        Assert.Equal(7L, new WilcoxonSignedRankTest<long>().Test(
            new[] { maximum, maximum - 1, maximum - 2, minimum, minimum + 1 },
            new[] { minimum, minimum, minimum, maximum, maximum }).Statistic);
        ulong maxUnsigned = ulong.MaxValue;
        Assert.Equal(7UL, new WilcoxonSignedRankTest<ulong>().Test(
            new[] { maxUnsigned, maxUnsigned - 1, maxUnsigned - 2, 0UL, 1UL },
            new[] { 0UL, 0UL, 0UL, maxUnsigned, maxUnsigned }).Statistic);
    }


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
