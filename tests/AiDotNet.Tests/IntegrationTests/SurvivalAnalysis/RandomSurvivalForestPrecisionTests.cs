using System;
using System.Reflection;
using AiDotNet.SurvivalAnalysis;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SurvivalAnalysis;

/// <summary>
/// Regression tests for the log-rank split statistic of <see cref="RandomSurvivalForest{T}"/>.
/// The hypergeometric variance term N^2 (N - 1) was evaluated in 32-bit integers, which overflows
/// as soon as a node has 1,291 or more subjects at risk, producing negative variance contributions
/// and a wrong split score.
/// </summary>
public class RandomSurvivalForestPrecisionTests
{
    [Fact]
    public void LogRankStatistic_WithMoreThan1290SubjectsAtRisk_MatchesDoublePrecisionReference()
    {
        const int n = 1400;
        var times = new Vector<double>(n);
        var events = new Vector<int>(n);
        var left = new int[n / 2];
        var right = new int[n / 2];
        for (int i = 0; i < n; i++)
        {
            times[i] = i + 1;
            events[i] = 1;
            if (i % 2 == 0)
            {
                left[i / 2] = i;
            }
            else
            {
                right[i / 2] = i;
            }
        }

        var method = typeof(RandomSurvivalForest<double>).GetMethod(
            "ComputeLogRankStatistic", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(method);

        var forest = new RandomSurvivalForest<double>(numTrees: 1, seed: 7);
        object? raw = method?.Invoke(forest, new object[] { times, events, left, right });
        double actual = Assert.IsType<double>(raw);

        // Independent double-precision log-rank reference. With distinct times, one event occurs at
        // each time t = 1..n; subject i (time i + 1) is at risk at t while i + 1 >= t.
        double observed = 0, expected = 0, variance = 0;
        for (int t = 1; t <= n; t++)
        {
            double leftAtRisk = 0, rightAtRisk = 0;
            for (int i = 0; i < n; i++)
            {
                if (i + 1 < t) continue;
                if (i % 2 == 0) leftAtRisk++; else rightAtRisk++;
            }

            double total = leftAtRisk + rightAtRisk;
            if (total <= 1) continue;

            double leftEvents = (t - 1) % 2 == 0 ? 1 : 0;
            observed += leftEvents;
            expected += leftAtRisk / total;
            variance += leftAtRisk * rightAtRisk * (total - 1) / (total * total * (total - 1));
        }

        double reference = Math.Abs((observed - expected) / Math.Sqrt(variance));

        // Old int arithmetic gave 0.12597...; the correct statistic is 0.11383...
        Assert.Equal(reference, actual, 9);
    }
    /// <summary>
    /// scikit-survival 0.28.0 reference: RandomSurvivalForest(n_estimators=3, bootstrap=False,
    /// max_features=None, min_samples_leaf=3, min_samples_split=6, max_depth=None) on this toy set.
    /// Without bootstrap and with every feature considered, each tree is the same for any random_state
    /// (checked for random_state 0..5), so the forest is deterministic.
    /// </summary>
    private static (Matrix<double> X, Vector<double> Times, Vector<int> Events, Matrix<double> Query, Vector<double> QueryTimes) ScikitSurvivalToySet()
    {
        double[,] x =
        {
            { 3, 11 }, { 7, 2 }, { 1, 14 }, { 12, 5 }, { 9, 9 }, { 15, 1 }, { 4, 13 }, { 10, 7 },
            { 2, 16 }, { 13, 3 }, { 6, 12 }, { 16, 8 }, { 8, 4 }, { 11, 15 }, { 5, 6 }, { 14, 10 },
        };
        var times = new Vector<double>(new[] { 5.0, 12.0, 3.0, 18.0, 9.0, 21.0, 4.0, 14.0, 2.0, 20.0, 7.0, 16.0, 12.0, 10.0, 8.0, 17.0 });
        var events = new Vector<int>(new[] { 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1 });
        double[,] query = { { 3, 11 }, { 8, 8 }, { 14, 2 }, { 6, 15 } };

        // scikit-survival unique_times_ (event and censoring times)
        var queryTimes = new Vector<double>(new[] { 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 17.0, 18.0, 20.0, 21.0 });
        return (new Matrix<double>(x), times, events, new Matrix<double>(query), queryTimes);
    }

    private static RandomSurvivalForest<double> FitScikitSurvivalToyForest()
    {
        var (x, times, events, _, _) = ScikitSurvivalToySet();
        var forest = new RandomSurvivalForest<double>(
            numTrees: 3, maxDepth: 50, minSamplesLeaf: 3, maxFeatures: 2, seed: 0, bootstrap: false);
        forest.FitSurvival(x, times, events);
        return forest;
    }

    [Fact]
    public void EnsembleMortality_MatchesScikitSurvival()
    {
        var (_, _, _, query, _) = ScikitSurvivalToySet();
        var risk = FitScikitSurvivalToyForest().PredictHazardRatio(query);

        // RandomSurvivalForest.predict: ensemble CHF summed over the event times (Ishwaran et al. 2008).
        double[] expected = { 17.666666666666668, 8.833333333333332, 2.0, 11.0 };
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], risk[i], 12);
    }

    [Fact]
    public void EnsembleMortality_WithOneEventTimePerSplitBlock_MatchesScikitSurvival()
    {
        // The split search sums the log-rank terms over blocks of event times to bound memory; with one event
        // time per block the forest must be the same as with all event times at once.
        var (x, times, events, query, _) = ScikitSurvivalToySet();
        var forest = new RandomSurvivalForest<double>(
            numTrees: 3, maxDepth: 50, minSamplesLeaf: 3, maxFeatures: 2, seed: 0, bootstrap: false)
        {
            SplitBlockElements = 1,
        };
        forest.FitSurvival(x, times, events);
        var risk = forest.PredictHazardRatio(query);

        double[] expected = { 17.666666666666668, 8.833333333333332, 2.0, 11.0 };
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], risk[i], 12);
    }

    [Fact]
    public void EnsembleCumulativeHazard_MatchesScikitSurvival()
    {
        var (_, _, _, query, queryTimes) = ScikitSurvivalToySet();
        var chf = FitScikitSurvivalToyForest().PredictCumulativeHazard(queryTimes, query);

        double[,] expected =
        {
            { 1.0 / 3, 0.8333333333333334, 0.8333333333333334, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333, 1.8333333333333333 },
            { 0, 0, 0, 0, 0, 0, 0.25, 0.25, 0.9166666666666666, 1.9166666666666667, 1.9166666666666667, 1.9166666666666667, 1.9166666666666667, 1.9166666666666667, 1.9166666666666667 },
            { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1.5 },
            { 0, 0, 0, 0, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 },
        };
        for (int i = 0; i < expected.GetLength(0); i++)
            for (int j = 0; j < expected.GetLength(1); j++)
                Assert.Equal(expected[i, j], chf[i, j], 12);
    }

    [Fact]
    public void EnsembleSurvival_MatchesScikitSurvival()
    {
        var (_, _, _, query, queryTimes) = ScikitSurvivalToySet();
        var survival = FitScikitSurvivalToyForest().PredictSurvivalProbability(query, queryTimes);

        double[,] expected =
        {
            { 2.0 / 3, 1.0 / 3, 1.0 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 1, 1, 1, 1, 1, 1, 0.75, 0.75, 0.25, 0, 0, 0, 0, 0, 0 },
            { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0 },
            { 1, 1, 1, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        };
        for (int i = 0; i < expected.GetLength(0); i++)
            for (int j = 0; j < expected.GetLength(1); j++)
                Assert.Equal(expected[i, j], survival[i, j], 12);
    }
}
