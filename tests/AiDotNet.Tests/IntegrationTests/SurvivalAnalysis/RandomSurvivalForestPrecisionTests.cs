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
}
