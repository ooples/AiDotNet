using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramMetricAggregatorTests
{
    private static Dictionary<string, ProgramMetricValue> Metrics(params (string Name, ProgramMetricValue Value)[] entries)
    {
        var result = new Dictionary<string, ProgramMetricValue>(StringComparer.Ordinal);
        foreach ((string name, ProgramMetricValue value) in entries) result[name] = value;
        return result;
    }

    [Fact]
    public void CombinedScoreWinsOverEveryOtherMetric()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("combined_score", ProgramMetricValue.Number(0.75)),
            ("accuracy", ProgramMetricValue.Number(0.1)),
            ("speed", ProgramMetricValue.Number(0.2))));

        Assert.True(result.UsedCombinedScore);
        Assert.Equal(0.75, result.Value, 12);
        Assert.Equal(new[] { "combined_score" }, result.ContributingMetrics);
    }

    [Fact]
    public void MissingCombinedScoreFallsBackToTheMeanOfNumericMetrics()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.8)),
            ("speed", ProgramMetricValue.Number(0.4))));

        Assert.False(result.UsedCombinedScore);
        Assert.Equal(0.6, result.Value, 12);
        Assert.Equal(new[] { "accuracy", "speed" }, result.ContributingMetrics);
    }

    [Fact]
    public void EmptyMetricsScoreZeroAndSayWhy()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator()
            .Aggregate(new Dictionary<string, ProgramMetricValue>(StringComparer.Ordinal));

        Assert.Equal(0.0, result.Value, 12);
        Assert.Empty(result.ContributingMetrics);
        Assert.Contains(result.Issues, issue => issue.Reason == ProgramMetricIssueReason.NoNumericValues);
    }

    [Fact]
    public void BooleanFlagsAreNeverAveragedAsScores()
    {
        // metrics_utils.safe_numeric_average excludes bool so {"error": 0.0, "timeout": True} scores 0.0, not 0.5.
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("error", ProgramMetricValue.Number(0.0)),
            ("timeout", ProgramMetricValue.Flag(true))));

        Assert.Equal(0.0, result.Value, 12);
        Assert.Equal(new[] { "error" }, result.ContributingMetrics);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "timeout" && issue.Reason == ProgramMetricIssueReason.BooleanFlag);
    }

    [Fact]
    public void NotANumberValuesAreDroppedFromTheMean()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.8)),
            ("broken", ProgramMetricValue.Number(double.NaN))));

        Assert.Equal(0.8, result.Value, 12);
        Assert.True(result.HasFiniteValue);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "broken" && issue.Reason == ProgramMetricIssueReason.NotANumber);
    }

    [Fact]
    public void InfiniteValuesStillContributeButAreReportedAsUnusable()
    {
        // safe_numeric_average filters NaN only, so an infinity reaches the mean upstream too.
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.8)),
            ("runaway", ProgramMetricValue.Number(double.PositiveInfinity))));

        Assert.False(result.HasFiniteValue);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "runaway" && issue.Reason == ProgramMetricIssueReason.NotFinite);
    }

    [Fact]
    public void FeatureDimensionsAreExcludedFromTheMean()
    {
        var options = new ProgramMetricAggregationOptions();
        options.ExcludedFeatureDimensions.Add("length");
        ProgramMetricAggregationResult result = new ProgramMetricAggregator(options).Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.8)),
            ("length", ProgramMetricValue.Number(400.0))));

        Assert.Equal(0.8, result.Value, 12);
        Assert.Equal(new[] { "accuracy" }, result.ContributingMetrics);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "length" && issue.Reason == ProgramMetricIssueReason.ExcludedFeatureDimension);
    }

    [Fact]
    public void ExcludingEveryMetricFallsBackToAveragingAllOfThem()
    {
        // get_fitness_score falls back to safe_numeric_average(metrics) when no non-feature metric survives.
        var options = new ProgramMetricAggregationOptions();
        options.ExcludedFeatureDimensions.Add("length");
        options.ExcludedFeatureDimensions.Add("depth");
        ProgramMetricAggregationResult result = new ProgramMetricAggregator(options).Aggregate(Metrics(
            ("length", ProgramMetricValue.Number(2.0)),
            ("depth", ProgramMetricValue.Number(4.0))));

        Assert.Equal(3.0, result.Value, 12);
        Assert.Equal(new[] { "depth", "length" }, result.ContributingMetrics);
        Assert.DoesNotContain(result.Issues,
            issue => issue.Reason == ProgramMetricIssueReason.ExcludedFeatureDimension);
    }

    [Fact]
    public void NonNumericMetricsAreReportedRatherThanSilentlySkipped()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.5)),
            ("stderr", ProgramMetricValue.Text("ZeroDivisionError: division by zero"))));

        Assert.Equal(0.5, result.Value, 12);
        ProgramMetricIssue issue = Assert.Single(result.Issues);
        Assert.Equal("stderr", issue.MetricName);
        Assert.Equal(ProgramMetricIssueReason.NonNumericText, issue.Reason);
        Assert.NotEqual(string.Empty, issue.Description);
    }

    [Fact]
    public void ATextAccuracyIsReportedInsteadOfLookingLikeAZeroScore()
    {
        // The failure mode the reference rule hides: a metric reported as text scores 0.0 with no explanation.
        ProgramMetricAggregationResult result = new ProgramMetricAggregator()
            .Aggregate(Metrics(("accuracy", ProgramMetricValue.Text("0.9"))));

        Assert.Equal(0.0, result.Value, 12);
        Assert.Empty(result.ContributingMetrics);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "accuracy" && issue.Reason == ProgramMetricIssueReason.NonNumericText);
        Assert.Contains(result.Issues, issue => issue.Reason == ProgramMetricIssueReason.NoNumericValues);
    }

    [Fact]
    public void ATextCombinedScoreIsConvertedTheWayPythonFloatWould()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("combined_score", ProgramMetricValue.Text("0.25")),
            ("accuracy", ProgramMetricValue.Number(0.9))));

        Assert.True(result.UsedCombinedScore);
        Assert.Equal(0.25, result.Value, 12);
    }

    [Fact]
    public void AnUnconvertibleTextCombinedScoreFallsThroughToTheMean()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("combined_score", ProgramMetricValue.Text("not a number")),
            ("accuracy", ProgramMetricValue.Number(0.9))));

        Assert.False(result.UsedCombinedScore);
        Assert.Equal(0.9, result.Value, 12);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "combined_score" && issue.Reason == ProgramMetricIssueReason.NonNumericText);
    }

    [Fact]
    public void NonFiniteTextIsRejectedDeterministicallyOnEveryFramework()
    {
        // Python float("inf") succeeds; parsing it differs between .NET Framework and .NET, so it is refused here.
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(Metrics(
            ("combined_score", ProgramMetricValue.Text("inf")),
            ("accuracy", ProgramMetricValue.Number(0.4))));

        Assert.False(result.UsedCombinedScore);
        Assert.Equal(0.4, result.Value, 12);
        Assert.True(result.HasFiniteValue);
    }

    [Fact]
    public void AFlagCombinedScoreScoresOneOrZeroAndSaysSo()
    {
        // Python float(True) is 1.0, so upstream would return 1.0 with no indication that a flag was scored.
        ProgramMetricAggregationResult result = new ProgramMetricAggregator()
            .Aggregate(Metrics(("combined_score", ProgramMetricValue.Flag(true))));

        Assert.True(result.UsedCombinedScore);
        Assert.Equal(1.0, result.Value, 12);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "combined_score" && issue.Reason == ProgramMetricIssueReason.BooleanFlag);
    }

    [Fact]
    public void ANotANumberCombinedScoreIsReturnedButMarkedUnusable()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator()
            .Aggregate(Metrics(("combined_score", ProgramMetricValue.Number(double.NaN))));

        Assert.True(result.UsedCombinedScore);
        Assert.False(result.HasFiniteValue);
        Assert.Contains(result.Issues, issue => issue.Reason == ProgramMetricIssueReason.NotANumber);
    }

    [Fact]
    public void MeanStrategyIgnoresTheCombinedScoreShortcut()
    {
        var options = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Mean };
        ProgramMetricAggregationResult result = new ProgramMetricAggregator(options).Aggregate(Metrics(
            ("combined_score", ProgramMetricValue.Number(0.9)),
            ("accuracy", ProgramMetricValue.Number(0.1))));

        Assert.False(result.UsedCombinedScore);
        Assert.Equal(0.5, result.Value, 12);
    }

    [Fact]
    public void WeightedStrategyCombinesDeclaredWeights()
    {
        var options = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Weighted };
        options.Weights["accuracy"] = 3.0;
        options.Weights["speed"] = 1.0;
        ProgramMetricAggregationResult result = new ProgramMetricAggregator(options).Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(1.0)),
            ("speed", ProgramMetricValue.Number(0.0))));

        Assert.Equal(0.75, result.Value, 12);
        Assert.Equal(EvolutionOptimizationDirection.Maximize, new ProgramMetricAggregator(options).PreferredDirection);
    }

    [Fact]
    public void AWeightedMetricTheEvaluatorNeverReportedIsAConfigurationError()
    {
        var options = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Weighted };
        options.Weights["accuracy"] = 1.0;
        options.Weights["latency"] = 1.0;
        var aggregator = new ProgramMetricAggregator(options);

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            aggregator.Aggregate(Metrics(("accuracy", ProgramMetricValue.Number(1.0)))));

        Assert.Contains("latency", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingWeightedMetricCanBeReportedInsteadOfThrowing()
    {
        var options = new ProgramMetricAggregationOptions
        {
            Strategy = ProgramMetricAggregationStrategy.Weighted,
            RequireAllWeightedMetrics = false
        };
        options.Weights["accuracy"] = 1.0;
        options.Weights["latency"] = 3.0;
        ProgramMetricAggregationResult result = new ProgramMetricAggregator(options)
            .Aggregate(Metrics(("accuracy", ProgramMetricValue.Number(0.6))));

        Assert.Equal(0.6, result.Value, 12);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "latency" && issue.Reason == ProgramMetricIssueReason.MissingMetric);
    }

    [Fact]
    public void AReportedMetricWithNoDeclaredWeightIsReportedRatherThanIgnored()
    {
        var options = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Weighted };
        options.Weights["accuracy"] = 1.0;
        ProgramMetricAggregationResult result = new ProgramMetricAggregator(options).Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.6)),
            ("memory", ProgramMetricValue.Number(120.0))));

        Assert.Equal(0.6, result.Value, 12);
        Assert.Contains(result.Issues,
            issue => issue.MetricName == "memory" && issue.Reason == ProgramMetricIssueReason.NoWeightDeclared);
    }

    [Fact]
    public void TchebycheffScoresTheLargestWeightedShortfallAndIsMinimized()
    {
        var options = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Tchebycheff };
        options.Weights["accuracy"] = 1.0;
        options.Weights["speed"] = 2.0;
        options.ReferencePoint["accuracy"] = 1.0;
        options.ReferencePoint["speed"] = 1.0;
        var aggregator = new ProgramMetricAggregator(options);

        ProgramMetricAggregationResult result = aggregator.Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.5)),
            ("speed", ProgramMetricValue.Number(0.9))));

        // shortfalls are 1*0.5 = 0.5 and 2*0.1 = 0.2, so the worst is 0.5.
        Assert.Equal(0.5, result.Value, 12);
        Assert.Equal(EvolutionOptimizationDirection.Minimize, aggregator.PreferredDirection);
    }

    [Fact]
    public void TchebycheffAugmentationAddsTheSummedShortfalls()
    {
        var options = new ProgramMetricAggregationOptions
        {
            Strategy = ProgramMetricAggregationStrategy.Tchebycheff,
            AugmentationCoefficient = 0.5
        };
        options.Weights["accuracy"] = 1.0;
        options.Weights["speed"] = 2.0;
        options.ReferencePoint["accuracy"] = 1.0;
        options.ReferencePoint["speed"] = 1.0;

        ProgramMetricAggregationResult result = new ProgramMetricAggregator(options).Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.5)),
            ("speed", ProgramMetricValue.Number(0.9))));

        Assert.Equal(0.5 + (0.5 * 0.7), result.Value, 12);
    }

    [Fact]
    public void TchebycheffRequiresAReferenceValueForEveryWeightedMetric()
    {
        var options = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Tchebycheff };
        options.Weights["accuracy"] = 1.0;

        Assert.Throws<InvalidOperationException>(() => new ProgramMetricAggregator(options));
    }

    [Fact]
    public void AWeightedStrategyWithNoPositiveWeightIsRejectedAtConstructionTime()
    {
        var options = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Weighted };
        options.Weights["accuracy"] = 0.0;

        Assert.Throws<InvalidOperationException>(() => new ProgramMetricAggregator(options));
    }

    [Fact]
    public void ANegativeWeightIsRejectedAtConstructionTime()
    {
        var options = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Weighted };
        options.Weights["accuracy"] = -1.0;

        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramMetricAggregator(options));
    }

    [Fact]
    public void OptionsAreSnapshotSoLaterMutationCannotChangeAValidatedAggregation()
    {
        var options = new ProgramMetricAggregationOptions();
        var aggregator = new ProgramMetricAggregator(options);
        options.ExcludedFeatureDimensions.Add("accuracy");

        ProgramMetricAggregationResult result = aggregator.Aggregate(Metrics(
            ("accuracy", ProgramMetricValue.Number(0.8)),
            ("speed", ProgramMetricValue.Number(0.4))));

        Assert.Equal(0.6, result.Value, 12);
    }

    [Fact]
    public void TheStaticUpstreamRuleExcludesTheGivenFeatureDimensions()
    {
        ProgramMetricAggregationResult result = ProgramMetricAggregator.UpstreamFitnessScore(
            Metrics(
                ("accuracy", ProgramMetricValue.Number(0.8)),
                ("length", ProgramMetricValue.Number(400.0))),
            new[] { "length" });

        Assert.Equal(0.8, result.Value, 12);
        Assert.Equal(ProgramMetricAggregationStrategy.CombinedScoreOrMean, result.Strategy);
    }

    [Fact]
    public void TheNumericOverloadAcceptsAPlainDictionary()
    {
        ProgramMetricAggregationResult result = new ProgramMetricAggregator().Aggregate(
            new Dictionary<string, double>(StringComparer.Ordinal) { ["a"] = 1.0, ["b"] = 3.0 });

        Assert.Equal(2.0, result.Value, 12);
    }

    [Fact]
    public void MetricValuesCompareByValueAndNeverReadAFlagAsANumber()
    {
        Assert.Equal(ProgramMetricValue.Number(1.5), ProgramMetricValue.Number(1.5));
        Assert.NotEqual(ProgramMetricValue.Number(1.0), ProgramMetricValue.Flag(true));
        Assert.False(ProgramMetricValue.Flag(true).TryGetNumber(allowTextConversion: true, out _));
        Assert.True(ProgramMetricValue.Text(" 2.5 ").TryGetNumber(allowTextConversion: true, out double parsed));
        Assert.Equal(2.5, parsed, 12);
    }

    [Fact]
    public void AggregationRejectsAnEmptyMetricName()
    {
        var aggregator = new ProgramMetricAggregator();
        var metrics = new Dictionary<string, ProgramMetricValue>(StringComparer.Ordinal)
        {
            [" "] = ProgramMetricValue.Number(1.0)
        };

        Assert.Throws<ArgumentException>(() => aggregator.Aggregate(metrics));
    }
}
