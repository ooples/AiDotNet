using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Models.Results;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>
/// Covers ranking a finished search by a metric other than the one it optimised. A run chases one number; an
/// evaluation reports several, and the question worth asking afterwards is often about one of the others.
/// </summary>
public sealed class EvolutionMetricQueryTests
{
    [Fact]
    public void BestByRanksOnTheNamedMetricRatherThanOnQuality()
    {
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 10, metrics: Metrics(("accuracy", 0.5), ("latency", 90))),
            Entry(2, quality: 90, metrics: Metrics(("accuracy", 0.1), ("latency", 10))));

        Assert.Equal(90, archive.Best?.Evaluation.Quality);
        Assert.Equal(0.5, archive.BestBy("accuracy")?.Evaluation.Metrics["accuracy"]);
    }

    [Fact]
    public void ADirectionCanBePassedForAMetricThatReadsTheOtherWay()
    {
        // Latency inside a maximising run: without an explicit direction the query would hand the answer to the
        // slowest candidate, which is the opposite of what the caller meant.
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 10, metrics: Metrics(("latency", 90))),
            Entry(2, quality: 90, metrics: Metrics(("latency", 10))));

        Assert.Equal(90, archive.BestBy("latency")?.Evaluation.Metrics["latency"]);
        Assert.Equal(10, archive.BestBy("latency", EvolutionOptimizationDirection.Minimize)?.Evaluation.Metrics["latency"]);
    }

    [Fact]
    public void ACandidateThatNeverReportedTheMetricIsAbsentRatherThanScoredZero()
    {
        // The obvious implementation reads the metric with a zero default, which quietly hands every minimising
        // query to whichever candidate simply failed to measure. Absent has to mean absent.
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 10, metrics: Metrics(("cost", 5))),
            Entry(2, quality: 90, metrics: Metrics(("unrelated", 1))));

        EvolutionArchiveEntry<TestGenome>? cheapest = archive.BestBy("cost", EvolutionOptimizationDirection.Minimize);

        Assert.NotNull(cheapest);
        Assert.Equal(5, cheapest.Evaluation.Metrics["cost"]);
        Assert.Single(archive.WithMetric("cost"));
    }

    [Fact]
    public void AnEvaluationCannotCarryANonFiniteMetricInTheFirstPlace()
    {
        // The archive is protected at the source, so a NaN never reaches a query from a live run.
        Assert.ThrowsAny<ArgumentException>(() => Entry(1, quality: 10, metrics: Metrics(("score", double.NaN))));
    }

    [Fact]
    public void ASummaryHoldingANonFiniteMetricRanksItOutRatherThanLettingItWin()
    {
        // A summary's metrics are a settable dictionary, so this is the reachable case: a hand-built or deserialized
        // summary carrying a NaN. Comparisons against NaN are all false, so an unguarded sort would seat it wherever
        // the sort happened to leave it and BestBy could return it.
        var summary = new EvolutionRunSummary { Direction = EvolutionOptimizationDirection.Maximize };
        summary.Elites.Add(Elite("broken", metric: double.NaN));
        summary.Elites.Add(Elite("real", metric: 3));

        Assert.Equal("real", summary.BestBy("score")?.GenomeId);
        Assert.Equal("real", Assert.Single(summary.TopBy("score", 5)).GenomeId);
        Assert.Equal(new[] { "score" }, summary.MetricNames());
    }

    [Fact]
    public void NothingReportingTheMetricGivesNothingBack()
    {
        MapElitesArchive<TestGenome> archive = Archive(Entry(1, quality: 10, metrics: Metrics(("accuracy", 0.5))));

        Assert.Null(archive.BestBy("nonexistent"));
        Assert.Empty(archive.TopBy("nonexistent", 5));
        Assert.Empty(archive.WithMetric("nonexistent"));
    }

    [Fact]
    public void TopByReturnsTheBestFirstAndStopsAtTheRequestedCount()
    {
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 1, metrics: Metrics(("accuracy", 0.2))),
            Entry(2, quality: 2, metrics: Metrics(("accuracy", 0.9))),
            Entry(3, quality: 3, metrics: Metrics(("accuracy", 0.5))));

        IReadOnlyList<EvolutionArchiveEntry<TestGenome>> top = archive.TopBy("accuracy", 2);

        Assert.Equal(new[] { 0.9, 0.5 }, top.Select(entry => entry.Evaluation.Metrics["accuracy"]));
        Assert.Equal(3, archive.TopBy("accuracy", 99).Count);
        Assert.Empty(archive.TopBy("accuracy", 0));
    }

    [Fact]
    public void MetricNamesIsTheSortedUnionOfWhatWasActuallyReported()
    {
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 1, metrics: Metrics(("recall", 1), ("accuracy", 1))),
            Entry(2, quality: 2, metrics: Metrics(("latency", 1), ("accuracy", 1))));

        Assert.Equal(new[] { "accuracy", "latency", "recall" }, archive.MetricNames());
    }

    [Fact]
    public void TiesAreBrokenTheSameWayTheArchiveBreaksThem()
    {
        // Two candidates with the same metric value must come back in a fixed order, or two runs with the same
        // seed disagree about their own shortlist.
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 1, metrics: Metrics(("accuracy", 0.5))),
            Entry(2, quality: 2, metrics: Metrics(("accuracy", 0.5))),
            Entry(3, quality: 3, metrics: Metrics(("accuracy", 0.5))));

        IReadOnlyList<string> once = archive.TopBy("accuracy", 3).Select(entry => entry.Evaluation.GenomeId).ToArray();
        IReadOnlyList<string> again = archive.TopBy("accuracy", 3).Select(entry => entry.Evaluation.GenomeId).ToArray();

        Assert.Equal(once, again);
        Assert.Equal(once.OrderBy(id => id, StringComparer.Ordinal), once);
    }

    [Fact]
    public void AnEmptyMetricNameIsRefusedRatherThanMatchingNothingQuietly()
    {
        MapElitesArchive<TestGenome> archive = Archive(Entry(1, quality: 1, metrics: Metrics(("accuracy", 0.5))));

        Assert.ThrowsAny<ArgumentException>(() => archive.BestBy("  "));
        Assert.ThrowsAny<ArgumentException>(() => archive.TopBy(string.Empty, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => archive.TopBy("accuracy", -1));
    }

    [Fact]
    public void TheRedactedSummaryAnswersTheSameQuestionAsTheLiveArchive()
    {
        // The summary is what survives serialization, so a saved result has to be able to answer a metric question
        // without the typed run result still being in memory.
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 10, metrics: Metrics(("accuracy", 0.5), ("latency", 90))),
            Entry(2, quality: 90, metrics: Metrics(("accuracy", 0.1), ("latency", 10))));

        EvolutionRunSummary summary = Summarize(archive);

        Assert.Equal(archive.BestBy("accuracy")?.Evaluation.GenomeId, summary.BestBy("accuracy")?.GenomeId);
        Assert.Equal(
            archive.BestBy("latency", EvolutionOptimizationDirection.Minimize)?.Evaluation.GenomeId,
            summary.BestBy("latency", EvolutionOptimizationDirection.Minimize)?.GenomeId);
        Assert.Equal(archive.MetricNames(), summary.MetricNames());
        Assert.Equal(
            archive.TopBy("accuracy", 2).Select(entry => entry.Evaluation.GenomeId),
            summary.TopBy("accuracy", 2).Select(elite => elite.GenomeId));
    }

    [Fact]
    public void TheSummaryCarriesTheMetricsItRanksBy()
    {
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 10, metrics: Metrics(("accuracy", 0.5))));

        EvolutionEliteSummary elite = Assert.Single(Summarize(archive).Elites);
        Assert.Equal(0.5, elite.Metrics["accuracy"]);
    }

    [Fact]
    public void TheRunLevelQueryLooksAcrossEveryIsland()
    {
        var first = Archive(Entry(1, quality: 10, metrics: Metrics(("accuracy", 0.2))));
        var second = Archive(Entry(2, quality: 20, metrics: Metrics(("accuracy", 0.9))));
        EvolutionRunResult<TestGenome> result = Run(first, second);

        Assert.Equal(0.9, result.BestBy("accuracy")?.Evaluation.Metrics["accuracy"]);
        Assert.Equal(new[] { 0.9, 0.2 }, result.TopBy("accuracy", 5).Select(e => e.Evaluation.Metrics["accuracy"]));
        Assert.Equal(new[] { "accuracy" }, result.MetricNames());
    }

    [Fact]
    public void OneCandidateInTwoIslandsFillsOneShortlistSlotRatherThanTwo()
    {
        // Migration copies an elite between islands. Asking for the top two should name two different candidates.
        var first = Archive(Entry(1, quality: 10, metrics: Metrics(("accuracy", 0.9))));
        var second = Archive(
            Entry(1, quality: 10, metrics: Metrics(("accuracy", 0.9))),
            Entry(2, quality: 20, metrics: Metrics(("accuracy", 0.4))));

        IReadOnlyList<EvolutionArchiveEntry<TestGenome>> top = Run(first, second).TopBy("accuracy", 2);

        Assert.Equal(2, top.Count);
        Assert.Equal(2, top.Select(entry => entry.Evaluation.GenomeId).Distinct(StringComparer.Ordinal).Count());
    }

    private static EvolutionEliteSummary Elite(string genomeId, double metric)
    {
        var elite = new EvolutionEliteSummary { GenomeId = genomeId };
        elite.Metrics["score"] = metric;
        return elite;
    }

    private static Dictionary<string, double> Metrics(params (string Name, double Value)[] values)
    {
        var metrics = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach ((string name, double value) in values) metrics[name] = value;
        return metrics;
    }

    private static MapElitesArchive<TestGenome> Archive(params (EvolutionCandidate<TestGenome> Candidate,
        EvolutionEvaluation Evaluation)[] entries)
    {
        var archive = new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
        });

        foreach ((EvolutionCandidate<TestGenome> candidate, EvolutionEvaluation evaluation) in entries)
        {
            Assert.NotEqual(EvolutionArchiveInsertionResult.Rejected, archive.TryAdd(candidate, evaluation));
        }

        return archive;
    }

    private static (EvolutionCandidate<TestGenome>, EvolutionEvaluation) Entry(
        int value, double quality, IReadOnlyDictionary<string, double> metrics)
    {
        var genome = new TestGenome(value);
        string genomeId = "genome-" + value.ToString(System.Globalization.CultureInfo.InvariantCulture);
        var lineage = new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
        var candidate = new EvolutionCandidate<TestGenome>(
            value, new EvolutionCanonicalGenome<TestGenome>(genome, genomeId), lineage);
        var evaluation = new EvolutionEvaluation(
            value,
            genomeId,
            EvolutionEvaluationStatus.Completed,
            quality,
            EvolutionOptimizationDirection.Maximize,
            // Scaled so each test candidate lands in its own bin of the ten-wide grid; entries sharing a cell would
            // evict one another and the query would be reading an archive of one.
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = value * 10 },
            Array.Empty<double>(),
            Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
            lineage,
            EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(),
            "task-v1",
            "evaluator-v1",
            "config-v1",
            metrics);

        return (candidate, evaluation);
    }

    private static EvolutionRunResult<TestGenome> Run(params MapElitesArchive<TestGenome>[] islands) => new(
        EvolutionStopReason.EvaluationBudgetReached,
        islands,
        new EvolutionRunCounters(0, 0, 0, new Dictionary<EvolutionEvaluationStatus, long>()),
        "state-hash");

    private static EvolutionRunSummary Summarize(MapElitesArchive<TestGenome> archive)
    {
        // Spelled out rather than DateTimeOffset.UnixEpoch, which net471 does not have.
        var fixedTime = new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero);
        return EvolutionRunSummary.Create("run", "compat", Run(archive), fixedTime, fixedTime, maxElites: 16);
    }
}
