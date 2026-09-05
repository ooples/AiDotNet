using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers ranking a finished program run by a metric other than the fitness it optimised: the most accurate program
/// rather than the highest blended score, or the fastest of the ones that were accurate enough.
/// </summary>
public sealed class ProgramMetricQueryTests
{
    [Fact]
    public void TheEliteCarriesEveryNumberTheEvaluatorReported()
    {
        ProgramEvolutionResult result = ProgramEvolutionResult.Create(Run(
            ("print(1)", 0.1, Metrics(("accuracy", 0.9), ("seconds", 8)))));

        ProgramEvolutionElite elite = Assert.Single(result.Elites);
        Assert.Equal(0.9, elite.Metrics["accuracy"]);
        Assert.Equal(8, elite.Metrics["seconds"]);
        Assert.Equal(0.1, elite.Quality);
    }

    [Fact]
    public void BestByRanksOnTheNamedMetricRatherThanOnFitness()
    {
        ProgramEvolutionResult result = ProgramEvolutionResult.Create(Run(
            ("print(1)", 0.1, Metrics(("accuracy", 0.9), ("seconds", 8))),
            ("print(2)", 0.9, Metrics(("accuracy", 0.2), ("seconds", 1)))));

        Assert.Equal(0.9, result.BestQuality);
        Assert.Equal(0.9, result.BestBy("accuracy")?.Metrics["accuracy"]);
        Assert.Equal(1, result.BestBy("seconds", EvolutionOptimizationDirection.Minimize)?.Metrics["seconds"]);
    }

    [Fact]
    public void AProgramThatNeverReportedTheMetricIsAbsentRatherThanScoredZero()
    {
        ProgramEvolutionResult result = ProgramEvolutionResult.Create(Run(
            ("print(1)", 0.1, Metrics(("seconds", 8))),
            ("print(2)", 0.9, Metrics(("unrelated", 1)))));

        ProgramEvolutionElite? fastest = result.BestBy("seconds", EvolutionOptimizationDirection.Minimize);

        Assert.NotNull(fastest);
        Assert.Equal(8, fastest.Metrics["seconds"]);
    }

    [Fact]
    public void TopByReturnsTheBestFirstAndStopsAtTheRequestedCount()
    {
        ProgramEvolutionResult result = ProgramEvolutionResult.Create(Run(
            ("print(1)", 0.1, Metrics(("accuracy", 0.2))),
            ("print(2)", 0.5, Metrics(("accuracy", 0.9))),
            ("print(3)", 0.9, Metrics(("accuracy", 0.5)))));

        Assert.Equal(new[] { 0.9, 0.5 }, result.TopBy("accuracy", 2).Select(elite => elite.Metrics["accuracy"]));
        Assert.Empty(result.TopBy("accuracy", 0));
        Assert.Empty(result.TopBy("nonexistent", 5));
        Assert.Null(result.BestBy("nonexistent"));
    }

    [Fact]
    public void MetricNamesIsTheSortedUnionOfWhatWasActuallyReported()
    {
        ProgramEvolutionResult result = ProgramEvolutionResult.Create(Run(
            ("print(1)", 0.1, Metrics(("recall", 1), ("accuracy", 1))),
            ("print(2)", 0.5, Metrics(("seconds", 1), ("accuracy", 1)))));

        Assert.Equal(new[] { "accuracy", "recall", "seconds" }, result.MetricNames());
    }

    [Fact]
    public void OnlyTheRetainedElitesAreSearched()
    {
        // The result keeps a bounded number of elites, so a query has to answer from those rather than pretend to
        // see the whole archive.
        ProgramEvolutionResult result = ProgramEvolutionResult.Create(
            Run(("print(1)", 0.1, Metrics(("accuracy", 0.9))),
                ("print(2)", 0.9, Metrics(("accuracy", 0.2)))),
            includeEliteSourceCount: 1);

        ProgramEvolutionElite elite = Assert.Single(result.Elites);
        Assert.Equal(0.2, elite.Metrics["accuracy"]);
        Assert.Equal(0.2, result.BestBy("accuracy")?.Metrics["accuracy"]);
    }

    [Fact]
    public void AnEmptyMetricNameIsRefusedRatherThanMatchingNothingQuietly()
    {
        ProgramEvolutionResult result = ProgramEvolutionResult.Create(Run(
            ("print(1)", 0.1, Metrics(("accuracy", 0.9)))));

        Assert.ThrowsAny<ArgumentException>(() => result.BestBy(" "));
        Assert.ThrowsAny<ArgumentException>(() => result.TopBy(string.Empty, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => result.TopBy("accuracy", -1));
    }

    private static Dictionary<string, double> Metrics(params (string Name, double Value)[] values)
    {
        var metrics = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach ((string name, double value) in values) metrics[name] = value;
        return metrics;
    }

    private static EvolutionRunResult<ProgramGenome> Run(
        params (string Source, double Quality, IReadOnlyDictionary<string, double> Metrics)[] entries)
    {
        var archive = new MapElitesArchive<ProgramGenome>(
            new[] { new EvolutionDescriptorDefinition("score", 0, 1, 4, EvolutionOutOfRangePolicy.Clamp) },
            EvolutionOptimizationDirection.Maximize);

        long evaluationId = 0;
        foreach ((string source, double quality, IReadOnlyDictionary<string, double> metrics) in entries)
        {
            var genome = new ProgramGenome(source, ProgramLanguage.Python);
            var lineage = new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
            var candidate = new EvolutionCandidate<ProgramGenome>(
                evaluationId, new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id), lineage);
            var evaluation = new EvolutionEvaluation(
                evaluationId,
                genome.Id,
                EvolutionEvaluationStatus.Completed,
                quality,
                EvolutionOptimizationDirection.Maximize,
                new Dictionary<string, double>(StringComparer.Ordinal) { ["score"] = quality },
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

            Assert.NotEqual(EvolutionArchiveInsertionResult.Rejected, archive.TryAdd(candidate, evaluation));
            evaluationId++;
        }

        return new EvolutionRunResult<ProgramGenome>(
            EvolutionStopReason.EvaluationBudgetReached,
            new IEvolutionArchiveView<ProgramGenome>[] { archive },
            new EvolutionRunCounters(entries.Length, entries.Length, entries.Length,
                new Dictionary<EvolutionEvaluationStatus, long>
                {
                    [EvolutionEvaluationStatus.Completed] = entries.Length
                }),
            "state-hash");
    }
}
