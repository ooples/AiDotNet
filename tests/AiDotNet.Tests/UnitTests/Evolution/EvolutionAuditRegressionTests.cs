using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>
/// Pins the defects an adversarial review of this branch found. Each one was reachable, silent, and passed the tests
/// that existed at the time, so each gets a test that fails if it comes back.
/// </summary>
public sealed class EvolutionAuditRegressionTests
{
    [Fact]
    public void AWatchedMetricHasItsOwnDirectionRatherThanTheArchivesu()
    {
        // Taking the sign from the archive negated a validation accuracy in a loss-minimising run, so a metric
        // climbing steadily read as one falling steadily and stopped the run exactly when it was working.
        var higherBetter = new EvolutionEarlyStoppingOptions { MetricName = "accuracy" };
        var lowerBetter = new EvolutionEarlyStoppingOptions { MetricName = "accuracy", MetricIsLowerBetter = true };

        Assert.False(higherBetter.SnapshotAndValidate().MetricIsLowerBetter);
        Assert.True(lowerBetter.SnapshotAndValidate().MetricIsLowerBetter);

        // The two watch the same metric in opposite directions, so they cannot share a resume identity.
        Assert.NotEqual(Hash(higherBetter), Hash(lowerBetter));
    }

    [Fact]
    public void TwoAggregationRulesThatScoreDifferentlyCannotShareAVersionHash()
    {
        // The options type had no string form of its own, so every configuration contributed the same text and a
        // resume accepted a scoring rule the restored elites had never been measured against.
        var mean = new ProgramMetricAggregationOptions { Strategy = ProgramMetricAggregationStrategy.Mean };
        var weighted = new ProgramMetricAggregationOptions
        {
            Strategy = ProgramMetricAggregationStrategy.Weighted,
            Weights = { ["accuracy"] = 1.0 }
        };
        var differentWeights = new ProgramMetricAggregationOptions
        {
            Strategy = ProgramMetricAggregationStrategy.Weighted,
            Weights = { ["latencyMs"] = 1.0 }
        };

        Assert.NotEqual(mean.ToString(), weighted.ToString());
        Assert.NotEqual(weighted.ToString(), differentWeights.ToString());

        // Insertion order must not matter, or two identical configurations would refuse each other's checkpoint.
        var oneOrder = new ProgramMetricAggregationOptions { Weights = { ["a"] = 1, ["b"] = 2 } };
        var otherOrder = new ProgramMetricAggregationOptions { Weights = { ["b"] = 2, ["a"] = 1 } };
        Assert.Equal(oneOrder.ToString(), otherOrder.ToString());
    }

    [Fact]
    public void TheWeightedStrategiesCanBeExpressedByAConfigurationFile()
    {
        // The three collections were get-only, so a YAML mapper silently dropped them and validation then rejected
        // the strategy for having no weights: two of the four strategies were unreachable from a file.
        const string yaml = @"
programEvolution:
  metrics:
    strategy: Weighted
    weights:
      accuracy: 2.0
      latencyMs: 0.5
    excludedFeatureDimensions:
      - length
";
        YamlModelConfig config = YamlConfigLoader.LoadFromString(yaml);
        ProgramEvolutionOptions options = Assert.IsType<ProgramEvolutionOptions>(config.ProgramEvolution);

        Assert.Equal(ProgramMetricAggregationStrategy.Weighted, options.Metrics.Strategy);
        Assert.Equal(2.0, options.Metrics.Weights["accuracy"]);
        Assert.Equal(0.5, options.Metrics.Weights["latencyMs"]);
        Assert.Contains("length", options.Metrics.ExcludedFeatureDimensions);
        options.Metrics.Validate();
    }

    [Fact]
    public void GrowthDoesNotWidenTheGridForACandidateAnotherAxisRejects()
    {
        // Widening first and discovering a second axis rejects the candidate left the grid permanently larger for a
        // candidate that never entered it, which quietly lowers coverage and moves an unbounded capacity.
        var archive = new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 1, 5, EvolutionOutOfRangePolicy.Grow),
            new EvolutionDescriptorDefinition("y", 0, 1, 5, EvolutionOutOfRangePolicy.Reject)
        });

        Assert.Equal(25, archive.TotalGridCells);
        Assert.Equal(EvolutionArchiveInsertionResult.Rejected, Add(archive, 1, "a", 1, x: 5.0, y: 99));

        Assert.Equal(25, archive.TotalGridCells);
        Assert.Equal(1.0, archive.Descriptors[0].Maximum, 12);
        Assert.Empty(archive.Entries);
    }

    [Fact]
    public void ARestoreRefusesACheckpointHoldingMoreElitesThanTheArchiveCanKeep()
    {
        // Accepting an eviction during a replay resumed a run that was quietly missing part of what it had found.
        MapElitesArchive<TestGenome> full = Archive(capacity: 0);
        Add(full, 1, "a", 1, 0.1, 0);
        Add(full, 2, "b", 2, 0.5, 0);
        Add(full, 3, "c", 3, 0.9, 0);

        MapElitesArchive<TestGenome> tooSmall = Archive(capacity: 2);
        Assert.Throws<InvalidDataException>(() => tooSmall.Restore(full.Entries.ToArray(), full.Version));
    }

    [Fact]
    public async Task ReadingACheckpointFromADifferentEngineVersionIsRefused()
    {
        // Newtonsoft is lenient, so a payload from an older engine deserialized into an all-default document and
        // read back as a complete record of a run that found nothing.
        var store = new InMemoryEvolutionCheckpointStore();
        await ReadableEngine(store).RunAsync(new[] { new TestGenome(1), new TestGenome(2) });
        EvolutionCheckpoint written = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("audit-run"));

        // The two-argument Replace is ordinal on every target framework; the comparison overload does not exist on
        // the oldest one, and the suite has to compile there.
        var older = new EvolutionCheckpoint(written.RunId, written.Sequence, written.CompatibilityHash,
            written.Payload.Replace("\"SchemaVersion\":5", "\"SchemaVersion\":4"));

        Assert.Throws<InvalidDataException>(() =>
            EvolutionEngine<TestGenome>.ReadCheckpoint(older, new TestGenomeCodec()));
    }

    private static string Hash(EvolutionEarlyStoppingOptions stopping) =>
        new EvolutionEngineOptions { EarlyStopping = stopping }.ToSemanticCanonicalString();

    private static MapElitesArchive<TestGenome> Archive(int capacity) => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 1, 5, EvolutionOutOfRangePolicy.Clamp),
        new EvolutionDescriptorDefinition("y", 0, 1, 5, EvolutionOutOfRangePolicy.Clamp)
    }, EvolutionOptimizationDirection.Maximize, capacity);

    private static EvolutionArchiveInsertionResult Add(
        MapElitesArchive<TestGenome> archive, long id, string genomeId, double quality, double x, double y)
    {
        var lineage = new EvolutionLineage(null, null, "test", null, 0, 0, (ulong)id);
        var candidate = new EvolutionCandidate<TestGenome>(id,
            new EvolutionCanonicalGenome<TestGenome>(new TestGenome((int)id), genomeId), lineage);
        var evaluation = new EvolutionEvaluation(id, genomeId, EvolutionEvaluationStatus.Completed, quality,
            EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = x, ["y"] = y },
            Array.Empty<double>(), Array.Empty<double>(), new EvolutionEvaluationCost(TimeSpan.Zero, 1, 0), lineage,
            EvolutionCacheStatus.Miss, Array.Empty<EvolutionDiagnostic>(), "task", "eval", "config");
        return archive.TryAdd(candidate, evaluation);
    }

    private static EvolutionEngine<TestGenome> ReadableEngine(IEvolutionCheckpointStore store) => new(
        new SyntheticEvolutionTask(), new IncrementVariation(), _ => new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
        }), new EvolutionEngineOptions
        {
            RunId = "audit-run",
            Seed = 5,
            MaxEvaluationAttempts = 6,
            MaxProposals = 50,
            MaxGenerations = 50,
            ProposalBatchSize = 2,
            IslandCount = 1,
            MigrationInterval = 0,
            MigrantsPerIsland = 1,
            CheckpointInterval = 0
        }, checkpointStore: store, genomeCodec: new TestGenomeCodec());
}
