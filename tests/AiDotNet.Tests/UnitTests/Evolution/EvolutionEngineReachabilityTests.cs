using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionEngineReachabilityTests
{
    [Fact]
    public async Task RatioSelectionConfiguredThroughOptionsExercisesTheExploitationBranch()
    {
        var ratioVariation = new RecordingParentVariation();
        EvolutionRunResult<TestGenome> ratioResult =
            await Engine(RatioOptions(), ratioVariation).RunAsync(Seeds());
        IReadOnlyList<int> ratioParents = ratioVariation.Parents;

        EvolutionEngineOptions uniformOptions = RatioOptions();
        uniformOptions.SelectionPolicy = EvolutionSelectionPolicyKind.Uniform;
        var uniformVariation = new RecordingParentVariation();
        await Engine(uniformOptions, uniformVariation).RunAsync(Seeds());
        IReadOnlyList<int> uniformParents = uniformVariation.Parents;

        Assert.NotEmpty(ratioParents);
        Assert.Equal(30, ratioParents[0]);
        Assert.Equal(ratioParents.OrderBy(value => value).ToArray(), ratioParents.ToArray());
        Assert.Equal(ratioParents.Count, ratioParents.Distinct().Count());
        Assert.NotEmpty(ratioResult.GlobalElites);
        Assert.NotEqual(ratioParents.ToArray(), uniformParents.ToArray());
        Assert.Contains(uniformParents, value => value < 30);
    }

    [Fact]
    public async Task AnExplicitlySuppliedSelectionPolicyWinsOverTheOptions()
    {
        EvolutionEngineOptions uniformOptions = RatioOptions();
        uniformOptions.SelectionPolicy = EvolutionSelectionPolicyKind.Uniform;
        var uniformVariation = new RecordingParentVariation();
        await Engine(uniformOptions, uniformVariation).RunAsync(Seeds());

        var overriddenVariation = new RecordingParentVariation();
        await Engine(RatioOptions(), overriddenVariation,
            new UniformEvolutionSelectionPolicy<TestGenome>()).RunAsync(Seeds());

        Assert.Equal(uniformVariation.Parents.ToArray(), overriddenVariation.Parents.ToArray());
        Assert.Equal(
            Engine(uniformOptions, new RecordingParentVariation()).CompatibilityHash,
            Engine(RatioOptions(), new RecordingParentVariation(),
                new UniformEvolutionSelectionPolicy<TestGenome>()).CompatibilityHash);
        Assert.NotEqual(
            Engine(uniformOptions, new RecordingParentVariation()).CompatibilityHash,
            Engine(RatioOptions(), new RecordingParentVariation()).CompatibilityHash);
    }

    [Fact]
    public void OpenEvolveDefaultsMatchTheDocumentedUpstreamNumbers()
    {
        EvolutionEngineOptions options = EvolutionEngineOptions.CreateOpenEvolveDefaults();

        Assert.Equal(42UL, options.Seed);
        Assert.Equal(10_000, options.MaxEvaluationAttempts);
        Assert.Equal(10_000, options.MaxProposals);
        Assert.Equal(10_000, options.MaxGenerations);
        Assert.Equal(100, options.CheckpointInterval);
        Assert.Equal(3, options.MaxRetries);
        Assert.Equal(TimeSpan.FromSeconds(300), options.EvaluationTimeout);
        Assert.Equal(TimeSpan.FromSeconds(1), options.RetryBaseDelay);
        Assert.Equal(1.0, options.RetryBackoffMultiplier);
        Assert.Equal(EvolutionRetryStatuses.Failed, options.RetryOn);
        Assert.Equal(5, options.IslandCount);
        Assert.Equal(50, options.MigrationInterval);
        Assert.Equal(EvolutionMigrationTrigger.IslandGenerations, options.MigrationTrigger);
        Assert.Equal(100, options.GlobalEliteCount);
        Assert.Equal(1_000, options.HistorySize);
        Assert.True(options.Artifacts.Enabled);
        Assert.Equal(1, options.ProposalBatchSize);
        Assert.Equal(1, options.MaxDegreeOfParallelism);
        Assert.Equal(5, options.InspirationCount);
        Assert.Equal(EvolutionSelectionPolicyKind.Ratio, options.SelectionPolicy);
        Assert.Equal(0.2, options.Selection.ExplorationRatio);
        Assert.Equal(0.7, options.Selection.ExploitationRatio);
        Assert.Equal(0.1, options.Selection.EliteRatio);
        Assert.Equal(3, options.Selection.TopInspirationCount);
        Assert.Equal(2, options.Selection.DiverseInspirationCount);
        Assert.Equal(1e-3, options.EarlyStopping.MinimumImprovement);
    }

    [Fact]
    public void OpenEvolveDefaultsLeaveTheClassDefaultsAloneAndStayValid()
    {
        var untouched = new EvolutionEngineOptions();

        Assert.Equal(1, untouched.IslandCount);
        Assert.Equal(0, untouched.MaxRetries);
        Assert.Null(untouched.EvaluationTimeout);
        Assert.Equal(0, untouched.GlobalEliteCount);
        Assert.Equal(0, untouched.HistorySize);
        Assert.False(untouched.Artifacts.Enabled);
        Assert.Equal(EvolutionSelectionPolicyKind.Uniform, untouched.SelectionPolicy);
        Assert.False(EvolutionEngineOptions.CreateOpenEvolveDefaults().Cascade.Enabled);

        EvolutionEngineOptions upstream = EvolutionEngineOptions.CreateOpenEvolveDefaults();
        upstream.MaxEvaluationAttempts = 4;
        upstream.MaxProposals = 4;
        upstream.MaxGenerations = 4;
        upstream.EvaluationTimeout = TimeSpan.FromSeconds(5);
        Exception? exception = Record.Exception(() => new EvolutionEngine<TestGenome>(
            new SyntheticEvolutionTask(), new RecordingParentVariation(), _ => Archive(), upstream));

        Assert.Null(exception);
    }

    [Fact]
    public void MapElitesAutoMLSurfacesIslandsAndMigrationWithoutChangingItsDefaults()
    {
        var defaults = new MapElitesAutoMLOptions();

        Assert.Equal(1, defaults.IslandCount);
        Assert.Equal(0, defaults.MigrationInterval);
        Assert.Equal(2, defaults.MigrantsPerIsland);
        Assert.Equal(1, defaults.SnapshotAndValidate().IslandCount);
        Assert.Equal(0, defaults.SnapshotAndValidate().MigrationInterval);
        Assert.Equal(2, defaults.SnapshotAndValidate().MigrantsPerIsland);

        Assert.Equal(4, new MapElitesAutoMLOptions { IslandCount = 4 }.SnapshotAndValidate().IslandCount);
        Assert.Equal(6, new MapElitesAutoMLOptions { MigrationInterval = 6 }.SnapshotAndValidate().MigrationInterval);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MapElitesAutoMLOptions { IslandCount = 0 }.SnapshotAndValidate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MapElitesAutoMLOptions { MigrationInterval = -1 }.SnapshotAndValidate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MapElitesAutoMLOptions { MigrantsPerIsland = 0 }.SnapshotAndValidate());
    }

    private static EvolutionEngine<TestGenome> Engine(EvolutionEngineOptions options,
        RecordingParentVariation variation, ISelectionPolicy<TestGenome>? selection = null) =>
        new(new SyntheticEvolutionTask(), variation, _ => Archive(), options, selection: selection);

    private static MapElitesArchive<TestGenome> Archive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    });

    private static TestGenome[] Seeds() => new[] { new TestGenome(10), new TestGenome(20), new TestGenome(30) };

    private static EvolutionEngineOptions RatioOptions() => new()
    {
        RunId = "selection-wiring",
        Seed = 77,
        MaxEvaluationAttempts = 12,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = 1,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        CheckpointInterval = 0,
        InspirationCount = 0,
        SelectionPolicy = EvolutionSelectionPolicyKind.Ratio,
        Selection = new EvolutionSelectionOptions
        {
            ExplorationRatio = 0,
            ExploitationRatio = 1,
            EliteRatio = 0,
            ExploitationEliteCount = 1,
            ExploitationSource = EvolutionExploitationSource.GlobalTopK,
            TopInspirationCount = 0,
            DiverseInspirationCount = 0,
            IncludeIslandBest = false
        }
    };
}

internal sealed class RecordingParentVariation : IVariationOperator<TestGenome>
{
    private readonly List<int> _parents = new();

    public string Id => "recording-parent";

    public string VersionHash => "recording-parent-v1";

    public IReadOnlyList<int> Parents
    {
        get { lock (_parents) return _parents.ToArray(); }
    }

    public ValueTask<TestGenome> ProposeAsync(EvolutionVariationContext<TestGenome> context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int parent = context.Parent.Candidate.CanonicalGenome.Genome.Value;
        lock (_parents) _parents.Add(parent);
        return new ValueTask<TestGenome>(new TestGenome(parent + 1));
    }
}
