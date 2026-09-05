using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.ArtifactStore;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>Proves the observer carries an evaluation's artifacts into a store that outlives the run.</summary>
public sealed class ProgramArtifactStoreObserverTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "aidotnet-artifact-observer-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public async Task AnEvaluationsArtifactsAreRetrievableByGenomeAfterTheRun()
    {
        var store = new FileSystemProgramArtifactStore(
            _root, new ProgramArtifactStoreOptions { InlineSizeThresholdBytes = 0 });
        var observer = new ProgramArtifactStoreObserver(store);

        await observer.OnEventAsync(EvaluatedEvent(
            "genome-a", new EvolutionArtifact("stderr", "Traceback: division by zero")));

        Assert.Equal(1, observer.StoredCount);
        Assert.Equal(0, observer.FailureCount);

        // The point of the store is retrieval AFTER the run, which an in-memory artifact cannot offer.
        IReadOnlyList<ProgramArtifact> retrieved = await store.GetAsync("genome-a");
        ProgramArtifact only = Assert.Single(retrieved);
        Assert.Equal("stderr", only.Name);
        Assert.True(only.IsText);
        Assert.Contains("division by zero", only.GetText());

        // Above the inline threshold the payload is promoted to a file rather than kept in the index.
        Assert.NotEmpty(Directory.GetFiles(_root, "*", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task AnEvaluationWithoutArtifactsWritesNothing()
    {
        var store = new FileSystemProgramArtifactStore(_root);
        var observer = new ProgramArtifactStoreObserver(store);

        await observer.OnEventAsync(EvaluatedEvent("genome-b"));

        Assert.Equal(0, observer.StoredCount);
        Assert.Empty(await store.GetAsync("genome-b"));
    }

    [Fact]
    public async Task AnUnwritableStoreIsCountedRatherThanEndingTheRun()
    {
        var observer = new ProgramArtifactStoreObserver(new ThrowingArtifactStore());

        // An observer runs inside the engine's commit path, so a full disk must not end a search that is
        // otherwise progressing.
        await observer.OnEventAsync(EvaluatedEvent("genome-c", new EvolutionArtifact("stderr", "boom")));

        Assert.Equal(0, observer.StoredCount);
        Assert.Equal(1, observer.FailureCount);
        Assert.NotNull(observer.LastError);
    }

    [Fact]
    public async Task NonEvaluationEventsAreIgnored()
    {
        var store = new FileSystemProgramArtifactStore(_root);
        var observer = new ProgramArtifactStoreObserver(store);

        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(
            EvolutionEventKind.Checkpointed, 1, message: "checkpoint"));

        Assert.Equal(0, observer.StoredCount);
        Assert.Equal(0, observer.FailureCount);
    }

    private static EvolutionEvent<ProgramGenome> EvaluatedEvent(
        string genomeId,
        params EvolutionArtifact[] artifacts)
    {
        var genome = new ProgramGenome("def solve(x):\n    return x\n", ProgramLanguage.Python);
        var canonical = new EvolutionCanonicalGenome<ProgramGenome>(genome, genomeId);
        var lineage = new EvolutionLineage(
            Array.Empty<string>(), Array.Empty<string>(), "variation", null, 1, 0, 1UL);
        var candidate = new EvolutionCandidate<ProgramGenome>(1, canonical, lineage);
        var evaluation = new EvolutionEvaluation(
            1,
            genomeId,
            EvolutionEvaluationStatus.Completed,
            0.5,
            EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal),
            Array.Empty<double>(),
            Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
            lineage,
            EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(),
            "task-hash",
            "evaluator-hash",
            "configuration-hash",
            artifacts: artifacts);

        return new EvolutionEvent<ProgramGenome>(
            EvolutionEventKind.Evaluated, 1, candidate, evaluation);
    }

    private sealed class ThrowingArtifactStore : AiDotNet.Interfaces.IProgramArtifactStore
    {
        public ProgramArtifactStoreOptions GetOptions() => new();

        public Task<IReadOnlyList<ProgramArtifactDescriptor>> StoreAsync(
            string genomeId, IEnumerable<ProgramArtifact> artifacts, CancellationToken cancellationToken = default) =>
            throw new IOException("the disk is full");

        public Task<IReadOnlyList<ProgramArtifact>> GetAsync(
            string genomeId, CancellationToken cancellationToken = default) =>
            Task.FromResult<IReadOnlyList<ProgramArtifact>>(Array.Empty<ProgramArtifact>());

        public Task<ProgramArtifact?> GetAsync(
            string genomeId, string name, CancellationToken cancellationToken = default) =>
            Task.FromResult<ProgramArtifact?>(null);

        public Task<IReadOnlyList<ProgramArtifactDescriptor>> ListAsync(
            string genomeId, CancellationToken cancellationToken = default) =>
            Task.FromResult<IReadOnlyList<ProgramArtifactDescriptor>>(Array.Empty<ProgramArtifactDescriptor>());

        public Task<bool> RemoveAsync(string genomeId, CancellationToken cancellationToken = default) =>
            Task.FromResult(false);

        public Task<int> PurgeAsync(DateTimeOffset utcNow, CancellationToken cancellationToken = default) =>
            Task.FromResult(0);
    }
}
