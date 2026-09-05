using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>
/// Covers <see cref="EvolutionDispatchMode.Continuous"/>: a window of evaluations that refills as each one commits,
/// instead of a batch barrier that drains before the next batch starts. The point of the mode is worker utilisation,
/// so the tests measure how many evaluations actually run at once, and they pin the determinism the mode must not
/// cost: the same state hash at every worker count, and a resume that reproduces an uninterrupted run.
/// </summary>
public sealed class EvolutionContinuousDispatchTests
{
    [Fact]
    public async Task ContinuousDispatchRunsAWholeWindowWhileBatchDispatchRunsOneAtATime()
    {
        // One proposal per batch is the clearest contrast: batch dispatch can never have two evaluations running,
        // while continuous dispatch keeps the window full regardless of how slow any single candidate is.
        var batched = new ConcurrencyProbeTask(gateAt: 0);
        EvolutionRunResult<TestGenome> batchRun = await Engine(batched, Options(EvolutionDispatchMode.Batch))
            .RunAsync(Seeds(4));

        var continuous = new ConcurrencyProbeTask(gateAt: 4);
        EvolutionRunResult<TestGenome> continuousRun = await Engine(continuous, Options(EvolutionDispatchMode.Continuous))
            .RunAsync(Seeds(4));

        Assert.Equal(1, batched.MaxConcurrency);
        Assert.Equal(4, continuous.MaxConcurrency);
        Assert.Equal(batchRun.Counters.CompletedEvaluations, continuousRun.Counters.CompletedEvaluations);
    }

    [Fact]
    public async Task ContinuousDispatchProducesOneStateHashAtEveryWorkerCount()
    {
        string? expected = null;
        foreach (int workers in new[] { 1, 2, 4, 8 })
        {
            EvolutionEngineOptions options = Options(EvolutionDispatchMode.Continuous);
            options.MaxDegreeOfParallelism = workers;
            options.MaxInFlight = 4;
            EvolutionRunResult<TestGenome> run = await Engine(new ConcurrencyProbeTask(gateAt: 0), options)
                .RunAsync(Seeds(4));

            expected ??= run.StateHash;
            Assert.Equal(expected, run.StateHash);
        }
    }

    [Fact]
    public async Task AnIslandInFlightQuotaThrottlesWhatTheWindowWouldOtherwiseRun()
    {
        EvolutionEngineOptions throttled = Options(EvolutionDispatchMode.Continuous);
        throttled.MaxInFlightPerIsland = 1;
        var probe = new ConcurrencyProbeTask(gateAt: 0);
        EvolutionRunResult<TestGenome> throttledRun = await Engine(probe, throttled).RunAsync(Seeds(4));

        // The window is four and the workers are four, so only the quota can hold concurrency down to one.
        Assert.Equal(1, probe.MaxConcurrency);

        var unthrottledProbe = new ConcurrencyProbeTask(gateAt: 4);
        EvolutionRunResult<TestGenome> unthrottledRun = await Engine(unthrottledProbe,
            Options(EvolutionDispatchMode.Continuous)).RunAsync(Seeds(4));
        Assert.Equal(4, unthrottledProbe.MaxConcurrency);

        // Throttling changes only the schedule, so both runs complete the same work.
        Assert.Equal(unthrottledRun.Counters.CompletedEvaluations, throttledRun.Counters.CompletedEvaluations);
    }

    [Fact]
    public async Task AContinuousRunResumesFromItsCheckpointWithTheUninterruptedStateHash()
    {
        EvolutionRunResult<TestGenome> uninterrupted = await Engine(new ConcurrencyProbeTask(gateAt: 0),
            Options(EvolutionDispatchMode.Continuous, 12), new InMemoryEvolutionCheckpointStore()).RunAsync(Seeds(4));

        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionRunResult<TestGenome> partial = await Engine(new ConcurrencyProbeTask(gateAt: 0),
            Options(EvolutionDispatchMode.Continuous, 4), store).RunAsync(Seeds(4));

        EvolutionEngineOptions resumedOptions = Options(EvolutionDispatchMode.Continuous, 12);
        resumedOptions.Resume = true;
        EvolutionRunResult<TestGenome> resumed = await Engine(new ConcurrencyProbeTask(gateAt: 0), resumedOptions, store)
            .RunAsync(Seeds(4));

        Assert.True(partial.Counters.CompletedEvaluations < uninterrupted.Counters.CompletedEvaluations);
        Assert.Equal(uninterrupted.Counters.Proposals, resumed.Counters.Proposals);
        Assert.Equal(uninterrupted.StateHash, resumed.StateHash);
    }

    [Fact]
    public void TheDispatchSettingsAreRecordedAsConfigurationRatherThanAsBudget()
    {
        string batch = Engine(new ConcurrencyProbeTask(gateAt: 0), Options(EvolutionDispatchMode.Batch))
            .CompatibilityHash;
        string continuous = Engine(new ConcurrencyProbeTask(gateAt: 0), Options(EvolutionDispatchMode.Continuous))
            .CompatibilityHash;
        Assert.NotEqual(batch, continuous);

        EvolutionEngineOptions widerWindow = Options(EvolutionDispatchMode.Continuous);
        widerWindow.MaxInFlight = 16;
        Assert.NotEqual(continuous, Engine(new ConcurrencyProbeTask(gateAt: 0), widerWindow).CompatibilityHash);

        EvolutionEngineOptions perIsland = Options(EvolutionDispatchMode.Continuous);
        perIsland.MaxInFlightPerIsland = 2;
        Assert.NotEqual(continuous, Engine(new ConcurrencyProbeTask(gateAt: 0), perIsland).CompatibilityHash);

        // Worker count remains a budget setting, so it cannot refuse a resume.
        EvolutionEngineOptions moreWorkers = Options(EvolutionDispatchMode.Continuous);
        moreWorkers.MaxDegreeOfParallelism = 8;
        Assert.Equal(continuous, Engine(new ConcurrencyProbeTask(gateAt: 0), moreWorkers).CompatibilityHash);
    }

    [Fact]
    public async Task TheFinalCheckpointHoldsTheRunsRealStateEvenWithPeriodicCheckpointingOff()
    {
        // The earlier version of this mode only refreshed the checkpoint payload on a periodic drain, so with
        // periodic checkpointing off the final checkpoint held the state the run started from: zero proposals and
        // empty archives, published under a header that looked healthy.
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionRunResult<TestGenome> run = await Engine(new ConcurrencyProbeTask(gateAt: 0),
            Options(EvolutionDispatchMode.Continuous), store).RunAsync(Seeds(4));

        EvolutionCheckpoint checkpoint = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("dispatch-run"));
        EvolutionCheckpointContents<TestGenome> contents =
            EvolutionEngine<TestGenome>.ReadCheckpoint(checkpoint, new TestGenomeCodec());

        Assert.NotEmpty(contents.Entries);
        Assert.Equal(
            run.Islands.SelectMany(island => island.Entries).Select(entry => entry.Evaluation.GenomeId)
                .OrderBy(id => id, StringComparer.Ordinal),
            contents.Entries.Where(entry => entry.Source == EvolutionCheckpointEntrySource.IslandArchive)
                .Select(entry => entry.GenomeId).OrderBy(id => id, StringComparer.Ordinal));
    }

    [Fact]
    public async Task AStopRequestEndsAContinuousRunInsteadOfBeingIgnored()
    {
        // The stop flag was read only where the window was already empty, which in steady state it never is, so a
        // request to stop was quietly dropped and the run refilled forever.
        EvolutionEngineOptions options = Options(EvolutionDispatchMode.Continuous, 1000);
        options.MaxProposals = 2000;
        options.MaxGenerations = 2000;

        // The stop is requested from inside an evaluation, so it lands while the window is full rather than racing
        // a run that may already have finished.
        var probe = new ConcurrencyProbeTask(gateAt: 0);
        var engine = Engine(probe, options);
        probe.OnEvaluation = count =>
        {
            if (count == 8) engine.RequestStop();
        };

        using var guard = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        EvolutionRunResult<TestGenome> result = await engine.RunAsync(Seeds(4), guard.Token);

        Assert.Equal(EvolutionStopReason.Canceled, result.StopReason);
        Assert.True(result.Counters.EvaluationAttempts < 1000,
            "the run should have stopped on request, not on its budget of 1000");
    }

    [Fact]
    public void AWindowThatFollowsTheWorkerCountMakesTheWorkerCountPartOfTheSearch()
    {
        // The window decides which committed evaluation a proposal is prepared after. Leaving it at zero defers it
        // to the worker count, which a resume may otherwise change freely, so the resolved window is what the
        // compatibility hash has to record.
        EvolutionEngineOptions narrow = Options(EvolutionDispatchMode.Continuous);
        narrow.MaxInFlight = 0;
        narrow.MaxDegreeOfParallelism = 2;

        EvolutionEngineOptions wide = Options(EvolutionDispatchMode.Continuous);
        wide.MaxInFlight = 0;
        wide.MaxDegreeOfParallelism = 8;

        Assert.NotEqual(
            Engine(new ConcurrencyProbeTask(gateAt: 0), narrow).CompatibilityHash,
            Engine(new ConcurrencyProbeTask(gateAt: 0), wide).CompatibilityHash);

        // An explicit window pins it, so the worker count goes back to being a budget setting a resume may change.
        EvolutionEngineOptions pinnedNarrow = Options(EvolutionDispatchMode.Continuous);
        pinnedNarrow.MaxInFlight = 4;
        pinnedNarrow.MaxDegreeOfParallelism = 2;
        EvolutionEngineOptions pinnedWide = Options(EvolutionDispatchMode.Continuous);
        pinnedWide.MaxInFlight = 4;
        pinnedWide.MaxDegreeOfParallelism = 8;
        Assert.Equal(
            Engine(new ConcurrencyProbeTask(gateAt: 0), pinnedNarrow).CompatibilityHash,
            Engine(new ConcurrencyProbeTask(gateAt: 0), pinnedWide).CompatibilityHash);

        // Batch dispatch reads neither setting, so neither may refuse a resume there.
        EvolutionEngineOptions batchNarrow = Options(EvolutionDispatchMode.Batch);
        batchNarrow.MaxInFlight = 0;
        batchNarrow.MaxDegreeOfParallelism = 2;
        EvolutionEngineOptions batchWide = Options(EvolutionDispatchMode.Batch);
        batchWide.MaxInFlight = 16;
        batchWide.MaxDegreeOfParallelism = 8;
        Assert.Equal(
            Engine(new ConcurrencyProbeTask(gateAt: 0), batchNarrow).CompatibilityHash,
            Engine(new ConcurrencyProbeTask(gateAt: 0), batchWide).CompatibilityHash);
    }

    private static TestGenome[] Seeds(int count) =>
        Enumerable.Range(1, count).Select(value => new TestGenome(value)).ToArray();

    private static EvolutionEngineOptions Options(EvolutionDispatchMode dispatch, int maxAttempts = 12) => new()
    {
        RunId = "dispatch-run",
        Seed = 31,
        Dispatch = dispatch,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = 1,
        MaxInFlight = 4,
        MaxDegreeOfParallelism = 4,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1,
        CheckpointInterval = 0
    };

    private static EvolutionEngine<TestGenome> Engine(IEvolutionTask<TestGenome> task, EvolutionEngineOptions options,
        IEvolutionCheckpointStore? checkpointStore = null) => new(
        task, new DistinctVariation(), _ => new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 1000, 50, EvolutionOutOfRangePolicy.Clamp)
        }), options, checkpointStore: checkpointStore,
        genomeCodec: checkpointStore is null ? null : new TestGenomeCodec());

    /// <summary>Proposes a genome nothing else can collide with, so every proposal reaches the evaluator.</summary>
    private sealed class DistinctVariation : IVariationOperator<TestGenome>
    {
        public string Id => "distinct";
        public string VersionHash => "distinct-v1";

        public ValueTask<TestGenome> ProposeAsync(EvolutionVariationContext<TestGenome> context,
            CancellationToken cancellationToken = default) =>
            new(new TestGenome(1000 + (int)context.Generation));
    }

    /// <summary>Records how many evaluations run at once, optionally holding them until a target is reached.</summary>
    /// <remarks>
    /// The gate turns a utilisation claim into a deterministic assertion. Without it a test could only time a run and
    /// hope the machine cooperated; with it, an evaluation waits until the engine has genuinely dispatched the target
    /// number at once, so reaching the target proves the dispatcher filled the window and failing to reach it proves
    /// it did not. The wait is bounded so a mode that cannot fill the window fails an assertion instead of hanging.
    /// </remarks>
    private sealed class ConcurrencyProbeTask : IEvolutionTask<TestGenome>
    {
        private readonly int _gateAt;
        private readonly TaskCompletionSource<bool> _gate =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _concurrency;
        private int _maxConcurrency;
        private long _evaluations;

        public ConcurrencyProbeTask(int gateAt) => _gateAt = gateAt;

        public string Id => "concurrency-probe";
        public string VersionHash => "concurrency-probe-v1";
        public string EvaluatorVersionHash => "concurrency-probe-evaluator-v1";
        public int MaxConcurrency => Volatile.Read(ref _maxConcurrency);

        /// <summary>Called with the running evaluation count, so a test can act from inside the run.</summary>
        public Action<int>? OnEvaluation { get; set; }

        public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
            CancellationToken cancellationToken = default) =>
            new(new EvolutionCanonicalGenome<TestGenome>(new TestGenome(genome.Value),
                genome.Value.ToString(CultureInfo.InvariantCulture)));

        public async ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
            EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
        {
            int current = Interlocked.Increment(ref _concurrency);
            RecordMaximum(current);
            OnEvaluation?.Invoke((int)Interlocked.Increment(ref _evaluations));
            try
            {
                if (_gateAt > 0)
                {
                    if (current >= _gateAt) _gate.TrySetResult(true);
                    await Task.WhenAny(_gate.Task, Task.Delay(TimeSpan.FromSeconds(20), cancellationToken))
                        .ConfigureAwait(false);
                }
                int value = candidate.CanonicalGenome.Genome.Value;
                return EvolutionTaskResult.Completed(value,
                    new Dictionary<string, double> { ["x"] = Math.Max(0, Math.Min(1000, value)) });
            }
            finally
            {
                Interlocked.Decrement(ref _concurrency);
            }
        }

        private void RecordMaximum(int value)
        {
            while (true)
            {
                int current = Volatile.Read(ref _maxConcurrency);
                if (value <= current || Interlocked.CompareExchange(ref _maxConcurrency, value, current) == current) return;
            }
        }
    }
}
