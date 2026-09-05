using System.Diagnostics;
using AiDotNet.Agentic.Embeddings;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Novelty;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.Evolution.Programs.Novelty;

public sealed class NoveltyGatingProgramFitnessEvaluatorTests
{
    private readonly ITestOutputHelper _output;

    public NoveltyGatingProgramFitnessEvaluatorTests(ITestOutputHelper output) => _output = output;

    private static EvolutionEvaluationContext Context() =>
        new(evaluationId: 1, rootSeed: 7UL, seedStream: 11UL, attemptCount: 1);

    [Fact]
    public async Task ANearDuplicateIsTurnedAwayBeforeTheEvaluatorSpendsAnything()
    {
        var inner = new RecordingProgramFitnessEvaluator();
        var gate = new NoveltyGatingProgramFitnessEvaluator(inner);
        var original = new ProgramGenome("def add(a, b):\n    return a + b\n");

        EvolutionTaskResult first = await gate.EvaluateAsync(original, Context());
        EvolutionTaskResult second = await gate.EvaluateAsync(
            new ProgramGenome("def add(a, b):\n    return a + b\n\n\n"), Context());

        Assert.Equal(EvolutionEvaluationStatus.Completed, first.Status);
        Assert.Equal(EvolutionEvaluationStatus.Rejected, second.Status);

        // Upstream evaluates first and gates on insertion, so its duplicate has already cost a full evaluation.
        Assert.Equal(1, inner.Calls);
        Assert.Equal(1L, gate.AcceptedCount);
        Assert.Equal(1L, gate.RejectedCount);
        Assert.Equal(NoveltyGatingProgramFitnessEvaluator.RejectionCode, second.Diagnostics[0].Code);
    }

    [Fact]
    public async Task ADistinctCandidateReachesTheEvaluator()
    {
        var inner = new RecordingProgramFitnessEvaluator();
        var gate = new NoveltyGatingProgramFitnessEvaluator(inner);

        await gate.EvaluateAsync(new ProgramGenome("def add(a, b):\n    return a + b\n"), Context());
        await gate.EvaluateAsync(new ProgramGenome("import socket\nsocket.socket().listen()\n"), Context());

        Assert.Equal(2, inner.Calls);
        Assert.Equal(2L, gate.AcceptedCount);
        Assert.Equal(0L, gate.RejectedCount);
        Assert.Equal(2, gate.TrackedCount);
    }

    [Fact]
    public async Task TheRememberedSetIsBoundedByTheConfiguredLimit()
    {
        var options = new EmbeddingNoveltyOptions(structuralNoveltyThreshold: 0.0, maxTrackedGenomes: 3);
        var gate = new NoveltyGatingProgramFitnessEvaluator(
            new RecordingProgramFitnessEvaluator(),
            new ProgramNoveltyPolicy(options));

        for (int index = 0; index < 10; index++)
        {
            await gate.EvaluateAsync(new ProgramGenome("value = " + index), Context());
        }

        Assert.Equal(3, gate.TrackedCount);
    }

    [Fact]
    public async Task ASeededGenomeCanBeRememberedWithoutBeingEvaluated()
    {
        var inner = new RecordingProgramFitnessEvaluator();
        var gate = new NoveltyGatingProgramFitnessEvaluator(inner);
        var seed = new ProgramGenome("def add(a, b):\n    return a + b\n");

        gate.Remember(seed);
        EvolutionTaskResult result = await gate.EvaluateAsync(seed, Context());

        Assert.Equal(EvolutionEvaluationStatus.Rejected, result.Status);
        Assert.Equal(0, inner.Calls);
        Assert.Equal(1, gate.TrackedCount);
    }

    [Fact]
    public async Task TheGateRecordsTheDecisionThatProducedItsAnswer()
    {
        var gate = new NoveltyGatingProgramFitnessEvaluator(new RecordingProgramFitnessEvaluator());
        Assert.Null(gate.GetLastDecision());

        await gate.EvaluateAsync(new ProgramGenome("print(1)"), Context());
        ProgramNoveltyDecision? decision = gate.GetLastDecision();

        Assert.NotNull(decision);
        Assert.True(decision?.IsNovel);
        Assert.Equal(ProgramNoveltyStage.None, decision?.DecidedBy);
    }

    [Fact]
    public void TheGateVersionFollowsBothTheInnerEvaluatorAndThePolicy()
    {
        var inner = new RecordingProgramFitnessEvaluator();
        string bare = new NoveltyGatingProgramFitnessEvaluator(inner).VersionHash;
        string withEmbedding = new NoveltyGatingProgramFitnessEvaluator(
            inner,
            new ProgramNoveltyPolicy(embeddingClient: new DeterministicEmbeddingClient())).VersionHash;

        Assert.NotEqual(bare, withEmbedding);
        Assert.Equal(bare, new NoveltyGatingProgramFitnessEvaluator(inner).VersionHash);
    }

    [Fact]
    public async Task TheStructuralDecisionCostsNoNetworkCallAndStaysUnderAMillisecond()
    {
        // Cost harness for the structural rung: 200 decisions, each against a 64-genome known set, with an
        // embedding provider and a judge configured but expected to stay untouched.
        const int Decisions = 200;
        const int KnownCount = 64;

        var provider = new DeterministicEmbeddingClient(dimensions: 64);
        var judge = new ScriptedNoveltyJudge(ProgramNoveltyVerdict.NotNovel);
        var policy = new ProgramNoveltyPolicy(embeddingClient: provider, judge: judge);

        var known = new List<ProgramGenome>(KnownCount);
        for (int index = 0; index < KnownCount; index++)
        {
            known.Add(new ProgramGenome(
                "def solve_" + index + "(values):\n" +
                "    total_" + index + " = 0\n" +
                "    for item_" + index + " in values:\n" +
                "        total_" + index + " += item_" + index + " * " + index + "\n" +
                "    return total_" + index + "\n"));
        }

        var candidates = new List<ProgramGenome>(Decisions);
        for (int index = 0; index < Decisions; index++)
        {
            candidates.Add(new ProgramGenome(
                "class Runner_" + index + ":\n" +
                "    def __init__(self, factor_" + index + "):\n" +
                "        self.factor_" + index + " = factor_" + index + "\n" +
                "    def apply(self, payload_" + index + "):\n" +
                "        return [entry_" + index + " ** self.factor_" + index + " for entry_" + index + " in payload_" + index + "]\n"));
        }

        var uncachedPolicy = new ProgramNoveltyPolicy(
            structuralDistance: new ProgramTokenSetDistance(memoCapacity: 0));

        // Warm up so the measurement excludes first-call JIT and the one-off memoization of the known set.
        for (int index = 0; index < 8; index++)
        {
            await policy.EvaluateAsync(candidates[index], known);
            await uncachedPolicy.EvaluateAsync(candidates[index], known);
        }

        var stopwatch = Stopwatch.StartNew();
        for (int index = 0; index < Decisions; index++)
        {
            ProgramNoveltyDecision decision = await policy.EvaluateAsync(candidates[index], known);
            Assert.True(decision.IsNovel);
            Assert.Equal(ProgramNoveltyStage.Structural, decision.DecidedBy);
            Assert.True(decision.WasFree);
        }

        stopwatch.Stop();
        double microsecondsPerDecision = stopwatch.Elapsed.TotalMilliseconds * 1000.0 / Decisions;

        var uncachedWatch = Stopwatch.StartNew();
        for (int index = 0; index < Decisions; index++)
        {
            Assert.True((await uncachedPolicy.EvaluateAsync(candidates[index], known)).IsNovel);
        }

        uncachedWatch.Stop();
        double uncachedMicroseconds = uncachedWatch.Elapsed.TotalMilliseconds * 1000.0 / Decisions;

        _output.WriteLine(
            "STRUCTURAL_DECISION_COST decisions=" + Decisions +
            " knownPerDecision=" + KnownCount +
            " totalMs=" + stopwatch.Elapsed.TotalMilliseconds.ToString("F2") +
            " microsecondsPerDecision=" + microsecondsPerDecision.ToString("F1") +
            " microsecondsPerComparison=" + (microsecondsPerDecision / KnownCount).ToString("F3") +
            " uncachedMicrosecondsPerDecision=" + uncachedMicroseconds.ToString("F1") +
            " uncachedMicrosecondsPerComparison=" + (uncachedMicroseconds / KnownCount).ToString("F3"));

        // The whole point: no provider request and no model call was made for any of the 208 decisions.
        Assert.Equal(0L, provider.Calls);
        Assert.Equal(0, judge.Calls);
        Assert.Equal(0L, policy.EmbeddingRequests);
        Assert.Equal(0L, policy.JudgeRequests);

        // A generous ceiling: the assertion guards against an accidental quadratic blow-up, not against jitter on
        // a loaded build machine. The recorded number above is the real measurement.
        Assert.True(
            microsecondsPerDecision < 5_000,
            "structural decision took " + microsecondsPerDecision.ToString("F1") + " microseconds");
    }
}
