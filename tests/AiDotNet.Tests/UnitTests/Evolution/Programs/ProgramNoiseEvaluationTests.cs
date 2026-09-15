using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramNoiseEvaluationTests
{
    private static EvolutionResourceLedger Ledger(decimal maximum = 10000) => new("consumer-noise", EvolutionResources.Of("cost_units", maximum));
    private static EvolutionTaskResult Result(double value) => EvolutionTaskResult.Completed(value, new Dictionary<string, double>(), costUnits: 1);
    private static IProgramFitnessEvaluator Eval(Func<ProgramGenome, double> score, string version = "test-v1") =>
        new DelegateProgramFitnessEvaluator((g, _, _) => new ValueTask<EvolutionTaskResult>(Result(score(g))), versionHash: version);
    private static ProgramNoiseEvaluationSession Session(EvolutionResourceLedger ledger, IProgramFitnessEvaluator? correctness = null,
        IProgramFitnessEvaluator? screen = null, IProgramFitnessEvaluator? full = null, IProgramFitnessEvaluator? hidden = null) =>
        new(new ProgramNoiseEvaluationOptions(confirmationSamples: 64, auditCandidates: 2, maximumChallenges: 2), ledger,
            correctness ?? Eval(_ => 1), screen ?? Eval(_ => 0), full ?? Eval(g => g.Source == "good" ? 1 : 0),
            Eval(_ => 1), hidden ?? Eval(g => g.Source == "good" ? 1 : 0));

    [Fact]
    public async Task ScreensAutomaticallyAuditUsefulRejectsAndChargeBothStages()
    {
        var ledger = Ledger();
        var report = await Session(ledger).ScreenAndAuditAsync("batch", new[] { new ProgramGenome("good"), new ProgramGenome("bad") }, 7, 11);
        Assert.True(report.IsComplete);
        Assert.All(report.Entries, row => Assert.False(row.Passed));
        Assert.Equal(.5, report.Audit!.FalseRejectionRateLower);
        Assert.Equal(.5, report.Audit.FalseRejectionRateUpper);
        Assert.Equal(264, report.ChargedCostUnits); // (2 programs * 2 screens + 2 * 64 full) * (check + fitness)
        Assert.Equal(report.ChargedCostUnits, ledger.Snapshot().Spent["cost_units"]);
    }

    [Fact]
    public async Task NoRejectsDoNotInvokeHiddenEvaluator()
    {
        var report = await Session(Ledger(), screen: Eval(_ => 1), hidden: Eval(_ => throw new Exception("must not call")))
            .ScreenAndAuditAsync("batch", new[] { new ProgramGenome("good") }, 7, 11);
        Assert.True(report.IsComplete); Assert.Null(report.Audit); Assert.Equal(4, report.ChargedCostUnits);
    }

    [Fact]
    public async Task InvalidCorrectnessPreventsExpensiveFitnessAndCannotPassScreen()
    {
        int calls = 0;
        var report = await Session(Ledger(), correctness: Eval(_ => 0), screen: Eval(_ => { calls++; return 1; }))
            .ScreenAndAuditAsync("batch", new[] { new ProgramGenome("bad") }, 7, 11);
        Assert.False(report.IsComplete); Assert.Equal(0, calls);
        Assert.False(report.Entries[0].Passed); Assert.Equal(1, report.ChargedCostUnits);
    }

    [Fact]
    public async Task OptimisticSearchNeverBypassesHiddenConfirmation()
    {
        var report = await Session(Ledger(), hidden: Eval(g => g.Source == "good" ? 0 : 1))
            .ChallengeAsync(0, new("good"), new("bad"), 7);
        Assert.False(report.IsConfirmed); Assert.Equal("not-confirmed", report.Outcome);
        Assert.Equal(272, report.ChargedCostUnits);
    }

    [Fact]
    public async Task CorrectCandidateIsConfirmedWithFreshChecksAndMeasurements()
    {
        var report = await Session(Ledger()).ChallengeAsync(0, new("good"), new("bad"), 7);
        Assert.True(report.IsConfirmed); Assert.Equal(272, report.ChargedCostUnits);
    }

    [Fact]
    public async Task BatchTombstoneSurvivesRestoreAndRejectsChangedPopulation()
    {
        var ledger = Ledger();
        await Session(ledger).ScreenAndAuditAsync("batch", new[] { new ProgramGenome("good") }, 7, 11);
        var restored = Ledger(); restored.RestoreState(ledger.CaptureState());
        var before = restored.CaptureState();
        await Assert.ThrowsAsync<InvalidOperationException>(() => Session(restored)
            .ScreenAndAuditAsync("batch", new[] { new ProgramGenome("different") }, 77, 111).AsTask());
        Assert.Equal(before, restored.CaptureState());
    }

    [Fact]
    public async Task BudgetShortAuditRemainsIncompleteAndPotentiallyUseful()
    {
        var report = await Session(Ledger(6)).ScreenAndAuditAsync("batch", new[] { new ProgramGenome("good") }, 7, 11);
        Assert.False(report.IsComplete); Assert.Equal(6, report.ChargedCostUnits);
        Assert.Equal(1, report.Audit!.FalseRejectionRateUpper);
    }

    [Fact]
    public async Task CancellationRetainsReceiptAndDoesNotClaimCompleteBatch()
    {
        using var cancel = new CancellationTokenSource();
        var evaluator = new DelegateProgramFitnessEvaluator((_, _, _) => { cancel.Cancel(); return new ValueTask<EvolutionTaskResult>(Result(0)); });
        var report = await Session(Ledger(), screen: evaluator).ScreenAndAuditAsync("batch",
            new[] { new ProgramGenome("good"), new ProgramGenome("bad") }, 7, 11, cancel.Token);
        Assert.True(report.Canceled); Assert.False(report.IsComplete);
        Assert.Equal(2, report.Requested); Assert.Single(report.Entries); Assert.Equal(2, report.ChargedCostUnits);
    }

    [Fact]
    public async Task PreCancellationConsumesNoSlotOrMoney()
    {
        using var cancel = new CancellationTokenSource(); cancel.Cancel(); var ledger = Ledger();
        await Assert.ThrowsAsync<OperationCanceledException>(() => Session(ledger)
            .ScreenAndAuditAsync("batch", new[] { new ProgramGenome("good") }, 7, 11, cancel.Token).AsTask());
        Assert.Equal(0, ledger.Snapshot().Admitted);
    }

    [Fact]
    public async Task DeclaredReuseInCorrectnessCannotBeLostByFitnessMerge()
    {
        var origin = new EvolutionMeasurementOrigin(new string('a', 64), "run", "evaluation", new[] { "old-sample" },
            DateTimeOffset.UtcNow, 1, "calls", "stats", EvolutionMeasurementOriginKind.PersistentReuse);
        var reused = new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(Result(1).WithMeasurementOrigin(origin)));
        var ledger = Ledger();
        var report = await Session(ledger, correctness: reused).ChallengeAsync(0, new("good"), new("bad"), 7);
        Assert.False(report.IsConfirmed); Assert.Equal(2, report.ChargedCostUnits); Assert.Equal(1, ledger.Snapshot().Unknown);
    }

    [Fact]
    public async Task VersionMutationFailsClosedWithConservativeCharge()
    {
        var backend = new ChangingEvaluator(); var ledger = Ledger();
        var report = await Session(ledger, screen: backend).ScreenAndAuditAsync("batch", new[] { new ProgramGenome("good") }, 7, 11);
        Assert.False(report.IsComplete); Assert.Equal(2, report.ChargedCostUnits); Assert.Equal(1, ledger.Snapshot().Unknown);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(129)]
    public async Task PopulationBoundsRefuseWork(int count)
    {
        var ledger = Ledger();
        await Assert.ThrowsAsync<ArgumentException>(() => Session(ledger).ScreenAndAuditAsync("batch",
            Enumerable.Range(0, count).Select(i => new ProgramGenome(i.ToString())), 7, 11).AsTask());
        Assert.Equal(0, ledger.Snapshot().Admitted);
    }

    private sealed class ChangingEvaluator : IProgramFitnessEvaluator
    {
        public string Id => "changing";
        public string VersionHash { get; private set; } = "v1";
        public ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
        { VersionHash = "v2"; return new(Result(1)); }
    }
}
