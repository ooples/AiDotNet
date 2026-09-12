using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class CorrectnessGateAdversarialTests
{
    private static readonly ProgramGenome Program = new("print(7)", ProgramLanguage.Python);
    private static readonly EvolutionEvaluationContext Context = new(0, 7, 0, 1);

    [Theory]
    [InlineData(EvolutionMeasurementOriginKind.Measured, true)]
    [InlineData(EvolutionMeasurementOriginKind.PersistentReuse, false)]
    [InlineData(EvolutionMeasurementOriginKind.RunLocalReuse, false)]
    [InlineData(EvolutionMeasurementOriginKind.MigrationCopy, false)]
    public async Task PreviouslyMeasuredCorrectnessCannotAuthorizeANewFitnessCall(EvolutionMeasurementOriginKind kind, bool allowed)
    {
        var origin = new EvolutionMeasurementOrigin(new string('a', 64), "authored-run", "check-0", new[] { "sample-0" },
            new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero), 2, "runner-dispatch", "checks-v1", kind);
        int calls = 0;
        var correctness = new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(
            new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 2).WithMeasurementOrigin(origin)));
        var fitness = new DelegateProgramFitnessEvaluator((_, _, _) =>
        {
            calls++;
            return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 5, costUnits: 3));
        });
        var result = await new CorrectnessGatedProgramFitnessEvaluator(correctness, fitness).EvaluateAsync(Program, Context);
        Assert.Equal(allowed ? 1 : 0, calls);
        Assert.Equal(allowed ? EvolutionEvaluationStatus.Completed : EvolutionEvaluationStatus.Rejected, result.Status);
        Assert.Equal(allowed ? 5 : 2, result.CostUnits);
        if (!allowed) Assert.Same(origin, result.MeasurementOrigin);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task StageIdentityChangesBeforeOrDuringDispatchAreNotAccepted(bool fitnessStage, bool during)
    {
        var correctness = new MutableEvaluator();
        var fitness = new MutableEvaluator();
        var gate = new CorrectnessGatedProgramFitnessEvaluator(correctness, fitness);
        var changed = fitnessStage ? fitness : correctness;
        if (during) changed.ChangeDuringCall = true;
        else changed.VersionHash = "changed";
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await gate.EvaluateAsync(Program, Context));
    }

    private sealed class MutableEvaluator : IProgramFitnessEvaluator
    {
        public string Id => "authored-mutable-stage";
        public string VersionHash { get; set; } = "stage-v1";
        public bool ChangeDuringCall { get; set; }
        public ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context,
            CancellationToken cancellationToken = default)
        {
            if (ChangeDuringCall) VersionHash = "changed";
            return new(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 1));
        }
    }
}
