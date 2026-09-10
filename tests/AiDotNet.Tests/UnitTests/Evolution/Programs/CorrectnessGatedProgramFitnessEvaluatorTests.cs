using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class CorrectnessGatedProgramFitnessEvaluatorTests
{
    private static readonly ProgramGenome Genome = new("return 1;", ProgramLanguage.CSharp);
    private static readonly EvolutionEvaluationContext Context = new(0, 1, 1, 1);

    [Theory]
    [InlineData(0)]
    [InlineData(0.99)]
    [InlineData(1.01)]
    public async Task FailedChecksNeverInvokeFitness(double fraction)
    {
        var fitness = new Stub(Completed(100));
        var validation = new Stub(Completed(fraction, costUnits: 3));
        EvolutionTaskResult result = await new CorrectnessGatedProgramFitnessEvaluator(validation, fitness).EvaluateAsync(Genome, Context);
        Assert.Equal(EvolutionEvaluationStatus.Rejected, result.Status);
        Assert.Null(result.Quality);
        Assert.Equal(3, result.CostUnits);
        Assert.Equal(0, fitness.Calls);
    }

    [Fact]
    public async Task PassingChecksChargeBothStagesAndPreserveFitnessMetadata()
    {
        var fitness = new Stub(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 2,
            EvolutionOptimizationDirection.Minimize, costUnits: 5,
            metrics: new Dictionary<string, double> { ["runtime"] = 2 }));
        EvolutionTaskResult result = await new CorrectnessGatedProgramFitnessEvaluator(
            new Stub(Completed(1, costUnits: 3)), fitness).EvaluateAsync(Genome, Context);
        Assert.Equal(8, result.CostUnits);
        Assert.Equal(2, result.Quality);
        Assert.Equal(EvolutionOptimizationDirection.Minimize, result.Direction);
        Assert.Equal(2, result.Metrics["runtime"]);
        Assert.Equal(1, fitness.Calls);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ConstraintsAreHardGatesOnEitherStage(bool validationViolation)
    {
        var validation = new Stub(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1,
            constraintViolations: new[] { validationViolation ? 1.0 : 0.0 }));
        var fitness = new Stub(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 100,
            constraintViolations: new[] { 1.0 }));
        EvolutionTaskResult result = await new CorrectnessGatedProgramFitnessEvaluator(validation, fitness).EvaluateAsync(Genome, Context);
        Assert.Equal(EvolutionEvaluationStatus.Rejected, result.Status);
        Assert.Equal(validationViolation ? 0 : 1, fitness.Calls);
    }

    [Fact]
    public async Task FailedValidationRetainsItsEvidence()
    {
        EvolutionTaskResult failure = EvolutionTaskResult.Failed("incorrect", "Public reference comparison failed.");
        var fitness = new Stub(Completed(100));
        EvolutionTaskResult result = await new CorrectnessGatedProgramFitnessEvaluator(new Stub(failure), fitness).EvaluateAsync(Genome, Context);
        Assert.Same(failure, result);
        Assert.Equal(0, fitness.Calls);
    }

    [Fact]
    public async Task CancellationBeforeScoringDoesNotSpendFitnessWork()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var validation = new Stub(Completed(1));
        var fitness = new Stub(Completed(2));
        var gate = new CorrectnessGatedProgramFitnessEvaluator(validation, fitness);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => gate.EvaluateAsync(Genome, Context, cancellation.Token).AsTask());
        Assert.Equal(0, validation.Calls);
        Assert.Equal(0, fitness.Calls);
    }

    [Fact]
    public void CheckDataVersionChangesCheckpointCompatibility()
    {
        var fitness = new Stub(Completed(2));
        var first = new CorrectnessGatedProgramFitnessEvaluator(new Stub(Completed(1), "checks-v1"), fitness);
        var second = new CorrectnessGatedProgramFitnessEvaluator(new Stub(Completed(1), "checks-v2"), fitness);
        Assert.NotEqual(first.VersionHash, second.VersionHash);
    }

    [Fact]
    public async Task DescriptorMergingRetainsReportingMetricsAndRepairArtifacts()
    {
        var evaluator = new Stub(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1,
            metrics: new Dictionary<string, double> { ["runtime"] = 3 },
            artifacts: new[] { new EvolutionArtifact("public-check", "Reference comparison passed.") }));
        var task = new ProgramEvolutionTask(evaluator, new ProgramDescriptorSet(new IProgramDescriptor[] { new ProgramLengthDescriptor() }));
        var candidate = new EvolutionCandidate<ProgramGenome>(0, new EvolutionCanonicalGenome<ProgramGenome>(Genome, Genome.Id),
            new EvolutionLineage(null, null, "seed", null, 0, 0, 0));
        EvolutionTaskResult result = await task.EvaluateAsync(candidate, Context);
        Assert.Equal(3, result.Metrics["runtime"]);
        Assert.Equal("Reference comparison passed.", Assert.Single(result.Artifacts).Text);
        Assert.NotEmpty(result.Descriptors);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task NullStageResultsFailClosedAndRetainKnownCosts(bool nullValidation)
    {
        var validation = new Stub(nullValidation ? null! : Completed(1, 3));
        var fitness = new Stub(null!);
        EvolutionTaskResult result = await new CorrectnessGatedProgramFitnessEvaluator(validation, fitness).EvaluateAsync(Genome, Context);
        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal(nullValidation ? 0 : 3, result.CostUnits);
        Assert.Equal(nullValidation ? 0 : 1, fitness.Calls);
    }

    [Fact]
    public async Task MinimizationIsNotAValidPassFractionContract()
    {
        var validation = new Stub(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1,
            EvolutionOptimizationDirection.Minimize));
        var fitness = new Stub(Completed(5));
        EvolutionTaskResult result = await new CorrectnessGatedProgramFitnessEvaluator(validation, fitness).EvaluateAsync(Genome, Context);
        Assert.Equal(EvolutionEvaluationStatus.Rejected, result.Status);
        Assert.Equal(0, fitness.Calls);
    }

    [Fact]
    public async Task FitnessFailuresRetainTheCostOfBothStages()
    {
        var fitness = new Stub(new EvolutionTaskResult(EvolutionEvaluationStatus.TimedOut, costUnits: 7));
        EvolutionTaskResult result = await new CorrectnessGatedProgramFitnessEvaluator(new Stub(Completed(1, 3)), fitness).EvaluateAsync(Genome, Context);
        Assert.Equal(EvolutionEvaluationStatus.TimedOut, result.Status);
        Assert.Equal(10, result.CostUnits);
    }

    [Fact]
    public async Task RejectsMissingDependenciesAndInvocationArguments()
    {
        var stage = new Stub(Completed(1));
        Assert.Throws<ArgumentNullException>(() => new CorrectnessGatedProgramFitnessEvaluator(null!, stage));
        Assert.Throws<ArgumentNullException>(() => new CorrectnessGatedProgramFitnessEvaluator(stage, null!));
        var gate = new CorrectnessGatedProgramFitnessEvaluator(stage, stage);
        await Assert.ThrowsAsync<ArgumentNullException>(() => gate.EvaluateAsync(null!, Context).AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => gate.EvaluateAsync(Genome, null!).AsTask());
    }

    private static EvolutionTaskResult Completed(double quality, double costUnits = 0) =>
        new(EvolutionEvaluationStatus.Completed, quality, costUnits: costUnits);

    private sealed class Stub(EvolutionTaskResult result, string version = "v1") : IProgramFitnessEvaluator
    {
        public string Id => "stub";
        public string VersionHash => version;
        public int Calls { get; private set; }
        public ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context,
            CancellationToken cancellationToken = default)
        {
            Calls++;
            return new ValueTask<EvolutionTaskResult>(result);
        }
    }
}
