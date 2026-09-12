using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class ProgramEvolutionPreflightTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "preflight-" + Guid.NewGuid().ToString("N"));

    [Theory]
    [InlineData(0)]
    [InlineData(0.5)]
    [InlineData(1)]
    public async Task CompletedIsNotCorrectnessAndFailedSeedsNeverReachFitnessOrProposal(double correctness)
    {
        int fitness = 0, checks = 0;
        var variation = new Variation();
        var builder = Builder(variation, () => { fitness++; return Result(2, 3); });
        builder.ConfigureProgramCorrectness(new DelegateProgramFitnessEvaluator((_, _, _) =>
        {
            checks++;
            return new ValueTask<EvolutionTaskResult>(Result(correctness, 7));
        }));
        var report = await builder.PreflightProgramEvolutionAsync();
        Assert.Equal(correctness == 1, report.IsReady);
        Assert.Equal(correctness == 1 ? 1 : 0, fitness);
        Assert.Equal(1, checks);
        Assert.Equal(0, variation.Calls);
        Assert.Equal(7, report.CorrectnessCostUnits);
        Assert.Equal(correctness == 1 ? 3d : null, report.AdditionalFitnessCostUnits);
        Assert.False(report.SharedCorrectnessAndFitness);
        Assert.True(report.OutputLocationsChecked);
        Assert.Empty(Directory.GetFiles(_root)); // Only the owned exclusive probe was removed.
    }

    [Fact]
    public async Task CustomFitnessWithoutCorrectnessIsNotReadyEvenWhenItsScoreIsOne()
    {
        int calls = 0;
        var builder = Builder(new Variation(), () => { calls++; return Result(1, 1); });
        var result = await builder.PreflightProgramEvolutionAsync();
        Assert.False(result.IsReady);
        Assert.Equal("correctness_not_configured", result.Code);
        Assert.Equal(0, calls);
        Assert.Null(result.CorrectnessCostUnits);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task WrongCorrectnessDirectionOrConstraintPreventsFitness(bool wrongDirection)
    {
        int calls = 0;
        var builder = Builder(new Variation(), () => { calls++; return Result(1, 1); });
        builder.ConfigureProgramCorrectness(new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(
            new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1,
                wrongDirection ? EvolutionOptimizationDirection.Minimize : EvolutionOptimizationDirection.Maximize,
                constraintViolations: wrongDirection ? null : new[] { 0.1 }, costUnits: 2))));
        Assert.False((await builder.PreflightProgramEvolutionAsync()).IsReady);
        Assert.Equal(0, calls);
    }

    [Fact]
    public async Task CorrectSeedWithIncompatibleFitnessDirectionStillFailsArchivePreflight()
    {
        var builder = Builder(new Variation(), () => new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1,
            EvolutionOptimizationDirection.Minimize, costUnits: 5));
        builder.ConfigureProgramCorrectness(new DelegateProgramFitnessEvaluator(_ => 1));
        var result = await builder.PreflightProgramEvolutionAsync();
        Assert.False(result.IsReady);
        Assert.Equal("seed_fitness_or_archive_rejected", result.Code);
        Assert.Equal(5, result.AdditionalFitnessCostUnits);
    }

    [Fact]
    public async Task PreflightDoesNotAppendConfiguredSeedsOrDescriptorsToTheBuildersStoredOptions()
    {
        var builder = Builder(new Variation(), () => Result(1, 1));
        builder.ConfigureProgramCorrectness(new DelegateProgramFitnessEvaluator(_ => 1));
        builder.ConfigureEvolutionSeeds(new EvolutionSeedOptions { ProgramSources = new List<string> { "return 9;" } });
        var first = await builder.PreflightProgramEvolutionAsync();
        var second = await builder.PreflightProgramEvolutionAsync();
        Assert.True(first.IsReady); Assert.True(second.IsReady);
        Assert.Equal(first.SeedGenomeId, second.SeedGenomeId);
        Assert.Equal(new ProgramGenome("return 9;", ProgramLanguage.CSharp).Id, first.SeedGenomeId);
        var view = (IConfiguredView<double, Matrix<double>, Vector<double>>)builder;
        Assert.Single(view.ConfiguredProgramEvolution!.SeedPrograms);
        Assert.Empty(view.ConfiguredEvolution!.Descriptors);
    }

    [Fact]
    public async Task MissingCheckpointAndUnwritableOutputAreRefusedBeforeAnyEvaluatorCall()
    {
        int calls = 0;
        var builder = Builder(new Variation(), () => { calls++; return Result(1, 1); });
        builder.ConfigureEvolution(new EvolutionOptions { RunId = "absent", OutputDirectory = _root, Resume = true });
        await Assert.ThrowsAsync<ArgumentException>(() => builder.PreflightProgramEvolutionAsync());
        Directory.CreateDirectory(_root);
        string obstruction = Path.Combine(_root, "file");
        File.WriteAllText(obstruction, "preserve");
        builder.ConfigureEvolution(new EvolutionOptions { OutputDirectory = obstruction });
        await Assert.ThrowsAsync<IOException>(() => builder.PreflightProgramEvolutionAsync());
        Assert.Equal("preserve", File.ReadAllText(obstruction));
        Assert.Equal(0, calls);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(4097)]
    public async Task PreflightLimitsAndCancellationAreValidatedBeforeDispatch(int limit)
    {
        int calls = 0;
        var builder = Builder(new Variation(), () => { calls++; return Result(1, 1); });
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => builder.PreflightProgramEvolutionAsync(limit));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => builder.PreflightProgramEvolutionAsync(cancellationToken: cancellation.Token));
        Assert.Equal(0, calls);
    }

    private AiModelBuilder<double, Matrix<double>, Vector<double>> Builder(Variation variation, Func<EvolutionTaskResult> evaluate)
    {
        var programs = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.CSharp, CustomVariation = variation,
            CustomFitnessEvaluator = new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(evaluate()))
        };
        programs.SeedPrograms.Add("return 1;");
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        builder.ConfigureProgramEvolution(programs);
        builder.ConfigureEvolution(new EvolutionOptions { RunId = "preflight", OutputDirectory = _root, MaxEvaluationAttempts = 2 });
        return builder;
    }

    private static EvolutionTaskResult Result(double quality, double cost) => new(EvolutionEvaluationStatus.Completed, quality, costUnits: cost);
    private sealed class Variation : IProgramVariationOperator
    {
        public int Calls { get; private set; }
        public string Id => "preflight-fixture";
        public string VersionHash => "preflight-fixture-v1";
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default)
        { Calls++; return new(new ProgramGenome("return 2;", ProgramLanguage.CSharp)); }
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        public ProgramEvolutionLlmUsage GetUsage() => new();
    }
    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); // This instance's own GUID directory only.
    }
}
