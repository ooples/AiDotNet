using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class CorrectnessCachePolicyTests
{
    [Fact]
    public async Task PublicTestGateAlsoDisablesOuterMemoizationWithoutRerunningDuplicates()
    {
        var observer = new Recorder();
        var runner = new ConstantRunner();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .WithProgramTestCaseCorrectness().ObserveProgramEvolution(observer);
        var programs = new ProgramEvolutionOptions { Language = ProgramLanguage.Python, CustomVariation = new RepeatSeed() };
        programs.SeedPrograms.Add("print(7)");
        programs.TestCases.Add(new ProgramInputOutputExample { Input = string.Empty, ExpectedOutput = "7" });
        builder.ConfigureProgramExecutionEngine(runner);
        builder.ConfigureProgramEvolution(programs);
        builder.ConfigureEvolution(new EvolutionOptions { MaxEvaluationAttempts = 2, MaxProposals = 2, EnableEvaluationCache = true });
        var result = await builder.BuildAsync();
        Assert.Equal(0, observer.CacheHits);
        Assert.Equal(1, runner.Calls);
        Assert.Equal(1, result.EvolutionSummary!.StatusCounts["Duplicate"]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EngineMemoizationMustNotBypassAnExplicitFreshCorrectnessGate(bool gated)
    {
        int correctnessCalls = 0;
        var observer = new Recorder();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>().ObserveProgramEvolution(observer);
        if (gated) builder.ConfigureProgramCorrectness(new DelegateProgramFitnessEvaluator(_ => { correctnessCalls++; return 1; }));
        var programs = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.Python,
            CustomFitnessEvaluator = new DelegateProgramFitnessEvaluator(_ => 1),
            CustomVariation = new RepeatSeed()
        };
        programs.SeedPrograms.Add("print(7)");
        builder.ConfigureProgramEvolution(programs);
        builder.ConfigureEvolution(new EvolutionOptions { MaxEvaluationAttempts = 2, MaxProposals = 2, EnableEvaluationCache = true });
        var result = await builder.BuildAsync();
        Assert.Equal(gated ? 0 : 1, observer.CacheHits);
        Assert.Equal(gated ? 1 : 0, correctnessCalls);
        Assert.Equal(gated ? 1 : 2, result.EvolutionSummary!.CompletedEvaluations);
    }

    private sealed class Recorder : IEvolutionObserver<ProgramGenome>
    {
        public int CacheHits { get; private set; }
        public ValueTask OnEventAsync(EvolutionEvent<ProgramGenome> item, CancellationToken cancellationToken = default)
        {
            if (item.Kind == EvolutionEventKind.Evaluated && item.Evaluation?.CacheStatus == EvolutionCacheStatus.Hit) CacheHits++;
            return default;
        }
    }

    private sealed class ConstantRunner : IProgramExecutionEngine
    {
        public int Calls { get; private set; }
        public bool TryExecute(ProgramLanguage language, string sourceCode, string input,
            out string output, out string? errorMessage, CancellationToken cancellationToken = default)
        {
            output = "7";
            errorMessage = null;
            return true;
        }
        public Task<ProgramExecuteResponse> ExecuteAsync(ProgramExecuteRequest request, CancellationToken cancellationToken = default)
        {
            Calls++;
            return Task.FromResult(new ProgramExecuteResponse { Success = true, Language = request.Language, ExitCode = 0, StdOut = "7" });
        }
    }

    private sealed class RepeatSeed : IProgramVariationOperator
    {
        public string Id => "repeat-seed";
        public string VersionHash => "repeat-seed-v1";
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default) =>
            new(new ProgramGenome("print(7)", ProgramLanguage.Python));
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        public ProgramEvolutionLlmUsage GetUsage() => new();
    }
}
