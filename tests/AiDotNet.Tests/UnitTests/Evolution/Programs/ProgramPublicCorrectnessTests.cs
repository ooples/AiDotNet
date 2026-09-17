using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Moq;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramPublicCorrectnessTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task AHighScoringWrongProposalCannotWinWhenPublicChecksAreRequired(bool strict)
    {
        int fitness = 0, execution = 0;
        var builder = Builder(strict, () => execution++, () => fitness++, scriptFitness: true);
        var result = await builder.BuildAsync();
        Assert.Equal(strict ? "print(7)" : "print(6)", result.ProgramEvolution!.BestProgram!.Source);
        Assert.Equal(strict ? 1 : 2, fitness);
        Assert.Equal(strict ? 2 : 0, execution);
        Assert.Equal(strict ? 1 : 0, result.EvolutionSummary!.StatusCounts.TryGetValue("Rejected", out long rejected) ? rejected : 0);
    }

    [Fact]
    public async Task BuiltInPassFractionBecomesAHardGateWithoutDoublingRunnerCalls()
    {
        int calls = 0;
        var builder = Builder(true, () => calls++, () => { }, scriptFitness: false);
        var result = await builder.BuildAsync();
        Assert.Equal(2, calls);
        Assert.Equal(1, result.EvolutionSummary!.StatusCounts["Rejected"]);
        Assert.Equal("print(7)", result.ProgramEvolution!.BestProgram!.Source);
    }

    [Fact]
    public async Task PreflightStillChecksPublicTestsWhenAnExplicitCorrectnessProviderApprovesTheSeed()
    {
        int fitness = 0, execution = 0;
        var builder = Builder(true, () => execution++, () => fitness++, scriptFitness: true, goodSeed: false);
        builder.ConfigureProgramCorrectness(new DelegateProgramFitnessEvaluator(_ => 1));
        var result = await builder.PreflightProgramEvolutionAsync();
        Assert.False(result.IsReady);
        Assert.Equal(1, execution);
        Assert.Equal(0, fitness);
        Assert.Equal(1, result.AdditionalFitnessCostUnits);
    }

    [Fact]
    public async Task BuiltInPreflightSharesItsSinglePublicEvaluationEvenWithTheHardGateEnabled()
    {
        int calls = 0;
        var result = await Builder(true, () => calls++, () => { }, scriptFitness: false).PreflightProgramEvolutionAsync();
        Assert.True(result.IsReady);
        Assert.True(result.SharedCorrectnessAndFitness);
        Assert.Equal(1, calls);
    }

    private static AiModelBuilder<double, Matrix<double>, Vector<double>> Builder(bool strict, Action executed,
        Action fitness, bool scriptFitness, bool goodSeed = true)
    {
        var runner = new Mock<IProgramExecutionEngine>();
        runner.Setup(item => item.ExecuteAsync(It.IsAny<ProgramExecuteRequest>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync((ProgramExecuteRequest request, CancellationToken _) =>
            {
                bool script = request.SourceCode.Contains("US24_FITNESS", StringComparison.Ordinal);
                if (script) fitness();
                else executed();
                return new ProgramExecuteResponse
                {
                    Success = true, Language = request.Language, ExitCode = 0,
                    StdOut = script ? "{\"quality\":" + (request.StdIn == "print(7)" ? "1" : "100") + "}"
                        : request.SourceCode == "print(7)" ? "7" : "6"
                };
            });
        var programs = new ProgramEvolutionOptions { Language = ProgramLanguage.Python, CustomVariation = new WrongVariation() };
        programs.SeedPrograms.Add(goodSeed ? "print(7)" : "print(6)");
        programs.TestCases.Add(new ProgramInputOutputExample { Input = string.Empty, ExpectedOutput = "7" });
        if (scriptFitness) programs.EvaluatorScript = "def evaluate():\n    pass\n# US24_FITNESS";
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        if (strict) builder.WithProgramTestCaseCorrectness();
        builder.ConfigureProgramExecutionEngine(runner.Object);
        builder.ConfigureProgramEvolution(programs);
        builder.ConfigureEvolution(new EvolutionOptions { MaxEvaluationAttempts = 2, MaxProposals = 3 });
        return builder;
    }

    private sealed class WrongVariation : IProgramVariationOperator
    {
        public string Id => "authored-wrong-proposal";
        public string VersionHash => "authored-wrong-v1";
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default) =>
            new(new ProgramGenome("print(6)", ProgramLanguage.Python));
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        public ProgramEvolutionLlmUsage GetUsage() => new();
    }
}
