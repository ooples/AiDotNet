using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramFitnessEvaluatorTests
{
    private static readonly EvolutionEvaluationContext Context = new(0, 1234UL, 7UL, 1);

    private static ProgramGenome Genome(string source = "print(1)") =>
        new(source, ProgramLanguage.Python);

    private static ProgramInputOutputExample Example(string input, string expected) =>
        new() { Input = input, ExpectedOutput = expected };

    [Fact]
    public async Task DelegateEvaluatorReturnsTheSuppliedScore()
    {
        var evaluator = new DelegateProgramFitnessEvaluator(genome => genome.LineCount);
        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome("a\nb\nc"), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(3.0, result.Quality);
        Assert.Equal(EvolutionOptimizationDirection.Maximize, result.Direction);
        Assert.Equal("delegate-program-evaluator", evaluator.Id);
    }

    [Fact]
    public async Task DelegateEvaluatorRejectsNonFiniteScores()
    {
        var evaluator = new DelegateProgramFitnessEvaluator(_ => double.NaN);
        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal("program_score_not_finite", result.Diagnostics[0].Code);
    }

    [Fact]
    public async Task DelegateEvaluatorSupportsMinimizationAndAsynchronousBodies()
    {
        var minimizing = new DelegateProgramFitnessEvaluator(
            genome => genome.NormalizedSource.Length,
            "size",
            "size-v1",
            EvolutionOptimizationDirection.Minimize);
        Assert.Equal(EvolutionOptimizationDirection.Minimize, (await minimizing.EvaluateAsync(Genome(), Context)).Direction);

        var asynchronous = new DelegateProgramFitnessEvaluator(
            (genome, context, cancellationToken) => new ValueTask<EvolutionTaskResult>(
                EvolutionTaskResult.Completed(context.SeedStream, new Dictionary<string, double>())),
            "async",
            "async-v1");
        Assert.Equal(7.0, (await asynchronous.EvaluateAsync(Genome(), Context)).Quality);
    }

    [Fact]
    public void DelegateEvaluatorValidatesItsArguments()
    {
#pragma warning disable CS8600, CS8625
        Assert.Throws<ArgumentNullException>(() => new DelegateProgramFitnessEvaluator((Func<ProgramGenome, double>)null));
#pragma warning restore CS8600, CS8625
        Assert.Throws<ArgumentException>(() => new DelegateProgramFitnessEvaluator(_ => 1.0, " "));
    }

    [Fact]
    public async Task InputOutputEvaluatorScoresThePassingFraction()
    {
        var engine = new FakeProgramExecutionEngine((_, input) => FakeExecutionOutcome.Success(input + "!"));
        var evaluator = new InputOutputProgramFitnessEvaluator(
            engine,
            new[] { Example("a", "a!"), Example("b", "b!"), Example("c", "wrong") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.NotNull(result.Quality);
        Assert.Equal(2.0 / 3.0, result.Quality.Value, 10);
        Assert.Equal(result.Quality.Value, result.Descriptors["passRate"], 10);
        Assert.Equal(3, engine.Calls);
        Assert.Equal(3.0, result.CostUnits);
        Assert.Equal(ProgramLanguage.Python, engine.LastLanguage);
    }

    [Fact]
    public async Task InputOutputEvaluatorPassesTheRawSourceToTheEngine()
    {
        string? seen = null;
        var engine = new FakeProgramExecutionEngine((source, _) =>
        {
            seen = source;
            return FakeExecutionOutcome.Success("ok");
        });

        var evaluator = new InputOutputProgramFitnessEvaluator(engine, new[] { Example("in", "ok") });
        await evaluator.EvaluateAsync(new ProgramGenome("print(1)\r\n", ProgramLanguage.Python), Context);

        Assert.Equal("print(1)\r\n", seen);
    }

    [Theory]
    [InlineData(ProgramOutputComparison.Ordinal, "  ok  ", false)]
    [InlineData(ProgramOutputComparison.TrimmedOrdinal, "  ok  ", true)]
    [InlineData(ProgramOutputComparison.TrimmedOrdinal, "OK", false)]
    [InlineData(ProgramOutputComparison.TrimmedOrdinalIgnoreCase, "OK", true)]
    [InlineData(ProgramOutputComparison.NormalizedWhitespace, "o k", false)]
    [InlineData(ProgramOutputComparison.NormalizedWhitespace, " ok ", true)]
    public async Task ComparisonModesControlStrictness(ProgramOutputComparison comparison, string actual, bool expectedPass)
    {
        var engine = new FakeProgramExecutionEngine((_, _) => FakeExecutionOutcome.Success(actual));
        var evaluator = new InputOutputProgramFitnessEvaluator(engine, new[] { Example("in", "ok") }, comparison);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);
        Assert.NotNull(result.Quality);
        Assert.Equal(expectedPass ? 1.0 : 0.0, result.Quality.Value);
    }

    [Fact]
    public async Task NormalizedWhitespaceComparisonCollapsesInternalRuns()
    {
        var engine = new FakeProgramExecutionEngine((_, _) => FakeExecutionOutcome.Success("1\t2   3\n"));
        var evaluator = new InputOutputProgramFitnessEvaluator(
            engine, new[] { Example("in", "1 2 3") }, ProgramOutputComparison.NormalizedWhitespace);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);
        Assert.Equal(1.0, result.Quality);
    }

    [Fact]
    public async Task EngineFailuresBecomeBoundedRedactedDiagnostics()
    {
        string longError = new string('e', 4000);
        var engine = new FakeProgramExecutionEngine((_, _) => FakeExecutionOutcome.Failure(longError));
        var evaluator = new InputOutputProgramFitnessEvaluator(engine, new[] { Example("in", "ok") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.0, result.Quality);
        EvolutionDiagnostic diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal("program_io_execution_failed", diagnostic.Code);
        Assert.True(diagnostic.IsRedacted);
        Assert.True(diagnostic.Message.Length < 300);
    }

    [Fact]
    public async Task DiagnosticsAreCappedForVeryBadCandidates()
    {
        var examples = new List<ProgramInputOutputExample>();
        for (int index = 0; index < 40; index++) examples.Add(Example("in", "ok"));
        var engine = new FakeProgramExecutionEngine((_, _) => FakeExecutionOutcome.Success("no"));
        var evaluator = new InputOutputProgramFitnessEvaluator(engine, examples);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);
        Assert.Equal(0.0, result.Quality);
        Assert.True(result.Diagnostics.Count <= 8);
    }

    [Fact]
    public async Task EngineExceptionsDoNotEscape()
    {
        var evaluator = new InputOutputProgramFitnessEvaluator(
            new ThrowingProgramExecutionEngine(), new[] { Example("in", "ok") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.0, result.Quality);
        Assert.Equal("program_io_engine_threw", result.Diagnostics[0].Code);
        Assert.Contains("InvalidOperationException", result.Diagnostics[0].Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CancellationIsReportedAsCanceled()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();
        var engine = new FakeProgramExecutionEngine((_, _) => FakeExecutionOutcome.Success("ok"));
        var evaluator = new InputOutputProgramFitnessEvaluator(engine, new[] { Example("in", "ok") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context, source.Token);

        Assert.Equal(EvolutionEvaluationStatus.Canceled, result.Status);
        Assert.Equal(0, engine.Calls);
    }

    [Fact]
    public void VersionHashTracksExamplesAndComparison()
    {
        var engine = new FakeProgramExecutionEngine((_, _) => FakeExecutionOutcome.Success("ok"));
        string first = new InputOutputProgramFitnessEvaluator(engine, new[] { Example("a", "1") }).VersionHash;
        string same = new InputOutputProgramFitnessEvaluator(engine, new[] { Example("a", "1") }).VersionHash;
        string differentExamples = new InputOutputProgramFitnessEvaluator(engine, new[] { Example("a", "2") }).VersionHash;
        string differentComparison = new InputOutputProgramFitnessEvaluator(
            engine, new[] { Example("a", "1") }, ProgramOutputComparison.Ordinal).VersionHash;

        Assert.Equal(first, same);
        Assert.NotEqual(first, differentExamples);
        Assert.NotEqual(first, differentComparison);
    }

    [Fact]
    public void InputOutputEvaluatorValidatesItsArguments()
    {
        var engine = new FakeProgramExecutionEngine((_, _) => FakeExecutionOutcome.Success("ok"));
        Assert.Throws<ArgumentException>(() =>
            new InputOutputProgramFitnessEvaluator(engine, Array.Empty<ProgramInputOutputExample>()));
#pragma warning disable CS8600, CS8625
        Assert.Throws<ArgumentNullException>(() =>
            new InputOutputProgramFitnessEvaluator(null, new[] { Example("a", "b") }));
#pragma warning restore CS8600, CS8625
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InputOutputProgramFitnessEvaluator(engine, new[] { Example("a", "b") }, (ProgramOutputComparison)99));
    }

    [Fact]
    public void ExamplesAreCopiedDefensively()
    {
        var engine = new FakeProgramExecutionEngine((_, _) => FakeExecutionOutcome.Success("ok"));
        var example = Example("a", "b");
        var evaluator = new InputOutputProgramFitnessEvaluator(engine, new[] { example });

        example.ExpectedOutput = "mutated";
        Assert.Equal("b", evaluator.Examples[0].ExpectedOutput);
    }
}
