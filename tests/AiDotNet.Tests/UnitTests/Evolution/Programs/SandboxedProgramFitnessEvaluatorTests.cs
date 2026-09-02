using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class SandboxedProgramFitnessEvaluatorTests
{
    private static readonly EvolutionEvaluationContext Context = new(0, 1234UL, 7UL, 1);

    private static ProgramGenome Genome(string source = "print(1)") => new(source, ProgramLanguage.Python);

    private static ProgramInputOutputExample Example(string input, string expected) =>
        new() { Input = input, ExpectedOutput = expected };

    private static ProgramExecuteResponse Ok(string stdOut, bool truncated = false) => new()
    {
        Success = true,
        Language = ProgramLanguage.Python,
        ExitCode = 0,
        StdOut = stdOut,
        StdOutTruncated = truncated
    };

    private static ProgramExecuteResponse Failed(ProgramExecuteErrorCode code, string error, int exitCode = -1) => new()
    {
        Success = false,
        Language = ProgramLanguage.Python,
        ExitCode = exitCode,
        Error = error,
        ErrorCode = code
    };

    [Fact]
    public async Task QualityIsThePassingFractionAndIsAlsoReportedAsADescriptor()
    {
        var engine = new ScriptedProgramExecutionEngine(request => Ok((request.StdIn ?? string.Empty) + "!"));
        var evaluator = new SandboxedProgramFitnessEvaluator(
            engine, new[] { Example("a", "a!"), Example("b", "b!"), Example("c", "nope") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.NotNull(result.Quality);
        Assert.Equal(2.0 / 3.0, result.Quality.Value, 10);
        Assert.Equal(result.Quality.Value, result.Descriptors["passRate"], 10);
        Assert.Equal(3.0, result.CostUnits);
        Assert.Equal(3, engine.Calls);
    }

    [Fact]
    public async Task TheRawSourceAndTheExampleInputReachTheEngine()
    {
        string? seenSource = null;
        string? seenInput = null;
        ProgramLanguage? seenLanguage = null;
        var engine = new ScriptedProgramExecutionEngine(request =>
        {
            seenSource = request.SourceCode;
            seenInput = request.StdIn;
            seenLanguage = request.Language;
            return Ok("ok");
        });

        var evaluator = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("in", "ok") });
        await evaluator.EvaluateAsync(new ProgramGenome("print(1)\r\n", ProgramLanguage.Python), Context);

        Assert.Equal("print(1)\r\n", seenSource);
        Assert.Equal("in", seenInput);
        Assert.Equal(ProgramLanguage.Python, seenLanguage);
    }

    [Fact]
    public async Task CarriageReturnsAreNormalizedBeforeComparison()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Ok("line one\r\nline two\r\n"));
        var evaluator = new SandboxedProgramFitnessEvaluator(
            engine, new[] { Example("in", "line one\nline two") }, ProgramOutputComparison.Ordinal);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(1.0, result.Quality);
    }

    [Theory]
    [InlineData(ProgramExecuteErrorCode.TimeoutOrCanceled, "program_sandbox_timeout")]
    [InlineData(ProgramExecuteErrorCode.CompilationFailed, "program_sandbox_compile_failed")]
    [InlineData(ProgramExecuteErrorCode.ExecutionFailed, "program_sandbox_execution_failed")]
    [InlineData(ProgramExecuteErrorCode.LanguageNotDetected, "program_sandbox_language_not_detected")]
    [InlineData(ProgramExecuteErrorCode.InvalidRequest, "program_sandbox_invalid_request")]
    public async Task EachKindOfSandboxFailureGetsItsOwnDiagnosticCode(
        ProgramExecuteErrorCode code,
        string expectedDiagnostic)
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Failed(code, "the sandbox said no"));
        var evaluator = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("in", "ok") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.0, result.Quality);
        EvolutionDiagnostic diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal(expectedDiagnostic, diagnostic.Code);
        Assert.True(diagnostic.IsRedacted);
    }

    [Fact]
    public async Task ATruncatedMismatchSaysSoInsteadOfLookingLikeAWrongAnswer()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Ok("partial", truncated: true));
        var evaluator = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("in", "the whole answer") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        EvolutionDiagnostic diagnostic = Assert.Single(result.Diagnostics);
        Assert.Equal("program_sandbox_output_mismatch", diagnostic.Code);
        Assert.Contains("truncated", diagnostic.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LongFailureTextIsBoundedAndRedacted()
    {
        string huge = new string('e', 5000);
        var engine = new ScriptedProgramExecutionEngine(
            _ => Failed(ProgramExecuteErrorCode.ExecutionFailed, huge));
        var evaluator = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("in", "ok") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        EvolutionDiagnostic diagnostic = Assert.Single(result.Diagnostics);
        Assert.True(diagnostic.Message.Length < 300, "Failure text must be bounded before it reaches a checkpoint.");
        Assert.True(diagnostic.IsRedacted);
    }

    [Fact]
    public async Task DiagnosticsAreCappedForAThoroughlyBrokenCandidate()
    {
        var examples = new List<ProgramInputOutputExample>();
        for (int index = 0; index < 40; index++) examples.Add(Example("in", "ok"));

        var engine = new ScriptedProgramExecutionEngine(_ => Ok("wrong"));
        var evaluator = new SandboxedProgramFitnessEvaluator(engine, examples);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(0.0, result.Quality);
        Assert.True(result.Diagnostics.Count <= 8);
    }

    [Fact]
    public async Task AnEngineThatThrowsDoesNotStopTheRun()
    {
        var evaluator = new SandboxedProgramFitnessEvaluator(
            new ThrowingProgramExecutionEngine(), new[] { Example("in", "ok") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.0, result.Quality);
        Assert.Equal("program_sandbox_engine_threw", result.Diagnostics[0].Code);
        Assert.Contains("InvalidOperationException", result.Diagnostics[0].Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CancellationIsReportedAsCanceledWithoutRunningAnything()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();

        var engine = new ScriptedProgramExecutionEngine(_ => Ok("ok"));
        var evaluator = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("in", "ok") });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context, source.Token);

        Assert.Equal(EvolutionEvaluationStatus.Canceled, result.Status);
        Assert.Equal(0, engine.Calls);
    }

    [Theory]
    [InlineData(ProgramOutputComparison.Ordinal, "  ok  ", false)]
    [InlineData(ProgramOutputComparison.TrimmedOrdinal, "  ok  ", true)]
    [InlineData(ProgramOutputComparison.TrimmedOrdinal, "OK", false)]
    [InlineData(ProgramOutputComparison.TrimmedOrdinalIgnoreCase, "OK", true)]
    [InlineData(ProgramOutputComparison.NormalizedWhitespace, " o k ", false)]
    [InlineData(ProgramOutputComparison.NormalizedWhitespace, " ok ", true)]
    public async Task ComparisonModesControlStrictness(
        ProgramOutputComparison comparison,
        string actual,
        bool expectedPass)
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Ok(actual));
        var evaluator = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("in", "ok") }, comparison);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.NotNull(result.Quality);
        Assert.Equal(expectedPass ? 1.0 : 0.0, result.Quality.Value);
    }

    [Fact]
    public void VersionHashTracksExamplesAndComparisonAndDiffersFromTheSynchronousEvaluator()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Ok("ok"));
        string first = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("a", "1") }).VersionHash;
        string same = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("a", "1") }).VersionHash;
        string otherExamples = new SandboxedProgramFitnessEvaluator(engine, new[] { Example("a", "2") }).VersionHash;
        string otherComparison = new SandboxedProgramFitnessEvaluator(
            engine, new[] { Example("a", "1") }, ProgramOutputComparison.Ordinal).VersionHash;
        string synchronous = new InputOutputProgramFitnessEvaluator(engine, new[] { Example("a", "1") }).VersionHash;

        Assert.Equal(first, same);
        Assert.NotEqual(first, otherExamples);
        Assert.NotEqual(first, otherComparison);
        Assert.NotEqual(first, synchronous);
    }

    [Fact]
    public void TheEvaluatorValidatesItsArguments()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Ok("ok"));

        Assert.Throws<ArgumentException>(
            () => new SandboxedProgramFitnessEvaluator(engine, Array.Empty<ProgramInputOutputExample>()));
#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(
            () => new SandboxedProgramFitnessEvaluator(null, new[] { Example("a", "b") }));
#pragma warning restore CS8625
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new SandboxedProgramFitnessEvaluator(
                engine, new[] { Example("a", "b") }, (ProgramOutputComparison)99));
        Assert.Throws<ArgumentException>(
            () => new SandboxedProgramFitnessEvaluator(engine, new[] { Example("a", "b") }, id: "  "));
    }

    [Fact]
    public void ExamplesAreCopiedDefensively()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Ok("ok"));
        var example = Example("a", "b");
        var evaluator = new SandboxedProgramFitnessEvaluator(engine, new[] { example });

        example.ExpectedOutput = "mutated";

        Assert.Equal("b", evaluator.Examples[0].ExpectedOutput);
        Assert.Equal("program-sandbox-evaluator", evaluator.Id);
        Assert.Equal(ProgramOutputComparison.TrimmedOrdinal, evaluator.Comparison);
    }
}
