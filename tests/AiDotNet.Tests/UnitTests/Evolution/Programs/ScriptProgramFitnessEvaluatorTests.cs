using System.Diagnostics;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ScriptProgramFitnessEvaluatorTests
{
    private const string Script = "def evaluate(source): return {\"quality\": 1.0}";

    private static readonly EvolutionEvaluationContext Context = new(0, 1234UL, 7UL, 1);

    private static ProgramGenome Genome(string source = "print(1)") => new(source, ProgramLanguage.Python);

    private static ProgramExecuteResponse Printed(string stdOut, bool truncated = false) => new()
    {
        Success = true,
        Language = ProgramLanguage.Python,
        ExitCode = 0,
        StdOut = stdOut,
        StdOutTruncated = truncated
    };

    private static ScriptProgramFitnessEvaluator Evaluator(
        Func<ProgramExecuteRequest, ProgramExecuteResponse> handler,
        ScriptProgramEvaluationOptions? options = null) =>
        new(new ScriptedProgramExecutionEngine(handler), Script, options);

    [Fact]
    public async Task QualityDescriptorsAndObjectivesAreReadFromTheScriptOutput()
    {
        ScriptProgramFitnessEvaluator evaluator = Evaluator(_ => Printed(
            "{\"quality\": 0.75, \"descriptors\": {\"length\": 0.4, \"depth\": 3}, \"objectives\": [0.75, 0.1]}"));

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.75, result.Quality);
        Assert.Equal(0.4, result.Descriptors["length"]);
        Assert.Equal(3.0, result.Descriptors["depth"]);
        Assert.Equal(new[] { 0.75, 0.1 }, result.Objectives);
        Assert.Equal(EvolutionOptimizationDirection.Maximize, result.Direction);
    }

    [Fact]
    public async Task TheCandidateSourceIsHandedToTheScriptOnStandardInput()
    {
        string? seenStdIn = null;
        string? seenScript = null;
        ProgramLanguage? seenLanguage = null;

        ScriptProgramFitnessEvaluator evaluator = Evaluator(request =>
        {
            seenStdIn = request.StdIn;
            seenScript = request.SourceCode;
            seenLanguage = request.Language;
            return Printed("{\"quality\": 1}");
        });

        await evaluator.EvaluateAsync(Genome("candidate-source"), Context);

        Assert.Equal("candidate-source", seenStdIn);
        Assert.Equal(Script, seenScript);
        Assert.Equal(ProgramLanguage.Python, seenLanguage);
    }

    [Fact]
    public async Task PreambleBeforeTheJsonObjectIsTolerated()
    {
        ScriptProgramFitnessEvaluator evaluator = Evaluator(
            _ => Printed("loading rules...\nscoring...\n{\"quality\": 0.5}\n"));

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.5, result.Quality);
    }

    [Theory]
    [InlineData("not json at all")]
    [InlineData("{\"quality\": }")]
    [InlineData("{\"quality\": \"high\"}")]
    [InlineData("{\"descriptors\": {\"a\": 1}}")]
    [InlineData("{\"quality\": 1, \"descriptors\": {\"a\": \"x\"}}")]
    [InlineData("{\"quality\": 1, \"descriptors\": [1, 2]}")]
    [InlineData("{\"quality\": 1, \"objectives\": {\"a\": 1}}")]
    [InlineData("{\"quality\": 1, \"objectives\": [\"x\"]}")]
    public async Task MalformedMetricsBecomeAFailedResultRatherThanACrash(string stdOut)
    {
        ScriptProgramFitnessEvaluator evaluator = Evaluator(_ => Printed(stdOut));

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal("program_script_invalid_metrics", Assert.Single(result.Diagnostics).Code);
        Assert.Null(result.Quality);
    }

    [Theory]
    [InlineData("{\"quality\": null}")]
    [InlineData("{\"quality\": 1e999}")]
    public async Task AQualityThatIsNotAFiniteNumberIsRejected(string stdOut)
    {
        ScriptProgramFitnessEvaluator evaluator = Evaluator(_ => Printed(stdOut));

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal("program_script_invalid_metrics", Assert.Single(result.Diagnostics).Code);
    }

    [Fact]
    public async Task TheScriptMayComeFromTheOptionsInsteadOfTheConstructorArgument()
    {
        var options = new ScriptProgramEvaluationOptions
        {
            EvaluatorScript = "def evaluate(source): pass"
        };

        string? seenScript = null;
        var engine = new ScriptedProgramExecutionEngine(request =>
        {
            seenScript = request.SourceCode;
            return Printed("{\"quality\": 1}");
        });

        var evaluator = new ScriptProgramFitnessEvaluator(engine, string.Empty, options);
        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal("def evaluate(source): pass", seenScript);
    }

    [Fact]
    public async Task TruncatedScriptOutputIsRefusedInsteadOfPartlyParsed()
    {
        ScriptProgramFitnessEvaluator evaluator = Evaluator(
            _ => Printed("{\"quality\": 0.9, \"descriptors\": {\"a\":", truncated: true));

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal("program_script_metrics_truncated", Assert.Single(result.Diagnostics).Code);
    }

    [Fact]
    public async Task ArtifactsBecomeBoundedRedactedDiagnostics()
    {
        var options = new ScriptProgramEvaluationOptions { MaxArtifactCount = 2, MaxArtifactLength = 40 };
        ScriptProgramFitnessEvaluator evaluator = Evaluator(
            _ => Printed(
                "{\"quality\": 1, \"artifacts\": {\"a\": \"" + new string('z', 500) +
                "\", \"b\": \"second\", \"c\": \"dropped\"}}"),
            options);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(2, result.Diagnostics.Count);
        foreach (EvolutionDiagnostic diagnostic in result.Diagnostics)
        {
            Assert.Equal("program_script_artifact", diagnostic.Code);
            Assert.True(diagnostic.IsRedacted);
            Assert.True(diagnostic.Message.Length <= 100);
        }
    }

    [Fact]
    public async Task ASandboxTimeoutIsReportedAsATimeoutNotAsAZeroScore()
    {
        ScriptProgramFitnessEvaluator evaluator = Evaluator(_ => new ProgramExecuteResponse
        {
            Success = false,
            Language = ProgramLanguage.Python,
            ExitCode = -1,
            Error = "Execution exceeded the 1 second limit and the process tree was terminated.",
            ErrorCode = ProgramExecuteErrorCode.TimeoutOrCanceled
        });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal("program_script_timeout", Assert.Single(result.Diagnostics).Code);
        Assert.Null(result.Quality);
    }

    [Fact]
    public async Task AScriptThatCrashesIsReportedAsAScriptFailure()
    {
        ScriptProgramFitnessEvaluator evaluator = Evaluator(_ => new ProgramExecuteResponse
        {
            Success = false,
            Language = ProgramLanguage.Python,
            ExitCode = 1,
            Error = "Execution failed with exit code 1.",
            ErrorCode = ProgramExecuteErrorCode.ExecutionFailed
        });

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal("program_script_script_failed", Assert.Single(result.Diagnostics).Code);
    }

    [Fact]
    public async Task AnEngineThatThrowsDoesNotStopTheRun()
    {
        var evaluator = new ScriptProgramFitnessEvaluator(new ThrowingProgramExecutionEngine(), Script);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal("program_script_engine_threw", Assert.Single(result.Diagnostics).Code);
    }

    [Fact]
    public async Task CancellationIsReportedWithoutRunningTheScript()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();

        var engine = new ScriptedProgramExecutionEngine(_ => Printed("{\"quality\": 1}"));
        var evaluator = new ScriptProgramFitnessEvaluator(engine, Script);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context, source.Token);

        Assert.Equal(EvolutionEvaluationStatus.Canceled, result.Status);
        Assert.Equal(0, engine.Calls);
    }

    [Fact]
    public void TheEntryPointIsValidatedWhenTheEvaluatorIsBuiltNotWhenItFirstRuns()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Printed("{\"quality\": 1}"));

        ArgumentException failure = Assert.Throws<ArgumentException>(
            () => new ScriptProgramFitnessEvaluator(engine, "print('no entry point here')"));
        Assert.Contains("evaluate", failure.Message, StringComparison.Ordinal);

        // The check is opt-out, and a custom marker is honoured.
        _ = new ScriptProgramFitnessEvaluator(
            engine, "print('anything')", new ScriptProgramEvaluationOptions { RequireEntryPoint = false });
        _ = new ScriptProgramFitnessEvaluator(
            engine, "function score() {}", new ScriptProgramEvaluationOptions { EntryPointMarker = "score" });
    }

    [Fact]
    public void TheEvaluatorValidatesItsArguments()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Printed("{\"quality\": 1}"));

#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(() => new ScriptProgramFitnessEvaluator(null, Script));
        Assert.Throws<ArgumentNullException>(() => new ScriptProgramFitnessEvaluator(engine, null));
#pragma warning restore CS8625
        Assert.Throws<ArgumentException>(() => new ScriptProgramFitnessEvaluator(engine, "   "));
        Assert.Throws<ArgumentException>(() => new ScriptProgramFitnessEvaluator(engine, Script, id: " "));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ScriptProgramFitnessEvaluator(
            engine, Script, new ScriptProgramEvaluationOptions { MaxArtifactCount = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ScriptProgramFitnessEvaluator(
            engine, Script, new ScriptProgramEvaluationOptions { EvaluatorScriptLanguage = (ProgramLanguage)99 }));
    }

    [Fact]
    public void VersionHashTracksTheScriptAndTheOptions()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Printed("{\"quality\": 1}"));

        string first = new ScriptProgramFitnessEvaluator(engine, Script).VersionHash;
        string same = new ScriptProgramFitnessEvaluator(engine, Script).VersionHash;
        string otherScript = new ScriptProgramFitnessEvaluator(engine, Script + "\n# tweak").VersionHash;
        string otherDirection = new ScriptProgramFitnessEvaluator(
            engine,
            Script,
            new ScriptProgramEvaluationOptions { Direction = EvolutionOptimizationDirection.Minimize }).VersionHash;

        Assert.Equal(first, same);
        Assert.NotEqual(first, otherScript);
        Assert.NotEqual(first, otherDirection);
    }

    [Fact]
    public void OptionsAreCopiedDefensively()
    {
        var engine = new ScriptedProgramExecutionEngine(_ => Printed("{\"quality\": 1}"));
        var options = new ScriptProgramEvaluationOptions { MaxArtifactCount = 3 };
        var evaluator = new ScriptProgramFitnessEvaluator(engine, Script, options);

        options.MaxArtifactCount = 9;

        Assert.Equal(3, evaluator.GetOptions().MaxArtifactCount);
        Assert.Equal(ProgramLanguage.Python, evaluator.EvaluatorScriptLanguage);
    }

    [SkippableFact]
    public async Task AnEvaluatorScriptIsSandboxedWithTheSameLimitsAsACandidate()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        ProgramSandboxOptions sandbox =
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.SleepTemplate(30));
        sandbox.Limits.TimeLimitSeconds = 1;

        using var engine = new ProcessProgramExecutionEngine(sandbox);
        var evaluator = new ScriptProgramFitnessEvaluator(engine, "evaluate forever");

        var stopwatch = Stopwatch.StartNew();
        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);
        stopwatch.Stop();

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal("program_script_timeout", Assert.Single(result.Diagnostics).Code);
        Assert.True(
            stopwatch.Elapsed < TimeSpan.FromSeconds(25),
            $"A runaway evaluator script must be killed by the sandbox limit: {stopwatch.Elapsed}.");
    }

    [SkippableFact]
    public async Task TheWholeScriptPathWorksEndToEndThroughTheProcessSandbox()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        // The shell stands in for an interpreter by printing the script verbatim, so the script text is itself the
        // JSON metrics document. That exercises the real sandbox with no language runtime installed.
        const string JsonScript = "{\"quality\": 0.25, \"artifacts\": {\"evaluate\": \"static check\"}}";

        using var engine = new ProcessProgramExecutionEngine(
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.EchoSourceTemplate));
        var evaluator = new ScriptProgramFitnessEvaluator(engine, JsonScript);

        EvolutionTaskResult result = await evaluator.EvaluateAsync(Genome(), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.25, result.Quality);
        Assert.Equal("program_script_artifact", Assert.Single(result.Diagnostics).Code);
    }
}
