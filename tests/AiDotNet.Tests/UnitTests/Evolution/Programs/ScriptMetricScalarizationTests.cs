using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Metrics;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Proves an evaluator script written for the reference implementation scores here, instead of failing because it
/// reports a metric dictionary rather than a single quality.
/// </summary>
public sealed class ScriptMetricScalarizationTests
{
    [Fact]
    public async Task AnUpstreamScriptReportingCombinedScoreIsScoredByIt()
    {
        // The reference implementation prefers a metric literally named combined_score over every other metric,
        // however many there are.
        EvolutionTaskResult result = await EvaluateAsync(
            "{\"combined_score\": 0.75, \"speed\": 1200.0, \"accuracy\": 0.1}");

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.75, result.Quality);
    }

    [Fact]
    public async Task AnUpstreamScriptWithoutCombinedScoreIsScoredByTheMeanOfItsNumerics()
    {
        EvolutionTaskResult result = await EvaluateAsync("{\"accuracy\": 0.8, \"coverage\": 0.6}");

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.7, Assert.IsType<double>(result.Quality), 10);
    }

    [Fact]
    public async Task AnExplicitQualityStillWinsOverTheMetricDictionary()
    {
        // Scripts written for this library are never second-guessed by the fallback.
        EvolutionTaskResult result = await EvaluateAsync("{\"quality\": 0.25, \"accuracy\": 0.9}");

        Assert.Equal(0.25, result.Quality);
    }

    [Fact]
    public async Task AMetricThatCannotBeCombinedIsReportedRatherThanSilentlySkipped()
    {
        // A silently dropped metric produces a score nobody can explain afterwards.
        EvolutionTaskResult result = await EvaluateAsync("{\"accuracy\": 0.8, \"notes\": \"ran clean\"}");

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(0.8, result.Quality);
        Assert.Contains(
            result.Diagnostics,
            d => d.Code == "metric_not_combined" && d.Message.Contains("notes", StringComparison.Ordinal));
    }

    [Fact]
    public async Task StructuralPropertiesAreNeverMistakenForMetrics()
    {
        // "descriptors" describes where the candidate sits in the archive; averaging it into the score would be
        // silently wrong rather than loudly wrong.
        EvolutionTaskResult result = await EvaluateAsync(
            "{\"accuracy\": 0.5, \"descriptors\": {\"length\": 900.0}}");

        Assert.Equal(0.5, result.Quality);
        Assert.Equal(900.0, result.Descriptors["length"]);
    }

    [Fact]
    public async Task WithoutAnAggregatorAMissingQualityStillFails()
    {
        EvolutionTaskResult result = await EvaluateAsync("{\"accuracy\": 0.8}", withAggregator: false);

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
    }

    private static async Task<EvolutionTaskResult> EvaluateAsync(string printed, bool withAggregator = true)
    {
        var evaluator = new ScriptProgramFitnessEvaluator(
            new PrintingExecutionEngine(printed),
            "# evaluate\nprint(payload)\n",
            new ScriptProgramEvaluationOptions { RequireEntryPoint = false },
            "program-script-evaluator",
            withAggregator ? new ProgramMetricAggregator() : null);

        var genome = new ProgramGenome("def solve(x):\n    return x\n", ProgramLanguage.Python);
        var context = new EvolutionEvaluationContext(1, 7UL, 1UL, 1);
        return await evaluator.EvaluateAsync(genome, context);
    }

    /// <summary>An execution engine that returns a fixed standard output, so no interpreter is needed.</summary>
    private sealed class PrintingExecutionEngine : IProgramExecutionEngine
    {
        private readonly string _stdOut;

        public PrintingExecutionEngine(string stdOut) => _stdOut = stdOut;

        public bool TryExecute(
            ProgramLanguage language,
            string sourceCode,
            string input,
            out string output,
            out string? errorMessage,
            CancellationToken cancellationToken = default)
        {
            output = _stdOut;
            errorMessage = null;
            return true;
        }

        public Task<ProgramExecuteResponse> ExecuteAsync(
            ProgramExecuteRequest request,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(new ProgramExecuteResponse
            {
                Success = true,
                Language = ProgramLanguage.Python,
                ExitCode = 0,
                StdOut = _stdOut
            });
    }
}
