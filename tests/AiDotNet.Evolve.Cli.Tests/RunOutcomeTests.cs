using AiDotNet.Evolution;
using AiDotNet.Models.Results;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class RunOutcomeTests
{
    [Theory]
    [InlineData(EvolutionStopReason.EvaluationBudgetReached, 1, 0)]
    [InlineData(EvolutionStopReason.EvaluationBudgetReached, 0, 3)]
    [InlineData(EvolutionStopReason.TimeLimitReached, 0, 3)]
    [InlineData(EvolutionStopReason.NoCandidates, 1, 3)]
    [InlineData(EvolutionStopReason.CandidateFailure, 1, 3)]
    [InlineData(EvolutionStopReason.Canceled, 0, 0)]
    [InlineData(EvolutionStopReason.Canceled, 1, 0)]
    public void AReturnedSummaryIsNotAlwaysASuccess(EvolutionStopReason reason, int archiveCount, int expected) =>
        Assert.Equal(expected, EvolveCommandLine.RunExitCode(new EvolutionRunSummary { StopReason = reason, ArchiveCount = archiveCount }));

    [Theory]
    [InlineData("run")]
    [InlineData("preflight")]
    public async Task AlreadyCanceledCommandDoesNotLoadConfigurationOrCreateServices(string command)
    {
        using var source = new CancellationTokenSource();
        source.Cancel();
        using var error = new StringWriter();
        int code = await EvolveCommandLine.ExecuteAsync(new[] { command, "--config", "this-file-does-not-exist.yaml" },
            TextWriter.Null, error, source.Token);
        Assert.Equal(EvolveCommandLine.ExitCancelled, code);
        Assert.DoesNotContain("this-file-does-not-exist", error.ToString());
    }
}
