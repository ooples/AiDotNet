using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Configuration;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class RunTelemetryTests
{
    [Fact]
    public async Task RealRunnerSeparatesQueuedAndActiveWorkAndReleasesCanceledQueueSlots()
    {
        var options = new ProgramSandboxOptions();
        options.Limits.MaxConcurrentExecutions = 1;
        options.Limits.TimeLimitSeconds = 30;
        options.Limits.MemoryLimitMb = 4096;
        string worker = Path.Combine(AppContext.BaseDirectory, "AiDotNet.CSharp.Worker.dll");
        options.SetInterpreter(ProgramLanguage.CSharp, new ProgramInterpreterSpecification("dotnet", "\"" + worker + "\" --source {source}"));
        using var runner = new ProcessProgramExecutionEngine(options);
        using var activeCancellation = new CancellationTokenSource();
        using var queuedCancellation = new CancellationTokenSource();
        var request = new ProgramExecuteRequest
        {
            Language = ProgramLanguage.CSharp,
            SourceCode = "public static class P { public static void Main() { System.Threading.Thread.Sleep(20000); } }"
        };
        var active = runner.ExecuteAsync(request, activeCancellation.Token);
        try
        {
            Assert.Equal(1, runner.ActiveExecutionCount);
            var queued = runner.ExecuteAsync(request, queuedCancellation.Token);
            Assert.Equal(1, runner.QueuedExecutionCount);
            queuedCancellation.Cancel();
            Assert.False((await queued).Success);
            Assert.Equal(0, runner.QueuedExecutionCount);
            Assert.Equal(1, runner.ActiveExecutionCount);
        }
        finally { activeCancellation.Cancel(); await active; }
        Assert.Equal(0, runner.ActiveExecutionCount);
        Assert.Equal(0, runner.QueuedExecutionCount);
    }

    [Fact]
    public async Task LiveCountersAreDetachedAtFinishAndMissingTokensAreNotReportedAsFree()
    {
        using var cancellation = new CancellationTokenSource();
        var inspection = new RunInspection(new EvolutionRunControl(), cancellation);
        int calls = 0;
        inspection.SetTelemetrySource(() =>
        {
            calls++;
            return new ProgramEvolutionTelemetrySnapshot("private-operator", "private-model",
                new ProgramEvolutionLlmUsage(proposals: 3, chatCalls: 4, providerErrors: 1, outputTokens: 7), 2, 1);
        });
        var first = inspection.Read();
        Assert.Equal(2, first.BackendQueueDepth);
        Assert.Equal(1, first.Runtime!.ActiveExecutions);
        Assert.Equal(4, first.Runtime.ChatCalls);
        Assert.Null(first.Runtime.ReportedInputTokens);
        Assert.Equal(7, first.Runtime.ReportedOutputTokens);
        Assert.DoesNotContain("private-model", first.Runtime.ConfiguredModelIdentity);
        Assert.DoesNotContain("private-operator", first.Runtime.OperatorIdentity);
        await inspection.FinishAsync(null);
        int finalCalls = calls;
        var final = inspection.Read();
        Assert.Equal(final.Runtime, inspection.Read().Runtime);
        Assert.Equal(finalCalls, calls);
        Assert.True(final.Finished);
    }

    [Fact]
    public void TelemetryErrorsDoNotSuppressFatalInnerExceptions()
    {
        using var cancellation = new CancellationTokenSource();
        var ordinary = new RunInspection(new EvolutionRunControl(), cancellation);
        ordinary.SetTelemetrySource(() => throw new IOException("private path"));
        Assert.Null(ordinary.Read().Runtime);
        var fatal = new RunInspection(new EvolutionRunControl(), cancellation);
        fatal.SetTelemetrySource(() => throw new InvalidOperationException("outer", new OutOfMemoryException("synthetic")));
        Assert.Throws<InvalidOperationException>(() => fatal.Read());
    }

    [Fact]
    public async Task SourceRegistrationIsOneTimeAndBeforeEngineEvents()
    {
        using var cancellation = new CancellationTokenSource();
        var inspection = new RunInspection(new EvolutionRunControl(), cancellation);
        inspection.SetTelemetrySource(() => new ProgramEvolutionTelemetrySnapshot("op", null, new()));
        Assert.Throws<InvalidOperationException>(() => inspection.SetTelemetrySource(() => new("other", null, new())));
        var late = new RunInspection(new EvolutionRunControl(), cancellation);
        await late.OnEventAsync(new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Stopped, 1));
        Assert.Throws<InvalidOperationException>(() => late.SetTelemetrySource(() => new("op", null, new())));
    }
}
