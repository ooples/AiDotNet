using System.Diagnostics;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Exercises the out-of-process sandbox against commands present on a stock machine. No test here needs Docker, a
/// network, a language runtime, or a real model: the operating-system shell plays the part of the interpreter.
/// </summary>
public sealed class ProcessProgramExecutionEngineTests
{
    private static ProgramExecuteRequest Request(string source, string? stdIn = null, bool compileOnly = false) =>
        new()
        {
            Language = ProgramLanguage.Python,
            SourceCode = source,
            StdIn = stdIn,
            CompileOnly = compileOnly
        };

    [SkippableFact]
    public async Task RunsTheConfiguredCommandAndCapturesStandardOutput()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        using var engine = new ProcessProgramExecutionEngine(
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.EchoSourceTemplate));

        ProgramExecuteResponse response = await engine.ExecuteAsync(Request("hello sandbox"));

        Assert.True(response.Success, response.Error ?? "no error reported");
        Assert.Equal(0, response.ExitCode);
        Assert.Equal(ProgramLanguage.Python, response.Language);
        Assert.Contains("hello sandbox", response.StdOut, StringComparison.Ordinal);
        Assert.False(response.StdOutTruncated);
        Assert.Null(response.ErrorCode);
    }

    [SkippableFact]
    public async Task StandardOutputIsCappedAndReportedAsTruncated()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        ProgramSandboxOptions options =
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.EchoSourceTemplate);
        options.Limits.MaxStdOutChars = 64;

        using var engine = new ProcessProgramExecutionEngine(options);
        ProgramExecuteResponse response = await engine.ExecuteAsync(Request(new string('x', 20_000)));

        Assert.True(response.Success, response.Error ?? "no error reported");
        Assert.Equal(64, response.StdOut.Length);
        Assert.True(response.StdOutTruncated, "Output beyond the cap must be reported as truncated.");
    }

    [SkippableFact]
    public async Task StandardInputIsForwardedToTheChildProcess()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        using var engine = new ProcessProgramExecutionEngine(
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.EchoStdInTemplate));

        ProgramExecuteResponse response = await engine.ExecuteAsync(Request("ignored", "forwarded-input\n"));

        Assert.True(response.Success, response.Error ?? "no error reported");
        Assert.Contains("forwarded-input", response.StdOut, StringComparison.Ordinal);
    }

    [SkippableFact]
    public async Task HostEnvironmentVariablesAreNotVisibleToTheChild()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        const string SecretName = "AIDOTNET_SANDBOX_SECRET_PROBE";
        const string SecretValue = "s3cr3t-value-that-must-not-leak";
        Environment.SetEnvironmentVariable(SecretName, SecretValue);
        try
        {
            using var engine = new ProcessProgramExecutionEngine(
                ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.PrintEnvironmentTemplate));

            ProgramExecuteResponse response = await engine.ExecuteAsync(Request("ignored"));

            Assert.True(response.Success, response.Error ?? "no error reported");
            Assert.Contains("PATH", response.StdOut, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(SecretName, response.StdOut, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(SecretValue, response.StdOut, StringComparison.Ordinal);
        }
        finally
        {
            Environment.SetEnvironmentVariable(SecretName, null);
        }
    }

    [SkippableFact]
    public async Task NonZeroExitIsReportedAsExecutionFailedRatherThanThrown()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        using var engine = new ProcessProgramExecutionEngine(
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.ExitWithCodeTemplate(7)));

        ProgramExecuteResponse response = await engine.ExecuteAsync(Request("ignored"));

        Assert.False(response.Success);
        Assert.Equal(7, response.ExitCode);
        Assert.Equal(ProgramExecuteErrorCode.ExecutionFailed, response.ErrorCode);
    }

    [SkippableFact]
    public async Task TimeoutTerminatesTheWholeProcessTree()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        string scratch = ProgramSandboxTestEnvironment.CreateScratchDirectory();
        try
        {
            string marker = Path.Combine(scratch, "marker.txt");
            string template = WriteDetachedGrandchild(scratch, marker);

            ProgramSandboxOptions options = ProgramSandboxTestEnvironment.Options(template);
            options.Limits.TimeLimitSeconds = 1;

            using var engine = new ProcessProgramExecutionEngine(options);
            var stopwatch = Stopwatch.StartNew();
            ProgramExecuteResponse response = await engine.ExecuteAsync(Request("ignored"));
            stopwatch.Stop();

            Assert.False(response.Success);
            Assert.Equal(ProgramExecuteErrorCode.TimeoutOrCanceled, response.ErrorCode);
            Assert.True(
                stopwatch.Elapsed < TimeSpan.FromSeconds(20),
                $"The timeout did not interrupt the run: {stopwatch.Elapsed}.");

            // The grandchild writes "start" immediately and "end" after a long sleep. Waiting well past the point
            // where it would have written "end" and still not seeing it is what proves the whole tree was killed;
            // seeing "start" is what proves the grandchild really ran, so the assertion cannot pass vacuously.
            await Task.Delay(TimeSpan.FromSeconds(4));
            string observed = File.Exists(marker) ? File.ReadAllText(marker) : string.Empty;

            Skip.IfNot(
                observed.IndexOf("start", StringComparison.Ordinal) >= 0,
                "The detached grandchild never started, so tree termination cannot be observed here.");
            Assert.DoesNotContain("end", observed, StringComparison.Ordinal);
        }
        finally
        {
            ProgramSandboxTestEnvironment.TryDelete(scratch);
        }
    }

    [SkippableFact]
    public async Task ConcurrencyLimitSerializesExecutions()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        ProgramSandboxOptions options =
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.SleepTemplate(1));
        options.Limits.MaxConcurrentExecutions = 1;
        options.Limits.TimeLimitSeconds = 30;

        using var engine = new ProcessProgramExecutionEngine(options);

        var stopwatch = Stopwatch.StartNew();
        Task<ProgramExecuteResponse>[] runs = new[]
        {
            engine.ExecuteAsync(Request("a")),
            engine.ExecuteAsync(Request("b")),
            engine.ExecuteAsync(Request("c"))
        };
        ProgramExecuteResponse[] responses = await Task.WhenAll(runs);
        stopwatch.Stop();

        foreach (ProgramExecuteResponse response in responses)
        {
            Assert.True(response.Success, response.Error ?? "no error reported");
        }

        // Three one-second sleeps behind a single permit cannot finish in under two seconds; running them in
        // parallel would take about one. Only the lower bound is asserted, so a loaded machine cannot fail this.
        Assert.True(
            stopwatch.Elapsed >= TimeSpan.FromMilliseconds(2000),
            $"Executions overlapped despite a concurrency limit of one: {stopwatch.Elapsed}.");
    }

    [SkippableFact]
    public async Task TheWorkspaceIsDeletedAfterSuccessAndAfterATimeout()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        string root = ProgramSandboxTestEnvironment.CreateScratchDirectory();
        try
        {
            ProgramSandboxOptions options =
                ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.EchoSourceTemplate);
            options.WorkingDirectory = root;

            using (var engine = new ProcessProgramExecutionEngine(options))
            {
                ProgramExecuteResponse response = await engine.ExecuteAsync(Request("hello"));
                Assert.True(response.Success, response.Error ?? "no error reported");
            }

            Assert.Empty(Directory.GetDirectories(root));

            ProgramSandboxOptions timingOut =
                ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.SleepTemplate(20));
            timingOut.WorkingDirectory = root;
            timingOut.Limits.TimeLimitSeconds = 1;

            using (var engine = new ProcessProgramExecutionEngine(timingOut))
            {
                ProgramExecuteResponse response = await engine.ExecuteAsync(Request("hello"));
                Assert.Equal(ProgramExecuteErrorCode.TimeoutOrCanceled, response.ErrorCode);
            }

            Assert.Empty(Directory.GetDirectories(root));
        }
        finally
        {
            ProgramSandboxTestEnvironment.TryDelete(root);
        }
    }

    [SkippableFact]
    public async Task CompileOnlyUsesTheCompileCommandAndReportsCompilationFailure()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        using var engine = new ProcessProgramExecutionEngine(
            ProgramSandboxTestEnvironment.Options(
                ProgramSandboxTestEnvironment.EchoSourceTemplate,
                ProgramSandboxTestEnvironment.ExitWithCodeTemplate(3)));

        ProgramExecuteResponse response = await engine.ExecuteAsync(Request("ignored", compileOnly: true));

        Assert.False(response.Success);
        Assert.True(response.CompilationAttempted);
        Assert.False(response.CompilationSucceeded);
        Assert.Equal(3, response.ExitCode);
        Assert.Equal(ProgramExecuteErrorCode.CompilationFailed, response.ErrorCode);
        Assert.DoesNotContain("ignored", response.StdOut, StringComparison.Ordinal);
    }

    [SkippableFact]
    public async Task CompileOnlyIsRefusedRatherThanExecutedWhenNoCompileCommandExists()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        using var engine = new ProcessProgramExecutionEngine(
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.EchoSourceTemplate));

        ProgramExecuteResponse response = await engine.ExecuteAsync(Request("must-not-run", compileOnly: true));

        Assert.False(response.Success);
        Assert.Equal(ProgramExecuteErrorCode.InvalidRequest, response.ErrorCode);
        Assert.Empty(response.StdOut);
    }

    [SkippableFact]
    public async Task SynchronousTryExecuteStillWorks()
    {
        Skip.IfNot(ProgramSandboxTestEnvironment.ShellExists, "No shell available on this machine.");

        using var engine = new ProcessProgramExecutionEngine(
            ProgramSandboxTestEnvironment.Options(ProgramSandboxTestEnvironment.EchoStdInTemplate));

        bool ran = await Task.Run(() => engine.TryExecute(
            ProgramLanguage.Python,
            "ignored",
            "sync-path\n",
            out string output,
            out string? error,
            CancellationToken.None) && output.IndexOf("sync-path", StringComparison.Ordinal) >= 0 && error is null);

        Assert.True(ran);
    }

    [Fact]
    public async Task SqlIsRoutedAwayFromTheProgramSandbox()
    {
        using var engine = new ProcessProgramExecutionEngine(new ProgramSandboxOptions());

        ProgramExecuteResponse response = await engine.ExecuteAsync(new ProgramExecuteRequest
        {
            Language = ProgramLanguage.SQL,
            SourceCode = "SELECT 1"
        });

        Assert.False(response.Success);
        Assert.Equal(ProgramExecuteErrorCode.SqlNotSupported, response.ErrorCode);
    }

    [Fact]
    public async Task ALanguageWithoutAConfiguredInterpreterIsRejected()
    {
        using var engine = new ProcessProgramExecutionEngine(new ProgramSandboxOptions());

        ProgramExecuteResponse response = await engine.ExecuteAsync(new ProgramExecuteRequest
        {
            Language = ProgramLanguage.Rust,
            SourceCode = "fn main() {}"
        });

        Assert.False(response.Success);
        Assert.Equal(ProgramExecuteErrorCode.InvalidRequest, response.ErrorCode);
        Assert.Equal(ProgramLanguage.Rust, response.Language);
    }

    [Fact]
    public async Task AnUndetectableGenericProgramIsRejected()
    {
        using var engine = new ProcessProgramExecutionEngine(new ProgramSandboxOptions());

        ProgramExecuteResponse response = await engine.ExecuteAsync(new ProgramExecuteRequest
        {
            Language = ProgramLanguage.Generic,
            SourceCode = "12345"
        });

        Assert.False(response.Success);
        Assert.Equal(ProgramExecuteErrorCode.LanguageNotDetected, response.ErrorCode);
    }

    [Fact]
    public async Task ALanguageOutsideTheAllowedListIsRejected()
    {
        using var engine = new ProcessProgramExecutionEngine(new ProgramSandboxOptions());

        ProgramExecuteResponse response = await engine.ExecuteAsync(new ProgramExecuteRequest
        {
            Language = ProgramLanguage.Python,
            SourceCode = "print(1)",
            AllowedLanguages = new List<ProgramLanguage> { ProgramLanguage.JavaScript }
        });

        Assert.False(response.Success);
        Assert.Equal(ProgramExecuteErrorCode.InvalidRequest, response.ErrorCode);
    }

    [Fact]
    public async Task OversizedSourceAndInputAreRejectedBeforeAnyProcessStarts()
    {
        var options = new ProgramSandboxOptions();
        options.Limits.MaxSourceCodeChars = 10;
        options.Limits.MaxStdInChars = 4;

        using var engine = new ProcessProgramExecutionEngine(options);

        ProgramExecuteResponse tooLarge = await engine.ExecuteAsync(Request(new string('p', 11)));
        Assert.Equal(ProgramExecuteErrorCode.SourceCodeTooLarge, tooLarge.ErrorCode);

        ProgramExecuteResponse inputTooLarge = await engine.ExecuteAsync(Request("print(1)", "abcdef"));
        Assert.Equal(ProgramExecuteErrorCode.StdInTooLarge, inputTooLarge.ErrorCode);

        ProgramExecuteResponse empty = await engine.ExecuteAsync(Request("   "));
        Assert.Equal(ProgramExecuteErrorCode.SourceCodeRequired, empty.ErrorCode);
    }

    [Fact]
    public async Task AMissingInterpreterIsReportedRatherThanThrown()
    {
        var options = new ProgramSandboxOptions();
        options.SetInterpreter(
            ProgramLanguage.Python,
            new ProgramInterpreterSpecification(
                "aidotnet-interpreter-that-does-not-exist",
                ProgramInterpreterSpecification.SourcePlaceholder));

        using var engine = new ProcessProgramExecutionEngine(options);
        ProgramExecuteResponse response = await engine.ExecuteAsync(Request("print(1)"));

        Assert.False(response.Success);
        Assert.Equal(ProgramExecuteErrorCode.ExecutionFailed, response.ErrorCode);
    }

    [Fact]
    public async Task CancellationBeforeStartIsReportedAsCanceled()
    {
        using var engine = new ProcessProgramExecutionEngine(new ProgramSandboxOptions());
        using var source = new CancellationTokenSource();
        source.Cancel();

        ProgramExecuteResponse response = await engine.ExecuteAsync(Request("print(1)"), source.Token);

        Assert.Equal(ProgramExecuteErrorCode.TimeoutOrCanceled, response.ErrorCode);
    }

    [Fact]
    public void TheEngineRefusesModesItDoesNotImplement()
    {
        var serving = new ProgramSandboxOptions { Mode = ProgramSandboxMode.Serving };
        Assert.Throws<ArgumentException>(() => new ProcessProgramExecutionEngine(serving));

        var unsafeMode = new ProgramSandboxOptions
        {
            Mode = ProgramSandboxMode.InProcessUnsafe,
            AllowUnsafeInProcessExecution = true
        };
        Assert.Throws<ArgumentException>(() => new ProcessProgramExecutionEngine(unsafeMode));
    }

    [Fact]
    public async Task TheEngineValidatesItsArguments()
    {
#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(() => new ProcessProgramExecutionEngine(null));
#pragma warning restore CS8625

        using var engine = new ProcessProgramExecutionEngine(new ProgramSandboxOptions());
#pragma warning disable CS8625
        await Assert.ThrowsAsync<ArgumentNullException>(() => engine.ExecuteAsync(null));
#pragma warning restore CS8625
    }

    [Fact]
    public void MemoryLimitEnforcementIsReportedHonestly()
    {
        using var engine = new ProcessProgramExecutionEngine(new ProgramSandboxOptions());

        // Windows always has job objects; elsewhere the cap needs a POSIX shell. The engine must never claim a
        // protection it cannot apply, so this mirrors the two mechanisms rather than asserting a constant.
        bool expected = ProgramSandboxTestEnvironment.IsWindows || File.Exists("/bin/sh");
        Assert.Equal(expected, engine.CanEnforceMemoryLimit);
    }

    [Fact]
    public void OptionsAreCopiedSoLaterMutationCannotChangeARunningEngine()
    {
        var options = new ProgramSandboxOptions();
        options.Limits.TimeLimitSeconds = 9;

        using var engine = new ProcessProgramExecutionEngine(options);
        options.Limits.TimeLimitSeconds = 1;

        Assert.Equal(9, engine.GetOptions().Limits.TimeLimitSeconds);
    }

    private static string WriteDetachedGrandchild(string scratch, string marker)
    {
        if (ProgramSandboxTestEnvironment.IsWindows)
        {
            string batch = Path.Combine(scratch, "grandchild.cmd");
            File.WriteAllText(
                batch,
                "@echo off\r\n" +
                ">>\"" + marker + "\" echo start\r\n" +
                "ping -n 21 127.0.0.1 >nul\r\n" +
                ">>\"" + marker + "\" echo end\r\n");

            return "/c type {source} >nul & start \"\" /b \"" + batch + "\" & ping -n 21 127.0.0.1 >nul";
        }

        string script = Path.Combine(scratch, "grandchild.sh");
        File.WriteAllText(
            script,
            "#!/bin/sh\n" +
            "echo start >> \"$1\"\n" +
            "sleep 20\n" +
            "echo end >> \"$1\"\n");

        return "-c \"cat {source} > /dev/null; /bin/sh '" + script + "' '" + marker + "' & sleep 20\"";
    }
}
