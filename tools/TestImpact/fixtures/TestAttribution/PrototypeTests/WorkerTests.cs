using System.Diagnostics;
using AttributionRuntime;
using Xunit;

namespace PrototypeTests;

public sealed class WorkerTests
{
    [Fact, Trait("Scenario", "Positive")]
    public Task Complete() => RunWorker(WorkerMode.Complete);

    [Fact, Trait("Scenario", "MissingWorker")]
    public Task Missing() => RunWorker(WorkerMode.Missing);

    [Fact, Trait("Scenario", "UnclosedWorker")]
    public Task Unclosed() => RunWorker(WorkerMode.Unclosed);

    [Fact, Trait("Scenario", "UnjoinedWorker")]
    public Task Unjoined() => RunWorker(WorkerMode.Complete, join: false);

    [Fact, Trait("Scenario", "WorkerReplay")]
    public Task ReplayStart() => RunWorker(WorkerMode.Complete, replay: true);

    [Fact, Trait("Scenario", "WorkerNeverStarted")]
    public void CompleteWithoutStarting()
    {
        var start = new ProcessStartInfo("dotnet") { UseShellExecute = false };
        WorkerTicket ticket = Tracker.AttachWorker(start);
        Assert.Throws<InvalidOperationException>(() => Tracker.CompleteWorker(ticket, 0));
    }

    [Fact, Trait("Scenario", "KilledWorker")]
    public async Task KilledWorker()
    {
        string worker = Environment.GetEnvironmentVariable("ATTRIBUTION_WORKER_DLL")
            ?? throw new InvalidOperationException("Missing worker fixture.");
        var start = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true
        };
        start.ArgumentList.Add(worker);
        start.ArgumentList.Add("WaitForKill");
        Tracker.AttachWorker(start);
        using var child = Process.Start(start) ?? throw new InvalidOperationException("Worker did not start.");
        try
        {
            Assert.Equal("ready", await child.StandardOutput.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(15)));
        }
        finally
        {
            if (!child.HasExited) child.Kill(entireProcessTree: true);
            await child.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(15));
        }
        Assert.NotEqual(0, child.ExitCode);
    }

    private static async Task RunWorker(WorkerMode mode, bool join = true, bool replay = false)
    {
        string worker = Environment.GetEnvironmentVariable("ATTRIBUTION_WORKER_DLL")
            ?? throw new InvalidOperationException("Missing worker fixture.");
        var start = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true
        };
        start.ArgumentList.Add(worker);
        start.ArgumentList.Add(mode.ToString());
        WorkerTicket? ticket = Environment.GetEnvironmentVariable("ATTRIBUTION_OUTPUT") is not null
            ? Tracker.AttachWorker(start) : null;
        using var process = Process.Start(start) ?? throw new InvalidOperationException("Worker did not start.");
        Task<string> output = process.StandardOutput.ReadToEndAsync();
        Task<string> error = process.StandardError.ReadToEndAsync();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        try { await process.WaitForExitAsync(deadline.Token); }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); }
            catch (InvalidOperationException) when (process.HasExited) { }
            await process.WaitForExitAsync();
            throw;
        }
        Assert.Equal(0, process.ExitCode);
        if (replay)
        {
            // The second process deliberately produces no report. A matching
            // legitimate report must not conceal reuse of its start ticket.
            start.ArgumentList[1] = WorkerMode.Missing.ToString();
            using var second = Process.Start(start) ?? throw new InvalidOperationException("Replay did not start.");
            try { await second.WaitForExitAsync(deadline.Token); }
            catch (OperationCanceledException)
            {
                try { second.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) when (second.HasExited) { }
                await second.WaitForExitAsync();
                throw;
            }
            Assert.Equal(0, second.ExitCode);
        }
        if (ticket is not null && join) Tracker.CompleteWorker(ticket, process.ExitCode);
        Assert.Equal(mode == WorkerMode.Complete ? "42" : "", (await output).Trim());
        Assert.Equal("", await error);
    }

    private enum WorkerMode { Complete, Missing, Unclosed }
}
