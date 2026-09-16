using System.Diagnostics;
using AttributionRuntime;
using Xunit;

namespace PrototypeTests;

[Boundary]
public sealed class WorkerTests
{
    [Fact, Trait("Scenario", "Positive")]
    public Task Complete() => RunWorker("complete");

    [Fact, Trait("Scenario", "MissingWorker")]
    public Task Missing() => RunWorker("missing");

    [Fact, Trait("Scenario", "UnclosedWorker")]
    public Task Unclosed() => RunWorker("unclosed");

    [Fact, Trait("Scenario", "UnjoinedWorker")]
    public Task Unjoined() => RunWorker("complete", join: false);

    private static async Task RunWorker(string mode, bool join = true)
    {
        string worker = Environment.GetEnvironmentVariable("ATTRIBUTION_WORKER_DLL")
            ?? throw new InvalidOperationException("Missing worker fixture.");
        var start = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true
        };
        start.ArgumentList.Add(worker);
        start.ArgumentList.Add(mode);
        WorkerTicket? ticket = Environment.GetEnvironmentVariable("ATTRIBUTION_OUTPUT") is not null
            ? Tracker.AttachWorker(start) : null;
        using var process = Process.Start(start) ?? throw new InvalidOperationException("Worker did not start.");
        Task<string> output = process.StandardOutput.ReadToEndAsync();
        Task<string> error = process.StandardError.ReadToEndAsync();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        try { await process.WaitForExitAsync(deadline.Token); }
        catch (OperationCanceledException)
        {
            process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync();
            throw;
        }
        Assert.Equal(0, process.ExitCode);
        if (ticket is not null && join) Tracker.CompleteWorker(ticket, process.ExitCode);
        Assert.Equal(mode == "complete" ? "42" : "", (await output).Trim());
        Assert.Equal("", await error);
    }
}
