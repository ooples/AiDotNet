using System.Diagnostics;
using Xunit;

public sealed class WorkerCoverageTests
{
    [Fact]
    public async Task WorkerExecutesCodeNeverCalledByParent()
    {
        await Task.WhenAll(RunWorkerAsync(21, 42), RunWorkerAsync(-1, 1), RunWorkerAsync(0, 0));
    }

    private static async Task RunWorkerAsync(int value, int expected)
    {
        string worker = Environment.GetEnvironmentVariable("WORKER_COVERAGE_FIXTURE_DLL")
            ?? throw new InvalidOperationException("Missing fixture worker path.");
        var start = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add(worker);
        start.ArgumentList.Add(value.ToString(System.Globalization.CultureInfo.InvariantCulture));
        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("Worker failed to start.");
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
        Assert.Equal(expected.ToString(System.Globalization.CultureInfo.InvariantCulture), (await output).Trim());
        Assert.Equal(string.Empty, await error);
    }
}
