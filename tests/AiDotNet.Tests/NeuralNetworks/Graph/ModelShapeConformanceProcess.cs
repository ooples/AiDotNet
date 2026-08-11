using System.Diagnostics;
using System.Text.Json;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

internal static class ModelShapeConformanceProcess
{
    internal sealed record Measurement(
        string Status,
        int[]? InputShape,
        int[]? PredictedShape,
        int[]? ActualShape,
        string? Error);

    public static async Task<Measurement> ProbeAsync(
        Type modelType,
        int extent,
        int classes,
        TimeSpan timeout)
    {
        await Task.Yield();

#if NET10_0_OR_GREATER
        string? worker = FindWorker();
        if (worker is null)
            return new Measurement("worker-missing", null, null, null, null);

        var start = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        start.ArgumentList.Add(worker);
        start.ArgumentList.Add("shape");
        start.ArgumentList.Add(modelType.AssemblyQualifiedName ?? modelType.FullName ?? modelType.Name);
        start.ArgumentList.Add(extent.ToString(System.Globalization.CultureInfo.InvariantCulture));
        start.ArgumentList.Add(classes.ToString(System.Globalization.CultureInfo.InvariantCulture));

        // A model gets a reclaimable process, a bounded managed heap, and a deadline. A pathological
        // constructor or forward can fail its own result but cannot kill the xUnit runner and erase
        // the other models' evidence.
        start.Environment["DOTNET_GCHeapHardLimit"] = "0x40000000";
        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("Could not start the shape conformance worker.");

        Task<string> stdoutTask = process.StandardOutput.ReadToEndAsync();
        Task<string> stderrTask = process.StandardError.ReadToEndAsync();
        using var timeoutCts = new CancellationTokenSource(timeout);
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token);
        }
        catch (OperationCanceledException)
        {
            try
            {
                if (!process.HasExited) process.Kill(entireProcessTree: true);
                await process.WaitForExitAsync();
                await Task.WhenAll(stdoutTask, stderrTask);
            }
            catch
            {
                // The timeout status remains the authoritative result even if an already-dying
                // process races cleanup. Disposal below still releases the local process handle.
            }
            return new Measurement("timeout", null, null, null, null);
        }

        string stdout = await stdoutTask;
        string stderr = await stderrTask;
        string? json = stdout.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
        if (json is null)
        {
            string error = string.IsNullOrWhiteSpace(stderr) ? $"exit {process.ExitCode}" : stderr.Trim();
            return new Measurement("crashed", null, null, null, error);
        }

        try
        {
            return JsonSerializer.Deserialize<Measurement>(json,
                       new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                   ?? new Measurement("invalid-result", null, null, null, json);
        }
        catch (JsonException)
        {
            return new Measurement("invalid-result", null, null, null, json);
        }
#else
        return new Measurement("worker-unavailable", null, null, null,
            "The isolated shape sweep is supported on net10.0 and later.");
#endif
    }

#if NET10_0_OR_GREATER
    private static string? FindWorker()
    {
        var frameworkDirectory = new DirectoryInfo(AppContext.BaseDirectory.TrimEnd(
            Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        string configuration = frameworkDirectory.Parent?.Name ?? "Debug";
        var testsDirectory = frameworkDirectory.Parent?.Parent?.Parent?.Parent;
        if (testsDirectory is null) return null;
        string candidate = Path.Combine(testsDirectory.FullName, "AiDotNet.ParameterSweepWorker",
            "bin", configuration, "net10.0", "AiDotNet.ParameterSweepWorker.dll");
        return File.Exists(candidate) ? candidate : null;
    }
#endif
}
