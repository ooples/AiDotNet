using System.Diagnostics;
using System.Text.Json;

namespace AiDotNet.Tests.IntegrationTests;

internal static class ParameterSweepProcess
{
    internal sealed record Measurement(
        string Status,
        long Declared,
        long Flat,
        long ChunkSum,
        int ChunkCount,
        string Readiness,
        string? Error);

    internal sealed record ModelMeasurement(Type ModelType, Measurement Measurement);

    public static async Task<IReadOnlyList<ModelMeasurement>> MeasureAllAsync(
        IReadOnlyList<Type> modelTypes,
        bool includeChunks,
        long maximum,
        TimeSpan timeout)
    {
        await Task.Yield();

        // Three isolated workers keep GitHub's cores busy while bounding the worst-case heap at
        // three worker limits. Results retain discovery order so reports stay deterministic.
        int concurrency = Math.Max(1, Math.Min(3, Environment.ProcessorCount));
        using var gate = new SemaphoreSlim(concurrency, concurrency);
        var results = new ModelMeasurement[modelTypes.Count];
        var tasks = new Task[modelTypes.Count];

        for (int i = 0; i < modelTypes.Count; i++)
        {
            int index = i;
            tasks[index] = MeasureOneAsync(index);
        }

        await Task.WhenAll(tasks);
        return results;

        async Task MeasureOneAsync(int index)
        {
            await gate.WaitAsync();
            try
            {
                var type = modelTypes[index];
                results[index] = new ModelMeasurement(
                    type,
                    await MeasureAsync(type, includeChunks, maximum, timeout));
            }
            finally
            {
                gate.Release();
            }
        }
    }

    public static async Task<Measurement> MeasureAsync(
        Type modelType,
        bool includeChunks,
        long maximum,
        TimeSpan timeout)
    {
        await Task.Yield();

#if NET10_0_OR_GREATER
        string? worker = FindWorker();
        if (worker is null)
            return new Measurement("worker-missing", -1, -1, -1, 0, "Unknown", null);

        var start = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        start.ArgumentList.Add(worker);
        start.ArgumentList.Add(modelType.AssemblyQualifiedName ?? modelType.FullName ?? modelType.Name);
        start.ArgumentList.Add(includeChunks.ToString());
        start.ArgumentList.Add(maximum.ToString(System.Globalization.CultureInfo.InvariantCulture));

        // One pathological constructor gets a bounded heap and its own process. If it exhausts
        // either resource, only this measurement dies; the shard keeps its accumulated report.
        start.Environment["DOTNET_GCHeapHardLimit"] = "0x40000000";
        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("Could not start the parameter sweep worker.");

        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();
        using var timeoutCts = new CancellationTokenSource(timeout);
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token);
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch { }
            return new Measurement("timeout", -1, -1, -1, 0, "Unknown", null);
        }

        string stdout = await stdoutTask;
        string stderr = await stderrTask;
        string? json = stdout.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
        if (json is null)
            return new Measurement("crashed", -1, -1, -1, 0, "Unknown",
                string.IsNullOrWhiteSpace(stderr) ? $"exit {process.ExitCode}" : stderr.Trim());

        try
        {
            return JsonSerializer.Deserialize<Measurement>(json,
                       new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                   ?? new Measurement("invalid-result", -1, -1, -1, 0, "Unknown", json);
        }
        catch (JsonException)
        {
            return new Measurement("invalid-result", -1, -1, -1, 0, "Unknown", json);
        }
#else
        return new Measurement("worker-unavailable", -1, -1, -1, 0, "Unknown", null);
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
