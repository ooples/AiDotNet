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

    /// <summary>One construction profile for <see cref="ObserveAsync"/>; mirrors the worker's record.</summary>
    internal sealed record ObservationProfile(
        string Name,
        string InputType,
        int InputSize,
        int InputDepth,
        int Classes,
        int[] Batches,
        int AxisCap,
        int AltExtent,
        bool UseDefaultConstructor,
        bool FallBackToDefaultConstructor,
        bool OverrideClassCountParameters,
        bool StopAtFirstFailure);

    internal sealed record ShapeObservation(int[] Input, int[]? Output, string? Failure);

    internal sealed record ProfileObservation(
        string? Failure,
        int[]? PerSampleInput,
        ShapeObservation[] Observations,
        long ElapsedMilliseconds);

    /// <summary>
    /// <c>Status</c> is "observed" when the worker ran every profile, otherwise the reason it did not
    /// ("timeout", "crashed", "error", "worker-missing", ...), with <c>Error</c> naming the cause.
    /// </summary>
    internal sealed record Observation(string Status, string? Error, ProfileObservation[] Profiles);

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public static async Task<Measurement> ProbeAsync(
        Type modelType,
        int extent,
        int classes,
        TimeSpan timeout)
    {
        var run = await RunWorkerAsync(
            new[]
            {
                "shape",
                modelType.AssemblyQualifiedName ?? modelType.FullName ?? modelType.Name,
                extent.ToString(System.Globalization.CultureInfo.InvariantCulture),
                classes.ToString(System.Globalization.CultureInfo.InvariantCulture),
            },
            timeout);

        if (run.Status is not null)
            return new Measurement(run.Status, null, null, null, run.Error);

        try
        {
            return JsonSerializer.Deserialize<Measurement>(run.Json!, JsonOptions)
                   ?? new Measurement("invalid-result", null, null, null, run.Json);
        }
        catch (JsonException)
        {
            return new Measurement("invalid-result", null, null, null, run.Json);
        }
    }

    /// <summary>
    /// Builds <paramref name="modelType"/> once per profile in an isolated worker process and returns
    /// the output shape Predict produced for each planned input shape. The worker gets a bounded
    /// managed heap and <paramref name="timeout"/>; a model that exceeds either is reported, killed,
    /// and cannot hold the calling sweep or leave work running after it.
    /// </summary>
    public static async Task<Observation> ObserveAsync(
        Type modelType,
        IReadOnlyList<ObservationProfile> profiles,
        TimeSpan timeout)
    {
        string plan = JsonSerializer.Serialize(new { Profiles = profiles });
        var run = await RunWorkerAsync(
            new[] { "observe", modelType.AssemblyQualifiedName ?? modelType.FullName ?? modelType.Name, plan },
            timeout);

        if (run.Status is not null)
            return new Observation(run.Status, run.Error, Array.Empty<ProfileObservation>());

        try
        {
            return JsonSerializer.Deserialize<Observation>(run.Json!, JsonOptions)
                   ?? new Observation("invalid-result", run.Json, Array.Empty<ProfileObservation>());
        }
        catch (JsonException)
        {
            return new Observation("invalid-result", run.Json, Array.Empty<ProfileObservation>());
        }
    }

    /// <summary>
    /// Runs the worker and returns its last stdout line, or a non-null <c>Status</c> saying why there
    /// is none.
    /// </summary>
    private static async Task<(string? Status, string? Error, string? Json)> RunWorkerAsync(
        IReadOnlyList<string> arguments,
        TimeSpan timeout)
    {
        await Task.Yield();

#if NET10_0_OR_GREATER
        string? worker = FindWorker();
        if (worker is null)
            return ("worker-missing", null, null);

        var start = new ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        start.ArgumentList.Add(worker);
        foreach (string argument in arguments) start.ArgumentList.Add(argument);

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
            return ("timeout", $"exceeded {timeout.TotalSeconds:F0} s", null);
        }

        string stdout = await stdoutTask;
        string stderr = await stderrTask;
        string? json = stdout.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
        if (json is null)
        {
            string error = string.IsNullOrWhiteSpace(stderr) ? $"exit {process.ExitCode}" : stderr.Trim();
            return ("crashed", error, null);
        }

        return (null, null, json);
#else
        return ("worker-unavailable", "The isolated shape sweep is supported on net10.0 and later.", null);
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
