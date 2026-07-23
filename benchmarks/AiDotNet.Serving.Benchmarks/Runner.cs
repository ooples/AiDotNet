using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Serving.Benchmarks.Backends;

namespace AiDotNet.Serving.Benchmarks;

/// <summary>Drives a workload against a backend, honoring the arrival schedule and a concurrency cap.</summary>
public static class Runner
{
    /// <summary>
    /// Runs all <paramref name="specs"/> against <paramref name="backend"/>. Requests are dispatched at
    /// their scheduled arrival times (bounded by <see cref="BenchmarkOptions.Concurrency"/> in-flight) and
    /// executed concurrently. Returns per-request results and the measured wall-clock duration in seconds.
    /// </summary>
    public static async Task<(IReadOnlyList<RequestResult> Results, double DurationSec)> RunAsync(
        IServingBackend backend, IReadOnlyList<RequestSpec> specs, BenchmarkOptions o, CancellationToken ct)
    {
        using var sem = new SemaphoreSlim(o.Concurrency);
        var tasks = new List<Task<RequestResult>>(specs.Count);
        var runSw = Stopwatch.StartNew();

        foreach (var spec in specs)
        {
            // Respect the arrival schedule (no-op when rate is infinite: all offsets are 0).
            double waitMs = spec.ArrivalOffsetMs - runSw.Elapsed.TotalMilliseconds;
            if (waitMs > 1.0)
                await Task.Delay(TimeSpan.FromMilliseconds(waitMs), ct).ConfigureAwait(false);

            await sem.WaitAsync(ct).ConfigureAwait(false);
            double dispatchMs = runSw.Elapsed.TotalMilliseconds;

            tasks.Add(Task.Run(async () =>
            {
                try { return await backend.ExecuteAsync(spec, dispatchMs, ct).ConfigureAwait(false); }
                finally { sem.Release(); }
            }, ct));
        }

        var results = await Task.WhenAll(tasks).ConfigureAwait(false);
        runSw.Stop();
        return (results, runSw.Elapsed.TotalSeconds);
    }
}
