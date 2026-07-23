using System.Text;
using AiDotNet.Tensors.Engines.Optimization;

namespace AiDotNet.Diagnostics;

/// <summary>
/// Structured performance report captured at build time when the builder opts in via
/// <c>EnableProfiling()</c>. Wraps Tensors' <see cref="PerformanceProfiler"/> output
/// so callers don't have to dip into Tensors internals for a timing breakdown.
/// </summary>
public sealed class TensorsOperationProfile
{
    /// <summary>Per-operation timing statistics, sorted by total time descending.</summary>
    public IReadOnlyList<OperationTiming> Operations { get; init; } = Array.Empty<OperationTiming>();

    /// <summary>Total wall-clock time across every profiled operation (ms).</summary>
    public double TotalMilliseconds { get; init; }

    /// <summary>The profiler's raw text report (via <c>PerformanceProfiler.GenerateReport</c>).</summary>
    public string RawReport { get; init; } = "";

    /// <summary>
    /// Builds a ProfilingReport from the live Tensors
    /// <see cref="PerformanceProfiler.Instance"/>.
    /// </summary>
    public static TensorsOperationProfile Capture()
    {
        var profiler = PerformanceProfiler.Instance;
        var stats = profiler.GetAllStats();
        var ops = new List<OperationTiming>(stats.Length);
        double total = 0;

        foreach (var s in stats)
        {
            ops.Add(new OperationTiming
            {
                Name = s.OperationName,
                CallCount = s.CallCount,
                TotalMilliseconds = s.TotalMilliseconds,
                AverageMilliseconds = s.AverageMilliseconds,
                MinMilliseconds = s.MinMilliseconds,
                MaxMilliseconds = s.MaxMilliseconds,
                ThroughputOpsPerSecond = s.ThroughputOpsPerSecond,
                TotalMemoryMB = s.TotalMemoryMB,
            });
            total += s.TotalMilliseconds;
        }

        ops.Sort((a, b) => b.TotalMilliseconds.CompareTo(a.TotalMilliseconds));

        return new TensorsOperationProfile
        {
            Operations = ops,
            TotalMilliseconds = total,
            RawReport = profiler.GenerateReport(),
        };
    }

    public string FormatSummary(int topN = 10)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"=== AiDotNet Profiling Summary ({Operations.Count} ops, {TotalMilliseconds:F2} ms total) ===");
        foreach (var op in Operations.Take(topN))
        {
            sb.AppendLine(
                $"  {op.Name,-40} " +
                $"calls={op.CallCount,6}  " +
                $"total={op.TotalMilliseconds,9:F2}ms  " +
                $"avg={op.AverageMilliseconds,7:F3}ms  " +
                $"throughput={op.ThroughputOpsPerSecond,10:F0}/s");
        }
        return sb.ToString();
    }
}

public sealed class OperationTiming
{
    public string Name { get; init; } = "";
    public long CallCount { get; init; }
    public double TotalMilliseconds { get; init; }
    public double AverageMilliseconds { get; init; }
    public double MinMilliseconds { get; init; }
    public double MaxMilliseconds { get; init; }
    public double ThroughputOpsPerSecond { get; init; }
    public double TotalMemoryMB { get; init; }
}
