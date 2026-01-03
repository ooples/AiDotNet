// Copyright (c) 2024 AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Profiling;

/// <summary>
/// Identifies the primary performance bottleneck.
/// </summary>
public enum BottleneckType
{
    /// <summary>Unknown or not yet analyzed.</summary>
    Unknown,

    /// <summary>Kernel launch overhead dominates - matrix too small.</summary>
    LaunchOverhead,

    /// <summary>Memory bandwidth limited - need better memory access patterns.</summary>
    MemoryBandwidth,

    /// <summary>Compute limited - achieving good hardware utilization.</summary>
    Compute,

    /// <summary>Occupancy limited - not enough waves to hide latency.</summary>
    Occupancy,

    /// <summary>LDS bandwidth limited - bank conflicts or insufficient LDS.</summary>
    LdsBandwidth
}

/// <summary>
/// Recommended optimization action based on bottleneck analysis.
/// </summary>
public enum OptimizationAction
{
    /// <summary>No action needed - already optimal.</summary>
    None,

    /// <summary>Batch multiple small operations together.</summary>
    BatchOperations,

    /// <summary>Use CPU instead - GPU overhead too high for this size.</summary>
    UseCpu,

    /// <summary>Improve memory tiling and prefetching.</summary>
    OptimizeTiling,

    /// <summary>Increase occupancy by reducing register/LDS usage.</summary>
    IncreaseOccupancy,

    /// <summary>Reduce LDS bank conflicts with padding or swizzling.</summary>
    ReduceBankConflicts,

    /// <summary>Try different GEMM parameters via auto-tuning.</summary>
    AutoTune,

    /// <summary>Use vectorized loads (float4) for better memory throughput.</summary>
    Vectorize
}

/// <summary>
/// Result of profiling a single GEMM operation.
/// </summary>
public sealed class GemmProfileEntry
{
    /// <summary>Matrix M dimension.</summary>
    public int M { get; init; }

    /// <summary>Matrix N dimension.</summary>
    public int N { get; init; }

    /// <summary>Matrix K dimension.</summary>
    public int K { get; init; }

    /// <summary>Time spent in kernel queue (microseconds).</summary>
    public double QueueTimeUs { get; init; }

    /// <summary>Time from queue to kernel start (microseconds).</summary>
    public double SubmitToStartUs { get; init; }

    /// <summary>Actual kernel execution time (microseconds).</summary>
    public double ExecutionTimeUs { get; init; }

    /// <summary>Total wall-clock time including overhead (microseconds).</summary>
    public double TotalTimeUs { get; init; }

    /// <summary>Achieved GFLOPS (2*M*N*K / execution_time).</summary>
    public double Gflops { get; init; }

    /// <summary>Efficiency as percentage of theoretical peak.</summary>
    public double EfficiencyPercent { get; init; }

    /// <summary>Achieved memory bandwidth in GB/s.</summary>
    public double MemoryBandwidthGBs { get; init; }

    /// <summary>Arithmetic intensity (FLOPs per byte transferred).</summary>
    public double ArithmeticIntensity { get; init; }

    /// <summary>Roofline-limited performance (GB/s).</summary>
    public double RooflineLimitGflops { get; init; }

    /// <summary>Primary bottleneck identified.</summary>
    public BottleneckType Bottleneck { get; init; }

    /// <summary>Recommended optimization action.</summary>
    public OptimizationAction RecommendedAction { get; init; }

    /// <summary>Occupancy analysis result (if available).</summary>
    public OccupancyResult? Occupancy { get; init; }

    /// <summary>Launch overhead as percentage of total time.</summary>
    public double LaunchOverheadPercent => TotalTimeUs > 0
        ? 100.0 * (TotalTimeUs - ExecutionTimeUs) / TotalTimeUs
        : 0;

    /// <summary>
    /// Gets a human-readable summary of this profile entry.
    /// </summary>
    public string GetSummary()
    {
        return $"{M}x{N}x{K}: {Gflops:F0} GFLOPS ({EfficiencyPercent:F1}%) - {Bottleneck} → {RecommendedAction}";
    }
}

/// <summary>
/// Comprehensive profiling result for a GPU GEMM analysis session.
/// </summary>
public sealed class ProfileResult
{
    /// <summary>GPU device name.</summary>
    public string DeviceName { get; init; } = string.Empty;

    /// <summary>Detected GPU architecture.</summary>
    public GpuArchitectureSpec? Architecture { get; init; }

    /// <summary>Theoretical peak GFLOPS for this GPU.</summary>
    public double PeakGflops { get; init; }

    /// <summary>Theoretical peak memory bandwidth in GB/s.</summary>
    public double PeakBandwidthGBs { get; init; }

    /// <summary>Ridge point (FLOPS/byte where compute = memory bound).</summary>
    public double RidgePoint { get; init; }

    /// <summary>Timestamp when profiling started.</summary>
    public DateTime ProfileStartTime { get; init; }

    /// <summary>Total profiling duration in seconds.</summary>
    public double ProfileDurationSeconds { get; init; }

    /// <summary>Individual GEMM profile entries.</summary>
    public List<GemmProfileEntry> Entries { get; init; } = [];

    /// <summary>Best achieved GFLOPS across all sizes.</summary>
    public double BestGflops => Entries.Count > 0 ? Entries.Max(e => e.Gflops) : 0;

    /// <summary>Best efficiency percentage achieved.</summary>
    public double BestEfficiencyPercent => Entries.Count > 0 ? Entries.Max(e => e.EfficiencyPercent) : 0;

    /// <summary>Matrix size that achieved best performance.</summary>
    public (int M, int N, int K) BestSize
    {
        get
        {
            var best = Entries.OrderByDescending(e => e.Gflops).FirstOrDefault();
            return best != null ? (best.M, best.N, best.K) : (0, 0, 0);
        }
    }

    /// <summary>
    /// Gets the crossover point where GPU becomes faster than CPU.
    /// </summary>
    /// <param name="cpuGflops">CPU baseline GFLOPS.</param>
    /// <returns>Minimum matrix dimension where GPU wins, or -1 if never.</returns>
    public int GetGpuCrossoverSize(double cpuGflops)
    {
        // Find smallest matrix where GPU GFLOPS exceeds CPU
        var gpuWins = Entries
            .Where(e => e.Gflops > cpuGflops)
            .OrderBy(e => e.M * e.N * e.K)
            .FirstOrDefault();

        return gpuWins != null ? Math.Min(gpuWins.M, Math.Min(gpuWins.N, gpuWins.K)) : -1;
    }

    /// <summary>
    /// Gets entries grouped by bottleneck type.
    /// </summary>
    public Dictionary<BottleneckType, List<GemmProfileEntry>> GetBottleneckGroups()
    {
        return Entries
            .GroupBy(e => e.Bottleneck)
            .ToDictionary(g => g.Key, g => g.ToList());
    }

    /// <summary>
    /// Generates a formatted console report.
    /// </summary>
    public string GenerateConsoleReport()
    {
        var sb = new System.Text.StringBuilder();

        sb.AppendLine("=".PadRight(100, '='));
        sb.AppendLine("GPU GEMM PROFILING REPORT");
        sb.AppendLine("=".PadRight(100, '='));
        sb.AppendLine();
        sb.AppendLine($"Device: {DeviceName}");
        if (Architecture != null)
        {
            sb.AppendLine($"Architecture: {Architecture.Name}");
            sb.AppendLine($"  VGPRs/SIMD: {Architecture.VgprsPerSimd}");
            sb.AppendLine($"  LDS/CU: {Architecture.LdsPerCuBytes / 1024} KB");
            sb.AppendLine($"  Max Waves/SIMD: {Architecture.MaxWavesPerSimd}");
            sb.AppendLine($"  Wavefront Size: {Architecture.WavefrontSize}");
        }
        sb.AppendLine($"Peak Performance: {PeakGflops:F0} GFLOPS");
        sb.AppendLine($"Peak Bandwidth: {PeakBandwidthGBs:F0} GB/s");
        sb.AppendLine($"Ridge Point: {RidgePoint:F1} FLOPS/byte");
        sb.AppendLine();
        sb.AppendLine($"Best Achieved: {BestGflops:F0} GFLOPS ({BestEfficiencyPercent:F1}% efficiency)");
        var (bm, bn, bk) = BestSize;
        sb.AppendLine($"Best Size: {bm}x{bn}x{bk}");
        sb.AppendLine();

        // Table header
        sb.AppendLine("-".PadRight(100, '-'));
        sb.AppendLine(string.Format("{0,-15} {1,10} {2,8} {3,10} {4,10} {5,12} {6,-15}",
            "Size", "GFLOPS", "Eff%", "BW GB/s", "AI F/B", "Bottleneck", "Action"));
        sb.AppendLine("-".PadRight(100, '-'));

        foreach (var entry in Entries.OrderBy(e => e.M * e.N * e.K))
        {
            string sizeStr = entry.M == entry.N && entry.N == entry.K
                ? $"{entry.M}"
                : $"{entry.M}x{entry.N}x{entry.K}";

            sb.AppendLine(string.Format("{0,-15} {1,10:F0} {2,7:F1}% {3,10:F1} {4,10:F2} {5,12} {6,-15}",
                sizeStr,
                entry.Gflops,
                entry.EfficiencyPercent,
                entry.MemoryBandwidthGBs,
                entry.ArithmeticIntensity,
                entry.Bottleneck,
                entry.RecommendedAction));
        }

        sb.AppendLine("-".PadRight(100, '-'));

        // Bottleneck summary
        sb.AppendLine();
        sb.AppendLine("BOTTLENECK SUMMARY:");
        var groups = GetBottleneckGroups();
        foreach (var group in groups.OrderByDescending(g => g.Value.Count))
        {
            var sizes = string.Join(", ", group.Value.Select(e => e.M.ToString()));
            sb.AppendLine($"  {group.Key}: {group.Value.Count} sizes ({sizes})");
        }

        sb.AppendLine();
        sb.AppendLine($"Profile completed at {ProfileStartTime:yyyy-MM-dd HH:mm:ss} ({ProfileDurationSeconds:F1}s)");
        sb.AppendLine("=".PadRight(100, '='));

        return sb.ToString();
    }
}
