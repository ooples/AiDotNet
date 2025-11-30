using System.Text;
using System.Text.Json;

namespace AiDotNet.Diagnostics;

/// <summary>
/// A comprehensive profiling report with statistics and analysis.
/// </summary>
/// <remarks>
/// <para><b>Features:</b>
/// - Summary statistics for all profiled operations
/// - Call hierarchy visualization
/// - Hotspot identification
/// - Export to JSON, CSV, and markdown formats
/// </para>
/// </remarks>
public class ProfileReport
{
    private readonly List<ProfilerStats> _stats;
    private readonly TimeSpan _totalRuntime;
    private readonly DateTime _startTime;

    /// <summary>
    /// Gets all profiled operation statistics.
    /// </summary>
    public IReadOnlyList<ProfilerStats> Stats => _stats;

    /// <summary>
    /// Gets the total profiling runtime.
    /// </summary>
    public TimeSpan TotalRuntime => _totalRuntime;

    /// <summary>
    /// Gets when profiling started.
    /// </summary>
    public DateTime StartTime => _startTime;

    /// <summary>
    /// Gets the total number of profiled operations.
    /// </summary>
    public int TotalOperations => _stats.Sum(s => s.Count);

    /// <summary>
    /// Gets the total time spent in profiled operations.
    /// </summary>
    public double TotalProfiledTimeMs => _stats.Sum(s => s.TotalMs);

    internal ProfileReport(List<ProfilerEntry> entries, TimeSpan runtime, DateTime startTime)
    {
        _stats = entries.Select(e => e.GetStats()).ToList();
        _totalRuntime = runtime;
        _startTime = startTime;
    }

    /// <summary>
    /// Gets statistics for a specific operation.
    /// </summary>
    public ProfilerStats? GetStats(string name)
    {
        return _stats.FirstOrDefault(s => s.Name == name);
    }

    /// <summary>
    /// Gets the top N hotspots by total time.
    /// </summary>
    public IEnumerable<ProfilerStats> GetHotspots(int topN = 10)
    {
        return _stats.OrderByDescending(s => s.TotalMs).Take(topN);
    }

    /// <summary>
    /// Gets operations sorted by mean time (slowest first).
    /// </summary>
    public IEnumerable<ProfilerStats> GetSlowest(int topN = 10)
    {
        return _stats.Where(s => s.Count > 0)
                     .OrderByDescending(s => s.MeanMs)
                     .Take(topN);
    }

    /// <summary>
    /// Gets operations with highest variance (most inconsistent).
    /// </summary>
    public IEnumerable<ProfilerStats> GetMostVariable(int topN = 10)
    {
        return _stats.Where(s => s.Count > 1)
                     .OrderByDescending(s => s.StdDevMs / Math.Max(s.MeanMs, 0.001))
                     .Take(topN);
    }

    /// <summary>
    /// Gets a summary string representation.
    /// </summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== Profile Report ===");
        sb.AppendLine($"Start Time: {_startTime:yyyy-MM-dd HH:mm:ss.fff}");
        sb.AppendLine($"Total Runtime: {_totalRuntime.TotalSeconds:F2}s");
        sb.AppendLine($"Total Operations: {TotalOperations:N0}");
        sb.AppendLine($"Total Profiled Time: {TotalProfiledTimeMs:F2}ms");
        sb.AppendLine();

        if (_stats.Count == 0)
        {
            sb.AppendLine("No operations profiled.");
            return sb.ToString();
        }

        sb.AppendLine("=== Top Operations by Total Time ===");
        sb.AppendLine($"{"Operation",-40} {"Count",10} {"Mean (ms)",12} {"P95 (ms)",12} {"Total (ms)",12}");
        sb.AppendLine(new string('-', 86));

        foreach (var stat in GetHotspots(15))
        {
            sb.AppendLine($"{TruncateName(stat.Name, 40),-40} {stat.Count,10:N0} {stat.MeanMs,12:F3} {stat.P95Ms,12:F3} {stat.TotalMs,12:F1}");
        }

        if (_stats.Any(s => s.AllocationCount > 0))
        {
            sb.AppendLine();
            sb.AppendLine("=== Memory Allocations ===");
            var withAllocs = _stats.Where(s => s.AllocationCount > 0)
                                   .OrderByDescending(s => s.TotalAllocations);
            foreach (var stat in withAllocs.Take(10))
            {
                sb.AppendLine($"{TruncateName(stat.Name, 40),-40} {FormatBytes(stat.TotalAllocations),15} ({stat.AllocationCount} allocations)");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Exports the report to JSON format.
    /// </summary>
    public string ToJson(bool indented = true)
    {
        var data = new
        {
            StartTime = _startTime,
            TotalRuntimeMs = _totalRuntime.TotalMilliseconds,
            TotalOperations,
            TotalProfiledTimeMs,
            Operations = _stats.Select(s => new
            {
                s.Name,
                s.Count,
                s.TotalMs,
                s.MeanMs,
                s.MinMs,
                s.MaxMs,
                s.P50Ms,
                s.P95Ms,
                s.P99Ms,
                s.StdDevMs,
                s.OpsPerSecond,
                s.TotalAllocations,
                s.AllocationCount,
                s.Parents
            }).ToList()
        };

        var options = new JsonSerializerOptions
        {
            WriteIndented = indented
        };

        return JsonSerializer.Serialize(data, options);
    }

    /// <summary>
    /// Exports the report to CSV format.
    /// </summary>
    public string ToCsv()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Name,Count,TotalMs,MeanMs,MinMs,MaxMs,P50Ms,P95Ms,P99Ms,StdDevMs,OpsPerSec,TotalAllocBytes,AllocCount");

        foreach (var stat in _stats)
        {
            sb.AppendLine($"\"{stat.Name}\",{stat.Count},{stat.TotalMs:F3},{stat.MeanMs:F3},{stat.MinMs:F3},{stat.MaxMs:F3},{stat.P50Ms:F3},{stat.P95Ms:F3},{stat.P99Ms:F3},{stat.StdDevMs:F3},{stat.OpsPerSecond:F2},{stat.TotalAllocations},{stat.AllocationCount}");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Exports the report to markdown format.
    /// </summary>
    public string ToMarkdown()
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Profile Report");
        sb.AppendLine();
        sb.AppendLine($"- **Start Time:** {_startTime:yyyy-MM-dd HH:mm:ss.fff}");
        sb.AppendLine($"- **Total Runtime:** {_totalRuntime.TotalSeconds:F2}s");
        sb.AppendLine($"- **Total Operations:** {TotalOperations:N0}");
        sb.AppendLine($"- **Total Profiled Time:** {TotalProfiledTimeMs:F2}ms");
        sb.AppendLine();

        if (_stats.Count == 0)
        {
            sb.AppendLine("No operations profiled.");
            return sb.ToString();
        }

        sb.AppendLine("## Operations by Total Time");
        sb.AppendLine();
        sb.AppendLine("| Operation | Count | Mean (ms) | P95 (ms) | Total (ms) |");
        sb.AppendLine("|-----------|------:|----------:|---------:|-----------:|");

        foreach (var stat in GetHotspots(20))
        {
            sb.AppendLine($"| {stat.Name} | {stat.Count:N0} | {stat.MeanMs:F3} | {stat.P95Ms:F3} | {stat.TotalMs:F1} |");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Compares this report with another to find regressions.
    /// </summary>
    /// <param name="baseline">The baseline report to compare against.</param>
    /// <param name="thresholdPercent">Threshold for reporting regressions (default 10%).</param>
    /// <returns>Comparison results.</returns>
    public ProfileComparison CompareTo(ProfileReport baseline, double thresholdPercent = 10.0)
    {
        var comparisons = new List<ProfileComparisonEntry>();

        foreach (var currentStat in _stats)
        {
            var baselineStat = baseline.GetStats(currentStat.Name);
            if (baselineStat != null && baselineStat.Count > 0 && currentStat.Count > 0)
            {
                double changePercent = ((currentStat.MeanMs - baselineStat.MeanMs) / baselineStat.MeanMs) * 100;

                comparisons.Add(new ProfileComparisonEntry
                {
                    Name = currentStat.Name,
                    BaselineMeanMs = baselineStat.MeanMs,
                    CurrentMeanMs = currentStat.MeanMs,
                    ChangePercent = changePercent,
                    IsRegression = changePercent > thresholdPercent,
                    IsImprovement = changePercent < -thresholdPercent
                });
            }
        }

        return new ProfileComparison(comparisons, thresholdPercent);
    }

    private static string TruncateName(string name, int maxLength)
    {
        if (name.Length <= maxLength) return name;
        return string.Concat(name.AsSpan(0, maxLength - 3), "...");
    }

    private static string FormatBytes(long bytes)
    {
        string[] suffixes = { "B", "KB", "MB", "GB", "TB" };
        int suffixIndex = 0;
        double size = bytes;

        while (size >= 1024 && suffixIndex < suffixes.Length - 1)
        {
            size /= 1024;
            suffixIndex++;
        }

        return $"{size:F2} {suffixes[suffixIndex]}";
    }
}

/// <summary>
/// Results of comparing two profile reports.
/// </summary>
public class ProfileComparison
{
    private readonly List<ProfileComparisonEntry> _entries;
    private readonly double _threshold;

    public IReadOnlyList<ProfileComparisonEntry> Entries => _entries;
    public double ThresholdPercent => _threshold;

    public int RegressionCount => _entries.Count(e => e.IsRegression);
    public int ImprovementCount => _entries.Count(e => e.IsImprovement);

    internal ProfileComparison(List<ProfileComparisonEntry> entries, double threshold)
    {
        _entries = entries;
        _threshold = threshold;
    }

    public IEnumerable<ProfileComparisonEntry> GetRegressions() =>
        _entries.Where(e => e.IsRegression).OrderByDescending(e => e.ChangePercent);

    public IEnumerable<ProfileComparisonEntry> GetImprovements() =>
        _entries.Where(e => e.IsImprovement).OrderBy(e => e.ChangePercent);

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"=== Profile Comparison (threshold: {_threshold}%) ===");
        sb.AppendLine($"Regressions: {RegressionCount}, Improvements: {ImprovementCount}");
        sb.AppendLine();

        if (RegressionCount > 0)
        {
            sb.AppendLine("Regressions:");
            foreach (var entry in GetRegressions().Take(10))
            {
                sb.AppendLine($"  {entry.Name}: {entry.BaselineMeanMs:F3}ms -> {entry.CurrentMeanMs:F3}ms (+{entry.ChangePercent:F1}%)");
            }
        }

        if (ImprovementCount > 0)
        {
            sb.AppendLine("Improvements:");
            foreach (var entry in GetImprovements().Take(10))
            {
                sb.AppendLine($"  {entry.Name}: {entry.BaselineMeanMs:F3}ms -> {entry.CurrentMeanMs:F3}ms ({entry.ChangePercent:F1}%)");
            }
        }

        return sb.ToString();
    }
}

/// <summary>
/// A single entry in a profile comparison.
/// </summary>
public class ProfileComparisonEntry
{
    public required string Name { get; init; }
    public double BaselineMeanMs { get; init; }
    public double CurrentMeanMs { get; init; }
    public double ChangePercent { get; init; }
    public bool IsRegression { get; init; }
    public bool IsImprovement { get; init; }
}
