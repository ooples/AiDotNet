#if !NET462
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Timing categories for GPU operations.
/// </summary>
public enum GpuTimingCategory
{
    /// <summary>Memory allocation from pool.</summary>
    MemoryAlloc,
    /// <summary>CPU to GPU memory transfer.</summary>
    CpuToGpuTransfer,
    /// <summary>GPU kernel execution.</summary>
    KernelExecution,
    /// <summary>GPU to CPU memory transfer.</summary>
    GpuToCpuTransfer,
    /// <summary>Memory return to pool.</summary>
    MemoryReturn,
    /// <summary>GPU synchronization.</summary>
    Synchronization,
    /// <summary>View/buffer creation.</summary>
    ViewCreation,
    /// <summary>Lock acquisition.</summary>
    LockAcquisition,
    /// <summary>Total operation time.</summary>
    TotalOperation
}

/// <summary>
/// Records timing data for a single GPU operation.
/// </summary>
public readonly struct GpuTimingRecord
{
    /// <summary>Operation name.</summary>
    public string Operation { get; }
    /// <summary>Timing category.</summary>
    public GpuTimingCategory Category { get; }
    /// <summary>Elapsed time in milliseconds.</summary>
    public double ElapsedMs { get; }
    /// <summary>Data size in bytes (if applicable).</summary>
    public long DataSizeBytes { get; }
    /// <summary>Timestamp when recorded.</summary>
    public long TimestampTicks { get; }

    public GpuTimingRecord(string operation, GpuTimingCategory category, double elapsedMs, long dataSizeBytes = 0)
    {
        Operation = operation;
        Category = category;
        ElapsedMs = elapsedMs;
        DataSizeBytes = dataSizeBytes;
        TimestampTicks = Stopwatch.GetTimestamp();
    }
}

/// <summary>
/// Provides detailed timing diagnostics for GPU operations.
/// Thread-safe for use in concurrent environments.
/// </summary>
/// <remarks>
/// <para><b>Phase B: GPU Performance Profiling</b></para>
/// <para>
/// This class helps identify performance bottlenecks by tracking:
/// - Memory allocation/deallocation overhead
/// - CPU-GPU transfer times
/// - Kernel execution times
/// - Synchronization overhead
/// </para>
/// </remarks>
public class GpuTimingDiagnostics
{
    private readonly ConcurrentQueue<GpuTimingRecord> _records;
    private readonly ConcurrentDictionary<string, long> _operationCounts;
    private readonly ConcurrentDictionary<string, double> _operationTotalMs;
    private readonly bool _isEnabled;
    private readonly int _maxRecords;

    /// <summary>
    /// Global instance for easy access. Disabled by default for performance.
    /// </summary>
    public static GpuTimingDiagnostics Instance { get; } = new GpuTimingDiagnostics(enabled: false);

    /// <summary>
    /// Creates a new diagnostic-enabled instance for testing.
    /// </summary>
    public static GpuTimingDiagnostics CreateEnabled(int maxRecords = 10000)
        => new GpuTimingDiagnostics(enabled: true, maxRecords);

    /// <summary>
    /// Initializes a new instance of GpuTimingDiagnostics.
    /// </summary>
    /// <param name="enabled">Whether timing collection is enabled.</param>
    /// <param name="maxRecords">Maximum records to retain before dropping oldest.</param>
    public GpuTimingDiagnostics(bool enabled = false, int maxRecords = 10000)
    {
        _isEnabled = enabled;
        _maxRecords = maxRecords;
        _records = new ConcurrentQueue<GpuTimingRecord>();
        _operationCounts = new ConcurrentDictionary<string, long>();
        _operationTotalMs = new ConcurrentDictionary<string, double>();
    }

    /// <summary>
    /// Whether timing collection is enabled.
    /// </summary>
    public bool IsEnabled => _isEnabled;

    /// <summary>
    /// Records a timing measurement.
    /// </summary>
    public void Record(string operation, GpuTimingCategory category, double elapsedMs, long dataSizeBytes = 0)
    {
        if (!_isEnabled) return;

        var record = new GpuTimingRecord(operation, category, elapsedMs, dataSizeBytes);
        _records.Enqueue(record);

        // Update aggregates
        string key = $"{operation}:{category}";
        _operationCounts.AddOrUpdate(key, 1, (_, c) => c + 1);
        _operationTotalMs.AddOrUpdate(key, elapsedMs, (_, t) => t + elapsedMs);

        // Trim if over max
        while (_records.Count > _maxRecords && _records.TryDequeue(out _)) { }
    }

    /// <summary>
    /// Starts a timing scope that automatically records when disposed.
    /// </summary>
    public TimingScope StartScope(string operation, GpuTimingCategory category, long dataSizeBytes = 0)
        => new TimingScope(this, operation, category, dataSizeBytes);

    /// <summary>
    /// Clears all recorded timing data.
    /// </summary>
    public void Clear()
    {
        while (_records.TryDequeue(out _)) { }
        _operationCounts.Clear();
        _operationTotalMs.Clear();
    }

    /// <summary>
    /// Gets a summary report of all timing data.
    /// </summary>
    public string GetSummary()
    {
        if (!_isEnabled)
            return "GPU Timing Diagnostics: DISABLED (enable for performance profiling)";

        var sb = new StringBuilder();
        sb.AppendLine("=== GPU TIMING DIAGNOSTICS SUMMARY ===");
        sb.AppendLine();

        // Group by category
        var categoryStats = _operationCounts.Keys
            .Select(k => new
            {
                Key = k,
                Parts = k.Split(':'),
                Count = _operationCounts[k],
                TotalMs = _operationTotalMs.TryGetValue(k, out var t) ? t : 0
            })
            .Where(x => x.Parts.Length == 2)
            .Select(x => new
            {
                Operation = x.Parts[0],
                Category = Enum.TryParse<GpuTimingCategory>(x.Parts[1], out var c) ? c : GpuTimingCategory.TotalOperation,
                x.Count,
                x.TotalMs,
                AvgMs = x.Count > 0 ? x.TotalMs / x.Count : 0
            })
            .GroupBy(x => x.Category)
            .OrderByDescending(g => g.Sum(x => x.TotalMs));

        foreach (var categoryGroup in categoryStats)
        {
            double categoryTotalMs = categoryGroup.Sum(x => x.TotalMs);
            long categoryCount = categoryGroup.Sum(x => x.Count);

            sb.AppendLine($"[{categoryGroup.Key}] Total: {categoryTotalMs:F3}ms | Count: {categoryCount}");
            sb.AppendLine(new string('-', 60));

            foreach (var op in categoryGroup.OrderByDescending(x => x.TotalMs).Take(10))
            {
                sb.AppendLine($"  {op.Operation,-30} | Total: {op.TotalMs,10:F3}ms | Count: {op.Count,6} | Avg: {op.AvgMs,8:F3}ms");
            }
            sb.AppendLine();
        }

        // Overall breakdown
        sb.AppendLine("=== TIME BREAKDOWN BY CATEGORY ===");
        var totalByCategory = categoryStats
            .Select(g => new { Category = g.Key, TotalMs = g.Sum(x => x.TotalMs) })
            .OrderByDescending(x => x.TotalMs)
            .ToList();

        double grandTotal = totalByCategory.Sum(x => x.TotalMs);
        foreach (var cat in totalByCategory)
        {
            double pct = grandTotal > 0 ? (cat.TotalMs / grandTotal * 100) : 0;
            sb.AppendLine($"  {cat.Category,-20} | {cat.TotalMs,10:F3}ms | {pct,5:F1}%");
        }
        sb.AppendLine($"  {"GRAND TOTAL",-20} | {grandTotal,10:F3}ms | 100.0%");

        return sb.ToString();
    }

    /// <summary>
    /// Gets detailed breakdown for a specific operation.
    /// </summary>
    public string GetOperationBreakdown(string operationName)
    {
        if (!_isEnabled)
            return "GPU Timing Diagnostics: DISABLED";

        var records = _records.Where(r => r.Operation.StartsWith(operationName)).ToList();
        if (records.Count == 0)
            return $"No timing data for operation: {operationName}";

        var sb = new StringBuilder();
        sb.AppendLine($"=== BREAKDOWN: {operationName} ===");

        var grouped = records
            .GroupBy(r => r.Category)
            .OrderByDescending(g => g.Sum(r => r.ElapsedMs));

        foreach (var g in grouped)
        {
            double total = g.Sum(r => r.ElapsedMs);
            double avg = g.Average(r => r.ElapsedMs);
            double min = g.Min(r => r.ElapsedMs);
            double max = g.Max(r => r.ElapsedMs);
            long dataBytes = g.Sum(r => r.DataSizeBytes);

            sb.AppendLine($"  {g.Key,-20} | Total: {total,10:F3}ms | Avg: {avg,8:F3}ms | Min: {min,8:F3}ms | Max: {max,8:F3}ms");
            if (dataBytes > 0)
            {
                double throughputMBps = dataBytes / (total / 1000.0) / (1024 * 1024);
                sb.AppendLine($"    Data: {dataBytes / 1024.0:F1} KB | Throughput: {throughputMBps:F1} MB/s");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Returns recent records for inspection.
    /// </summary>
    public GpuTimingRecord[] GetRecentRecords(int count = 100)
        => _records.TakeLast(count).ToArray();

    /// <summary>
    /// RAII-style timing scope.
    /// </summary>
    public readonly struct TimingScope : IDisposable
    {
        private readonly GpuTimingDiagnostics _diagnostics;
        private readonly string _operation;
        private readonly GpuTimingCategory _category;
        private readonly long _dataSizeBytes;
        private readonly long _startTicks;

        internal TimingScope(GpuTimingDiagnostics diagnostics, string operation, GpuTimingCategory category, long dataSizeBytes)
        {
            _diagnostics = diagnostics;
            _operation = operation;
            _category = category;
            _dataSizeBytes = dataSizeBytes;
            _startTicks = diagnostics._isEnabled ? Stopwatch.GetTimestamp() : 0;
        }

        public void Dispose()
        {
            if (!_diagnostics._isEnabled) return;

            long endTicks = Stopwatch.GetTimestamp();
            double elapsedMs = (endTicks - _startTicks) * 1000.0 / Stopwatch.Frequency;
            _diagnostics.Record(_operation, _category, elapsedMs, _dataSizeBytes);
        }
    }
}
#endif
