using System.Diagnostics;

namespace AiDotNet.Diagnostics;

/// <summary>
/// Tracks memory usage and allocations during ML operations.
/// </summary>
/// <remarks>
/// <para><b>Features:</b>
/// - GC heap tracking
/// - Working set monitoring
/// - Allocation rate calculation
/// - Memory snapshot comparison
/// </para>
/// <para><b>Usage:</b>
/// <code>
/// // Take a snapshot before an operation
/// var before = MemoryTracker.Snapshot();
///
/// // Run your operation
/// model.Train(data);
///
/// // Take a snapshot after
/// var after = MemoryTracker.Snapshot();
///
/// // Compare
/// var diff = after.CompareTo(before);
/// Console.WriteLine($"Memory delta: {diff.TotalMemoryDelta / 1024 / 1024:F2} MB");
/// </code>
/// </para>
/// </remarks>
public static class MemoryTracker
{
    private static readonly List<MemorySnapshot> _history = new();
    private static readonly object _lock = new();
    private static bool _enabled = false;
    private static DateTime _startTime = DateTime.UtcNow;
    private static int _maxHistorySize = 10000; // Prevent unbounded memory growth

    /// <summary>
    /// Gets whether memory tracking is enabled.
    /// </summary>
    public static bool IsEnabled => _enabled;

    /// <summary>
    /// Gets or sets the maximum history size (default: 10000).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Limits how many snapshots are kept in history to prevent
    /// unbounded memory growth in long-running applications. When the limit is reached,
    /// older snapshots are removed to make room for new ones.
    /// </para>
    /// </remarks>
    public static int MaxHistorySize
    {
        get { lock (_lock) return _maxHistorySize; }
        set { lock (_lock) _maxHistorySize = Math.Max(1, value); }
    }

    /// <summary>
    /// Enables memory tracking.
    /// </summary>
    public static void Enable()
    {
        lock (_lock)
        {
            _enabled = true;
            _startTime = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Disables memory tracking.
    /// </summary>
    public static void Disable()
    {
        lock (_lock)
        {
            _enabled = false;
        }
    }

    /// <summary>
    /// Clears all recorded history.
    /// </summary>
    public static void Reset()
    {
        lock (_lock)
        {
            _history.Clear();
            _startTime = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Takes a snapshot of current memory usage.
    /// </summary>
    /// <param name="label">Optional label for this snapshot.</param>
    /// <param name="forceGC">Whether to force garbage collection before measuring.</param>
    /// <returns>A MemorySnapshot with current memory metrics.</returns>
    public static MemorySnapshot Snapshot(string? label = null, bool forceGC = false)
    {
        if (forceGC)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        long workingSet;
        long privateMemory;
        long virtualMemory;

        using (var process = Process.GetCurrentProcess())
        {
            workingSet = process.WorkingSet64;
            privateMemory = process.PrivateMemorySize64;
            virtualMemory = process.VirtualMemorySize64;
        }

        var snapshot = new MemorySnapshot
        {
            Label = label ?? $"Snapshot_{_history.Count}",
            Timestamp = DateTime.UtcNow,
            TotalMemory = GC.GetTotalMemory(forceGC),
            WorkingSet = workingSet,
            PrivateMemory = privateMemory,
            VirtualMemory = virtualMemory,
            Gen0Collections = GC.CollectionCount(0),
            Gen1Collections = GC.CollectionCount(1),
            Gen2Collections = GC.CollectionCount(2),
#if NET5_0_OR_GREATER
            HeapSizeBytes = GC.GetGCMemoryInfo().HeapSizeBytes,
            FragmentedBytes = GC.GetGCMemoryInfo().FragmentedBytes,
            PromotedBytes = GC.GetGCMemoryInfo().PromotedBytes,
            PinnedObjectsCount = GC.GetGCMemoryInfo().PinnedObjectsCount,
            FinalizationPendingCount = GC.GetGCMemoryInfo().FinalizationPendingCount
#else
            HeapSizeBytes = GC.GetTotalMemory(false),
            FragmentedBytes = 0,
            PromotedBytes = 0,
            PinnedObjectsCount = 0,
            FinalizationPendingCount = 0
#endif
        };

        if (_enabled)
        {
            lock (_lock)
            {
                // Enforce maximum history size to prevent unbounded memory growth
                while (_history.Count >= _maxHistorySize)
                {
                    _history.RemoveAt(0);
                }
                _history.Add(snapshot);
            }
        }

        return snapshot;
    }

    /// <summary>
    /// Gets all recorded snapshots.
    /// </summary>
    public static IReadOnlyList<MemorySnapshot> GetHistory()
    {
        lock (_lock)
        {
            return _history.ToList();
        }
    }

    /// <summary>
    /// Creates a memory tracking scope that records before/after snapshots.
    /// </summary>
    public static MemoryScope TrackScope(string label)
    {
        return new MemoryScope(label);
    }

    /// <summary>
    /// Gets the current memory pressure level.
    /// </summary>
    public static MemoryPressureLevel GetPressureLevel()
    {
#if NET5_0_OR_GREATER
        var gcInfo = GC.GetGCMemoryInfo();
        double usagePercent = (double)gcInfo.HeapSizeBytes / gcInfo.TotalAvailableMemoryBytes * 100;

        return usagePercent switch
        {
            < 50 => MemoryPressureLevel.Low,
            < 75 => MemoryPressureLevel.Medium,
            < 90 => MemoryPressureLevel.High,
            _ => MemoryPressureLevel.Critical
        };
#else
        // In .NET Framework, use a simple heuristic based on available physical memory
        var totalMemory = GC.GetTotalMemory(false);
        // Estimate based on typical application memory limits
        double usagePercent = (double)totalMemory / (2L * 1024 * 1024 * 1024) * 100; // Assume 2GB limit

        if (usagePercent < 50) return MemoryPressureLevel.Low;
        if (usagePercent < 75) return MemoryPressureLevel.Medium;
        if (usagePercent < 90) return MemoryPressureLevel.High;
        return MemoryPressureLevel.Critical;
#endif
    }

    /// <summary>
    /// Estimates the memory footprint of a tensor.
    /// </summary>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="elementSize">Size of each element in bytes.</param>
    /// <returns>Estimated memory in bytes.</returns>
    public static long EstimateTensorMemory(int[] shape, int elementSize = 4)
    {
        long elements = 1;
        foreach (int dim in shape)
        {
            elements *= dim;
        }
        return elements * elementSize;
    }

    /// <summary>
    /// Estimates KV-cache memory for a model configuration.
    /// </summary>
    public static long EstimateKVCacheMemory(
        int numLayers,
        int numHeads,
        int headDim,
        int maxSeqLen,
        int batchSize = 1,
        int bytesPerElement = 4)
    {
        // K and V each: [batch, heads, seq, dim]
        long perLayer = (long)batchSize * numHeads * maxSeqLen * headDim * bytesPerElement * 2;
        return perLayer * numLayers;
    }
}

/// <summary>
/// A snapshot of memory usage at a point in time.
/// </summary>
public class MemorySnapshot
{
    public required string Label { get; init; }
    public DateTime Timestamp { get; init; }

    // GC metrics
    public long TotalMemory { get; init; }
    public long HeapSizeBytes { get; init; }
    public long FragmentedBytes { get; init; }
    public long PromotedBytes { get; init; }
    public long PinnedObjectsCount { get; init; }
    public long FinalizationPendingCount { get; init; }

    // Process metrics
    public long WorkingSet { get; init; }
    public long PrivateMemory { get; init; }
    public long VirtualMemory { get; init; }

    // GC collections
    public int Gen0Collections { get; init; }
    public int Gen1Collections { get; init; }
    public int Gen2Collections { get; init; }

    /// <summary>
    /// Compares this snapshot with another.
    /// </summary>
    public MemoryDiff CompareTo(MemorySnapshot baseline)
    {
        return new MemoryDiff
        {
            From = baseline,
            To = this,
            TotalMemoryDelta = TotalMemory - baseline.TotalMemory,
            WorkingSetDelta = WorkingSet - baseline.WorkingSet,
            HeapSizeDelta = HeapSizeBytes - baseline.HeapSizeBytes,
            Gen0CollectionsDelta = Gen0Collections - baseline.Gen0Collections,
            Gen1CollectionsDelta = Gen1Collections - baseline.Gen1Collections,
            Gen2CollectionsDelta = Gen2Collections - baseline.Gen2Collections,
            Duration = Timestamp - baseline.Timestamp
        };
    }

    public override string ToString()
    {
        return $"[{Label}] Total: {FormatBytes(TotalMemory)}, Heap: {FormatBytes(HeapSizeBytes)}, WorkingSet: {FormatBytes(WorkingSet)}";
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
/// Difference between two memory snapshots.
/// </summary>
public class MemoryDiff
{
    public required MemorySnapshot From { get; init; }
    public required MemorySnapshot To { get; init; }
    public long TotalMemoryDelta { get; init; }
    public long WorkingSetDelta { get; init; }
    public long HeapSizeDelta { get; init; }
    public int Gen0CollectionsDelta { get; init; }
    public int Gen1CollectionsDelta { get; init; }
    public int Gen2CollectionsDelta { get; init; }
    public TimeSpan Duration { get; init; }

    /// <summary>
    /// Memory allocation rate in bytes per second.
    /// </summary>
    public double AllocationRatePerSecond =>
        Duration.TotalSeconds > 0 ? TotalMemoryDelta / Duration.TotalSeconds : 0;

    public override string ToString()
    {
        return $"Memory delta: {FormatBytes(TotalMemoryDelta)} in {Duration.TotalMilliseconds:F2}ms " +
               $"(GC: Gen0={Gen0CollectionsDelta}, Gen1={Gen1CollectionsDelta}, Gen2={Gen2CollectionsDelta})";
    }

    private static string FormatBytes(long bytes)
    {
        string sign = bytes >= 0 ? "+" : "";
        string[] suffixes = { "B", "KB", "MB", "GB" };
        int suffixIndex = 0;
        double size = Math.Abs(bytes);

        while (size >= 1024 && suffixIndex < suffixes.Length - 1)
        {
            size /= 1024;
            suffixIndex++;
        }

        return $"{sign}{(bytes >= 0 ? size : -size):F2} {suffixes[suffixIndex]}";
    }
}

/// <summary>
/// Memory pressure levels.
/// </summary>
public enum MemoryPressureLevel
{
    /// <summary>Low memory usage (&lt;50% of available)</summary>
    Low,

    /// <summary>Medium memory usage (50-75% of available)</summary>
    Medium,

    /// <summary>High memory usage (75-90% of available)</summary>
    High,

    /// <summary>Critical memory usage (&gt;90% of available)</summary>
    Critical
}

/// <summary>
/// A scope that automatically captures before/after memory snapshots.
/// </summary>
public readonly struct MemoryScope : IDisposable
{
    private readonly string _label;
    private readonly MemorySnapshot _before;
    private readonly ProfilerSession? _profilerSession;

    /// <summary>
    /// Creates a memory tracking scope.
    /// </summary>
    /// <param name="label">Label for this scope.</param>
    /// <param name="profilerSession">Optional profiler session to record allocations to.</param>
    public MemoryScope(string label, ProfilerSession? profilerSession = null)
    {
        _label = label;
        _profilerSession = profilerSession;
        _before = MemoryTracker.Snapshot($"{label}_before");
    }

    /// <summary>
    /// Gets the before snapshot.
    /// </summary>
    public MemorySnapshot Before => _before;

    public void Dispose()
    {
        var after = MemoryTracker.Snapshot($"{_label}_after");
        var diff = after.CompareTo(_before);

        // Record to profiler session if provided and enabled
        if (_profilerSession?.IsEnabled == true && diff.TotalMemoryDelta > 0)
        {
            _profilerSession.RecordAllocation(_label, diff.TotalMemoryDelta);
        }
    }
}
