using System.Collections.Concurrent;
using System.Diagnostics;

namespace AiDotNet.Diagnostics;

/// <summary>
/// Thread-safe performance profiler for ML operations.
/// </summary>
/// <remarks>
/// <para><b>Overview:</b>
/// The Profiler provides comprehensive performance monitoring for machine learning
/// workloads including timing, memory tracking, and hierarchical call analysis.
/// </para>
/// <para><b>Features:</b>
/// - Thread-safe timing collection
/// - Hierarchical call tree tracking
/// - Memory allocation monitoring
/// - Statistical aggregation (min, max, mean, p95, p99)
/// - Profile scope pattern (using blocks)
/// - Export to various formats
/// </para>
/// <para><b>Usage Example:</b>
/// <code>
/// // Enable profiling
/// Profiler.Enable();
///
/// // Profile a region
/// using (Profiler.Scope("Forward Pass"))
/// {
///     model.Forward(input);
/// }
///
/// // Or manual timing
/// var timer = Profiler.Start("Backward Pass");
/// model.Backward(gradient);
/// timer.Stop();
///
/// // Get report
/// var report = Profiler.GetReport();
/// Console.WriteLine(report.ToString());
///
/// // Disable when done
/// Profiler.Disable();
/// </code>
/// </para>
/// </remarks>
public static class Profiler
{
    private static bool _enabled = false;
    private static readonly ConcurrentDictionary<string, ProfilerEntry> _entries = new();
    private static readonly ConcurrentDictionary<int, Stack<ProfilerTimer>> _callStacks = new();
    private static readonly object _lock = new();
    private static DateTime _startTime = DateTime.UtcNow;

    /// <summary>
    /// Gets whether the profiler is currently enabled.
    /// </summary>
    public static bool IsEnabled => _enabled;

    /// <summary>
    /// Enables the profiler. Must be called before profiling starts.
    /// </summary>
    public static void Enable()
    {
        lock (_lock)
        {
            if (!_enabled)
            {
                _enabled = true;
                _startTime = DateTime.UtcNow;
                Console.WriteLine($"Profiler enabled at {_startTime:yyyy-MM-dd HH:mm:ss.fff}");
            }
        }
    }

    /// <summary>
    /// Disables the profiler.
    /// </summary>
    public static void Disable()
    {
        lock (_lock)
        {
            _enabled = false;
        }
    }

    /// <summary>
    /// Resets all collected profiling data.
    /// </summary>
    public static void Reset()
    {
        lock (_lock)
        {
            _entries.Clear();
            _callStacks.Clear();
            _startTime = DateTime.UtcNow;
        }
    }

    /// <summary>
    /// Creates a scoped profiler that automatically records duration.
    /// </summary>
    /// <param name="name">Name of the operation being profiled.</param>
    /// <returns>A disposable scope that stops timing when disposed.</returns>
    public static ProfilerScope Scope(string name)
    {
        return new ProfilerScope(name);
    }

    /// <summary>
    /// Starts a manual profiler timer.
    /// </summary>
    /// <param name="name">Name of the operation being profiled.</param>
    /// <returns>A timer that must be stopped manually.</returns>
    public static ProfilerTimer Start(string name)
    {
        return new ProfilerTimer(name);
    }

    /// <summary>
    /// Records a timing sample for a named operation.
    /// </summary>
    /// <param name="name">Operation name.</param>
    /// <param name="duration">Duration of the operation.</param>
    /// <param name="parentName">Optional parent operation name for hierarchy.</param>
    internal static void RecordTiming(string name, TimeSpan duration, string? parentName = null)
    {
        if (!_enabled) return;

        var entry = _entries.GetOrAdd(name, _ => new ProfilerEntry(name));
        entry.RecordSample(duration.TotalMilliseconds);

        if (parentName != null)
        {
            entry.AddParent(parentName);
        }
    }

    /// <summary>
    /// Records a memory allocation.
    /// </summary>
    /// <param name="name">Operation name.</param>
    /// <param name="bytes">Number of bytes allocated.</param>
    internal static void RecordAllocation(string name, long bytes)
    {
        if (!_enabled) return;

        var entry = _entries.GetOrAdd(name, _ => new ProfilerEntry(name));
        entry.RecordAllocation(bytes);
    }

    /// <summary>
    /// Gets the current call stack for hierarchical tracking.
    /// </summary>
    internal static Stack<ProfilerTimer> GetCallStack()
    {
        int threadId = Environment.CurrentManagedThreadId;
        return _callStacks.GetOrAdd(threadId, _ => new Stack<ProfilerTimer>());
    }

    /// <summary>
    /// Gets a comprehensive profiling report.
    /// </summary>
    /// <returns>A ProfileReport containing all collected data.</returns>
    public static ProfileReport GetReport()
    {
        var entries = _entries.Values.ToList();
        var runtime = DateTime.UtcNow - _startTime;

        return new ProfileReport(entries, runtime, _startTime);
    }

    /// <summary>
    /// Gets a summary string of profiling results.
    /// </summary>
    public static string GetSummary()
    {
        return GetReport().ToString();
    }

    /// <summary>
    /// Gets timing statistics for a specific operation.
    /// </summary>
    /// <param name="name">Operation name.</param>
    /// <returns>Statistics or null if not found.</returns>
    public static ProfilerStats? GetStats(string name)
    {
        if (_entries.TryGetValue(name, out var entry))
        {
            return entry.GetStats();
        }
        return null;
    }
}

/// <summary>
/// A single profiler entry tracking an operation's performance.
/// </summary>
public class ProfilerEntry
{
    private readonly string _name;
    private readonly List<double> _samples = new();
    private readonly HashSet<string> _parents = new();
    private readonly object _lock = new();
    private long _totalAllocations;
    private int _allocationCount;

    public string Name => _name;
    public int SampleCount { get { lock (_lock) return _samples.Count; } }

    internal ProfilerEntry(string name)
    {
        _name = name;
    }

    internal void RecordSample(double milliseconds)
    {
        lock (_lock)
        {
            _samples.Add(milliseconds);
        }
    }

    internal void RecordAllocation(long bytes)
    {
        lock (_lock)
        {
            _totalAllocations += bytes;
            _allocationCount++;
        }
    }

    internal void AddParent(string parentName)
    {
        lock (_lock)
        {
            _parents.Add(parentName);
        }
    }

    public ProfilerStats GetStats()
    {
        lock (_lock)
        {
            if (_samples.Count == 0)
            {
                return new ProfilerStats
                {
                    Name = _name,
                    Count = 0,
                    TotalMs = 0,
                    MinMs = 0,
                    MaxMs = 0,
                    MeanMs = 0,
                    P50Ms = 0,
                    P95Ms = 0,
                    P99Ms = 0,
                    StdDevMs = 0,
                    TotalAllocations = _totalAllocations,
                    AllocationCount = _allocationCount,
                    Parents = _parents.ToList()
                };
            }

            var sorted = _samples.OrderBy(x => x).ToList();
            double sum = sorted.Sum();
            double mean = sum / sorted.Count;
            double variance = sorted.Sum(x => (x - mean) * (x - mean)) / sorted.Count;

            return new ProfilerStats
            {
                Name = _name,
                Count = sorted.Count,
                TotalMs = sum,
                MinMs = sorted[0],
                MaxMs = sorted[^1],
                MeanMs = mean,
                P50Ms = GetPercentile(sorted, 0.50),
                P95Ms = GetPercentile(sorted, 0.95),
                P99Ms = GetPercentile(sorted, 0.99),
                StdDevMs = Math.Sqrt(variance),
                TotalAllocations = _totalAllocations,
                AllocationCount = _allocationCount,
                Parents = _parents.ToList()
            };
        }
    }

    private static double GetPercentile(List<double> sorted, double percentile)
    {
        if (sorted.Count == 0) return 0;
        if (sorted.Count == 1) return sorted[0];

        double index = percentile * (sorted.Count - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper) return sorted[lower];

        double fraction = index - lower;
        return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
    }
}

/// <summary>
/// Statistics for a profiled operation.
/// </summary>
public class ProfilerStats
{
    public required string Name { get; init; }
    public int Count { get; init; }
    public double TotalMs { get; init; }
    public double MinMs { get; init; }
    public double MaxMs { get; init; }
    public double MeanMs { get; init; }
    public double P50Ms { get; init; }
    public double P95Ms { get; init; }
    public double P99Ms { get; init; }
    public double StdDevMs { get; init; }
    public long TotalAllocations { get; init; }
    public int AllocationCount { get; init; }
    public List<string> Parents { get; init; } = new();

    /// <summary>
    /// Gets operations per second based on mean time.
    /// </summary>
    public double OpsPerSecond => MeanMs > 0 ? 1000.0 / MeanMs : 0;

    public override string ToString()
    {
        return $"{Name}: {Count} calls, mean={MeanMs:F3}ms, p95={P95Ms:F3}ms, total={TotalMs:F1}ms";
    }
}

/// <summary>
/// A manual profiler timer that must be explicitly stopped.
/// </summary>
public class ProfilerTimer : IDisposable
{
    private readonly string _name;
    private readonly Stopwatch _stopwatch;
    private readonly string? _parentName;
    private bool _stopped;

    internal ProfilerTimer(string name)
    {
        _name = name;
        _stopwatch = Stopwatch.StartNew();
        _stopped = false;

        var stack = Profiler.GetCallStack();
        _parentName = stack.Count > 0 ? stack.Peek()._name : null;
        stack.Push(this);
    }

    /// <summary>
    /// Stops the timer and records the duration.
    /// </summary>
    public void Stop()
    {
        if (_stopped) return;
        _stopped = true;

        _stopwatch.Stop();
        Profiler.RecordTiming(_name, _stopwatch.Elapsed, _parentName);

        var stack = Profiler.GetCallStack();
        if (stack.Count > 0 && stack.Peek() == this)
        {
            stack.Pop();
        }
    }

    /// <summary>
    /// Gets the elapsed time so far.
    /// </summary>
    public TimeSpan Elapsed => _stopwatch.Elapsed;

    public void Dispose()
    {
        Stop();
    }
}
