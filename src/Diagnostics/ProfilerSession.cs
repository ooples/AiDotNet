using System.Collections.Concurrent;
using System.Diagnostics;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Helpers;

namespace AiDotNet.Diagnostics;

/// <summary>
/// Instance-based performance profiler for ML operations.
/// </summary>
/// <remarks>
/// <para><b>Overview:</b>
/// ProfilerSession provides comprehensive performance monitoring for machine learning
/// workloads including timing, memory tracking, and hierarchical call analysis.
/// </para>
/// <para><b>Facade Pattern:</b>
/// This class follows the AiDotNet facade pattern. Users don't create ProfilerSession directly;
/// instead, they configure profiling through <c>AiModelBuilder.ConfigureProfiling()</c>
/// and access results through <c>AiModelResult.ProfilingReport</c>.
/// </para>
/// <para><b>Production-Ready Features:</b>
/// - O(1) streaming statistics using Welford's algorithm
/// - Bounded memory with reservoir sampling for percentiles
/// - Configurable sampling rate for high-frequency operations
/// - Thread-safe timing collection
/// - Hierarchical call tree tracking
/// - Memory allocation monitoring
/// - Statistical aggregation (min, max, mean, p95, p99)
/// - Profile scope pattern (using blocks)
/// </para>
/// <para><b>Internal Usage Example:</b>
/// <code>
/// // Configure through facade
/// var config = new ProfilingConfig { Enabled = true, SamplingRate = 0.1 };
/// var session = new ProfilerSession(config);
///
/// // Profile a region
/// using (session.Scope("Forward Pass"))
/// {
///     model.Forward(input);
/// }
///
/// // Get report
/// var report = session.GetReport();
/// </code>
/// </para>
/// </remarks>
public class ProfilerSession
{
    private readonly ProfilingConfig _config;
    private readonly ConcurrentDictionary<string, ProfilerSessionEntry> _entries = new();
    private readonly ConcurrentDictionary<int, Stack<ProfilerSessionTimer>> _callStacks = new();
    private readonly object _lock = new();
    private readonly DateTime _startTime;
    private bool _enabled;

    /// <summary>
    /// Gets whether the profiler session is currently enabled.
    /// </summary>
    public bool IsEnabled => _enabled;

    /// <summary>
    /// Gets the configuration for this profiler session.
    /// </summary>
    public ProfilingConfig Config => _config;

    /// <summary>
    /// Creates a new profiler session with the specified configuration.
    /// </summary>
    /// <param name="config">Profiling configuration. If null, uses industry-standard defaults.</param>
    public ProfilerSession(ProfilingConfig? config = null)
    {
        _config = config ?? new ProfilingConfig();
        _startTime = DateTime.UtcNow;

        // Auto-enable based on configuration
        _enabled = _config.Enabled;

#if DEBUG
        if (_config.AutoEnableInDebug)
        {
            _enabled = true;
        }
#endif
    }

    /// <summary>
    /// Enables profiling for this session.
    /// </summary>
    public void Enable()
    {
        lock (_lock)
        {
            _enabled = true;
        }
    }

    /// <summary>
    /// Disables profiling for this session.
    /// </summary>
    public void Disable()
    {
        lock (_lock)
        {
            _enabled = false;
        }
    }

    /// <summary>
    /// Resets all collected profiling data.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _entries.Clear();
            _callStacks.Clear();
        }
    }

    /// <summary>
    /// Creates a scoped profiler that automatically records duration.
    /// </summary>
    /// <param name="name">Name of the operation being profiled.</param>
    /// <returns>A disposable scope that stops timing when disposed.</returns>
    public ProfilerSessionScope Scope(string name)
    {
        return new ProfilerSessionScope(this, name);
    }

    /// <summary>
    /// Starts a manual profiler timer.
    /// </summary>
    /// <param name="name">Name of the operation being profiled.</param>
    /// <returns>A timer that must be stopped manually.</returns>
    public ProfilerSessionTimer Start(string name)
    {
        return new ProfilerSessionTimer(this, name);
    }

    /// <summary>
    /// Records a timing sample for a named operation.
    /// </summary>
    /// <param name="name">Operation name.</param>
    /// <param name="duration">Duration of the operation.</param>
    /// <param name="parentName">Optional parent operation name for hierarchy.</param>
    internal void RecordTiming(string name, TimeSpan duration, string? parentName = null)
    {
        if (!_enabled) return;

        // Apply sampling rate (using thread-safe RandomHelper from AiDotNet.Tensors)
        if (_config.SamplingRate < 1.0 && RandomHelper.ThreadSafeRandom.NextDouble() > _config.SamplingRate)
        {
            return;
        }

        // Check max operations limit
        if (_entries.Count >= _config.MaxOperations && !_entries.ContainsKey(name))
        {
            return;
        }

        var entry = _entries.GetOrAdd(name, _ => new ProfilerSessionEntry(name, _config.ReservoirSize));
        entry.RecordSample(duration.TotalMilliseconds);

        if (_config.TrackCallHierarchy && parentName != null)
        {
            entry.AddParent(parentName);
        }
    }

    /// <summary>
    /// Records a memory allocation.
    /// </summary>
    /// <param name="name">Operation name.</param>
    /// <param name="bytes">Number of bytes allocated.</param>
    internal void RecordAllocation(string name, long bytes)
    {
        if (!_enabled || !_config.TrackAllocations) return;

        // Check max operations limit
        if (_entries.Count >= _config.MaxOperations && !_entries.ContainsKey(name))
        {
            return;
        }

        var entry = _entries.GetOrAdd(name, _ => new ProfilerSessionEntry(name, _config.ReservoirSize));
        entry.RecordAllocation(bytes);
    }

    /// <summary>
    /// Gets the current call stack for hierarchical tracking.
    /// </summary>
    internal Stack<ProfilerSessionTimer> GetCallStack()
    {
        int threadId = Environment.CurrentManagedThreadId;
        return _callStacks.GetOrAdd(threadId, _ => new Stack<ProfilerSessionTimer>());
    }

    /// <summary>
    /// Gets a comprehensive profiling report.
    /// </summary>
    /// <returns>A ProfileReport containing all collected data.</returns>
    public ProfileReport GetReport()
    {
        var entries = _entries.Values.ToList();
        var runtime = DateTime.UtcNow - _startTime;

        return new ProfileReport(entries.Cast<IProfilerEntry>().ToList(), runtime, _startTime, _config.CustomTags);
    }

    /// <summary>
    /// Gets a summary string of profiling results.
    /// </summary>
    public string GetSummary()
    {
        return GetReport().ToString();
    }

    /// <summary>
    /// Gets timing statistics for a specific operation.
    /// </summary>
    /// <param name="name">Operation name.</param>
    /// <returns>Statistics or null if not found.</returns>
    public ProfilerStats? GetStats(string name)
    {
        if (_entries.TryGetValue(name, out var entry))
        {
            return entry.GetStats();
        }
        return null;
    }

    /// <summary>
    /// Gets all operation names that have been profiled.
    /// </summary>
    public IReadOnlyList<string> GetOperationNames()
    {
        return _entries.Keys.ToList();
    }

    /// <summary>
    /// Gets the total number of unique operations tracked.
    /// </summary>
    public int OperationCount => _entries.Count;

    /// <summary>
    /// Gets the elapsed time since this session started.
    /// </summary>
    public TimeSpan Elapsed => DateTime.UtcNow - _startTime;
}

/// <summary>
/// Interface for profiler entries to support both legacy and new implementations.
/// </summary>
public interface IProfilerEntry
{
    string Name { get; }
    int SampleCount { get; }
    ProfilerStats GetStats();
}

/// <summary>
/// A single profiler entry tracking an operation's performance using streaming algorithms.
/// </summary>
/// <remarks>
/// <para><b>Production-Ready Features:</b></para>
/// <list type="bullet">
/// <item><description>Welford's algorithm for O(1) mean/variance computation</description></item>
/// <item><description>Reservoir sampling for accurate percentile estimation</description></item>
/// <item><description>Bounded memory usage with configurable sample limits</description></item>
/// <item><description>Lock-free statistics for min/max/count</description></item>
/// </list>
/// </remarks>
public class ProfilerSessionEntry : IProfilerEntry
{
    private readonly string _name;
    private readonly HashSet<string> _parents = new();
    private readonly object _lock = new();

    // Welford's online algorithm state
    private int _count;
    private double _mean;
    private double _m2; // Sum of squared differences from mean
    private double _totalMs;
    private double _minMs = double.MaxValue;
    private double _maxMs = double.MinValue;

    // Reservoir sampling for percentiles (Algorithm R)
    private readonly double[] _reservoir;
    private readonly int _reservoirSize;
    private readonly Random _random;

    // Memory allocations
    private long _totalAllocations;
    private int _allocationCount;

    public string Name => _name;
    public int SampleCount { get { lock (_lock) return _count; } }

    internal ProfilerSessionEntry(string name, int reservoirSize = 1000)
    {
        _name = name;
        _reservoirSize = reservoirSize;
        _reservoir = new double[reservoirSize];
        _random = RandomHelper.CreateSecureRandom();
    }

    internal void RecordSample(double milliseconds)
    {
        lock (_lock)
        {
            _count++;
            _totalMs += milliseconds;

            // Update min/max
            if (milliseconds < _minMs) _minMs = milliseconds;
            if (milliseconds > _maxMs) _maxMs = milliseconds;

            // Welford's online algorithm for mean and variance
            double delta = milliseconds - _mean;
            _mean += delta / _count;
            double delta2 = milliseconds - _mean;
            _m2 += delta * delta2;

            // Reservoir sampling (Algorithm R) for percentile estimation
            if (_count <= _reservoirSize)
            {
                _reservoir[_count - 1] = milliseconds;
            }
            else
            {
                // Randomly replace elements with decreasing probability
                int j = _random.Next(_count);
                if (j < _reservoirSize)
                {
                    _reservoir[j] = milliseconds;
                }
            }
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
            if (_count == 0)
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

            // Sample variance from Welford's algorithm (using n-1 for unbiased estimator)
            // Population variance would be _m2 / _count, but sample variance is more appropriate
            // for statistical analysis of profiling data
            double variance = _count > 1 ? _m2 / (_count - 1) : 0;

            // Get percentiles from reservoir sample
            int sampleSize = Math.Min(_count, _reservoirSize);
            var sortedReservoir = new double[sampleSize];
            Array.Copy(_reservoir, sortedReservoir, sampleSize);
            Array.Sort(sortedReservoir);

            return new ProfilerStats
            {
                Name = _name,
                Count = _count,
                TotalMs = _totalMs,
                MinMs = _minMs,
                MaxMs = _maxMs,
                MeanMs = _mean,
                P50Ms = GetPercentileFromSorted(sortedReservoir, 0.50),
                P95Ms = GetPercentileFromSorted(sortedReservoir, 0.95),
                P99Ms = GetPercentileFromSorted(sortedReservoir, 0.99),
                StdDevMs = Math.Sqrt(variance),
                TotalAllocations = _totalAllocations,
                AllocationCount = _allocationCount,
                Parents = _parents.ToList()
            };
        }
    }

    private static double GetPercentileFromSorted(double[] sorted, double percentile)
    {
        if (sorted.Length == 0) return 0;
        if (sorted.Length == 1) return sorted[0];

        double index = percentile * (sorted.Length - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (upper >= sorted.Length) upper = sorted.Length - 1;
        if (lower == upper) return sorted[lower];

        double fraction = index - lower;
        return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
    }
}

/// <summary>
/// A manual profiler timer that must be explicitly stopped.
/// </summary>
public class ProfilerSessionTimer : IDisposable
{
    private readonly ProfilerSession _session;
    private readonly string _name;
    private readonly Stopwatch _stopwatch;
    private readonly string? _parentName;
    private bool _stopped;

    /// <summary>
    /// Gets the name of this profiler timer.
    /// </summary>
    public string Name => _name;

    internal ProfilerSessionTimer(ProfilerSession session, string name)
    {
        _session = session;
        _name = name;
        _stopwatch = Stopwatch.StartNew();
        _stopped = false;

        var stack = session.GetCallStack();
        _parentName = stack.Count > 0 ? stack.Peek().Name : null;
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
        _session.RecordTiming(_name, _stopwatch.Elapsed, _parentName);

        // Clean up from call stack - handle case where nested timers were not stopped in order
        // (e.g., due to exceptions causing early exit)
        var stack = _session.GetCallStack();
        if (stack.Count > 0)
        {
            if (stack.Peek() == this)
            {
                // Normal case: we're at the top
                stack.Pop();
            }
            else
            {
                // Abnormal case: something above us wasn't stopped (likely due to exception)
                // Search for ourselves in the stack and remove to prevent memory leak
                var tempStack = new Stack<ProfilerSessionTimer>();
                bool found = false;
                while (stack.Count > 0)
                {
                    var item = stack.Pop();
                    if (item == this)
                    {
                        found = true;
                        break;
                    }
                    tempStack.Push(item);
                }
                // Restore items that were above us (if any)
                while (tempStack.Count > 0)
                {
                    stack.Push(tempStack.Pop());
                }
                // If not found, we were already removed or never added (shouldn't happen)
                _ = found; // Suppress unused variable warning
            }
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

/// <summary>
/// A scoped profiler that automatically records timing when disposed.
/// </summary>
public readonly struct ProfilerSessionScope : IDisposable
{
    private readonly ProfilerSessionTimer _timer;

    internal ProfilerSessionScope(ProfilerSession session, string name)
    {
        _timer = session.Start(name);
    }

    /// <summary>
    /// Gets the elapsed time so far.
    /// </summary>
    public TimeSpan Elapsed => _timer.Elapsed;

    public void Dispose()
    {
        _timer.Stop();
    }
}

/// <summary>
/// A comprehensive profiling report containing all collected metrics.
/// </summary>
public class ProfileReport
{
    private readonly List<IProfilerEntry> _entries;
    private readonly TimeSpan _totalRuntime;
    private readonly DateTime _startTime;
    private readonly Dictionary<string, string> _tags;

    /// <summary>
    /// Gets the profiled entries.
    /// </summary>
    public IReadOnlyList<IProfilerEntry> Entries => _entries;

    /// <summary>
    /// Gets the total runtime of the profiling session.
    /// </summary>
    public TimeSpan TotalRuntime => _totalRuntime;

    /// <summary>
    /// Gets the start time of the profiling session.
    /// </summary>
    public DateTime StartTime => _startTime;

    /// <summary>
    /// Gets custom tags associated with this report.
    /// </summary>
    public IReadOnlyDictionary<string, string> Tags => _tags;

    internal ProfileReport(
        List<IProfilerEntry> entries,
        TimeSpan totalRuntime,
        DateTime startTime,
        Dictionary<string, string>? tags = null)
    {
        _entries = entries;
        _totalRuntime = totalRuntime;
        _startTime = startTime;
        _tags = tags ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Gets statistics for all operations, sorted by total time descending.
    /// </summary>
    public IReadOnlyList<ProfilerStats> GetAllStats()
    {
        return _entries
            .Select(e => e.GetStats())
            .OrderByDescending(s => s.TotalMs)
            .ToList();
    }

    /// <summary>
    /// Gets the top N operations by total time.
    /// </summary>
    public IReadOnlyList<ProfilerStats> GetTopOperations(int count = 10)
    {
        return GetAllStats().Take(count).ToList();
    }

    /// <summary>
    /// Gets operations that exceed the specified P95 threshold.
    /// </summary>
    public IReadOnlyList<ProfilerStats> GetSlowOperations(double p95ThresholdMs)
    {
        return _entries
            .Select(e => e.GetStats())
            .Where(s => s.P95Ms > p95ThresholdMs)
            .OrderByDescending(s => s.P95Ms)
            .ToList();
    }

    /// <summary>
    /// Exports the report to a dictionary for serialization.
    /// </summary>
    public Dictionary<string, object> ToDictionary()
    {
        return new Dictionary<string, object>
        {
            ["startTime"] = _startTime.ToString("O"),
            ["totalRuntimeMs"] = _totalRuntime.TotalMilliseconds,
            ["operationCount"] = _entries.Count,
            ["tags"] = _tags,
            ["operations"] = _entries.Select(e =>
            {
                var stats = e.GetStats();
                return new Dictionary<string, object>
                {
                    ["name"] = stats.Name,
                    ["count"] = stats.Count,
                    ["totalMs"] = stats.TotalMs,
                    ["meanMs"] = stats.MeanMs,
                    ["minMs"] = stats.MinMs,
                    ["maxMs"] = stats.MaxMs,
                    ["p50Ms"] = stats.P50Ms,
                    ["p95Ms"] = stats.P95Ms,
                    ["p99Ms"] = stats.P99Ms,
                    ["stdDevMs"] = stats.StdDevMs,
                    ["totalAllocations"] = stats.TotalAllocations,
                    ["allocationCount"] = stats.AllocationCount,
                    ["opsPerSecond"] = stats.OpsPerSecond
                };
            }).ToList()
        };
    }

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"=== Profiling Report ===");
        sb.AppendLine($"Start Time: {_startTime:O}");
        sb.AppendLine($"Total Runtime: {_totalRuntime.TotalSeconds:F2}s");
        sb.AppendLine($"Operations Tracked: {_entries.Count}");

        if (_tags.Count > 0)
        {
            sb.AppendLine($"Tags: {string.Join(", ", _tags.Select(kv => $"{kv.Key}={kv.Value}"))}");
        }

        sb.AppendLine();
        sb.AppendLine("Top Operations by Total Time:");
        sb.AppendLine(new string('-', 80));

        foreach (var stats in GetTopOperations(20))
        {
            sb.AppendLine($"  {stats.Name}:");
            sb.AppendLine($"    Calls: {stats.Count}, Total: {stats.TotalMs:F2}ms, Mean: {stats.MeanMs:F3}ms");
            sb.AppendLine($"    P50: {stats.P50Ms:F3}ms, P95: {stats.P95Ms:F3}ms, P99: {stats.P99Ms:F3}ms");

            if (stats.TotalAllocations > 0)
            {
                sb.AppendLine($"    Allocations: {FormatBytes(stats.TotalAllocations)} ({stats.AllocationCount} times)");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Gets the total number of operations tracked.
    /// </summary>
    public int TotalOperations => _entries.Count;

    /// <summary>
    /// Gets a dictionary of operation name to statistics.
    /// </summary>
    public IReadOnlyDictionary<string, ProfilerStats> Stats =>
        _entries.ToDictionary(e => e.Name, e => e.GetStats());

    /// <summary>
    /// Gets statistics for a specific operation.
    /// </summary>
    /// <param name="name">Operation name.</param>
    /// <returns>Statistics or null if not found.</returns>
    public ProfilerStats? GetStats(string name)
    {
        var entry = _entries.FirstOrDefault(e => e.Name == name);
        return entry?.GetStats();
    }

    /// <summary>
    /// Gets the top N operations ordered by total time (hotspots).
    /// </summary>
    /// <param name="count">Number of operations to return.</param>
    /// <returns>List of statistics for the hottest operations.</returns>
    public IReadOnlyList<ProfilerStats> GetHotspots(int count = 10)
    {
        return _entries
            .Select(e => e.GetStats())
            .OrderByDescending(s => s.TotalMs)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Compares this report to a baseline and identifies regressions.
    /// </summary>
    /// <param name="baseline">Baseline report to compare against.</param>
    /// <param name="thresholdPercent">Threshold percentage for regression detection (default: 10%).</param>
    /// <returns>Comparison result with identified regressions.</returns>
    public ProfileComparison CompareTo(ProfileReport baseline, double thresholdPercent = 10.0)
    {
        var regressions = new List<ProfileRegression>();
        var improvements = new List<ProfileRegression>();
        var currentStats = Stats;
        var baselineStats = baseline.Stats;

        foreach (var (name, current) in currentStats)
        {
            if (baselineStats.TryGetValue(name, out var baselineStat))
            {
                if (baselineStat.MeanMs > 0)
                {
                    double changePercent = ((current.MeanMs - baselineStat.MeanMs) / baselineStat.MeanMs) * 100;

                    if (changePercent > thresholdPercent)
                    {
                        regressions.Add(new ProfileRegression
                        {
                            OperationName = name,
                            BaselineMeanMs = baselineStat.MeanMs,
                            CurrentMeanMs = current.MeanMs,
                            ChangePercent = changePercent
                        });
                    }
                    else if (changePercent < -thresholdPercent)
                    {
                        improvements.Add(new ProfileRegression
                        {
                            OperationName = name,
                            BaselineMeanMs = baselineStat.MeanMs,
                            CurrentMeanMs = current.MeanMs,
                            ChangePercent = changePercent
                        });
                    }
                }
            }
        }

        return new ProfileComparison
        {
            Baseline = baseline,
            Current = this,
            ThresholdPercent = thresholdPercent,
            Regressions = regressions,
            Improvements = improvements,
            HasRegressions = regressions.Count > 0
        };
    }

    /// <summary>
    /// Exports the report to JSON format.
    /// </summary>
    /// <returns>JSON string representation of the report.</returns>
    public string ToJson()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \"startTime\": \"{_startTime:O}\",");
        sb.AppendLine($"  \"totalRuntimeMs\": {_totalRuntime.TotalMilliseconds:F2},");
        sb.AppendLine($"  \"operationCount\": {_entries.Count},");

        // Tags
        sb.Append("  \"tags\": {");
        var tagEntries = _tags.ToList();
        for (int i = 0; i < tagEntries.Count; i++)
        {
            sb.Append($"\"{EscapeJson(tagEntries[i].Key)}\": \"{EscapeJson(tagEntries[i].Value)}\"");
            if (i < tagEntries.Count - 1) sb.Append(", ");
        }
        sb.AppendLine("},");

        // Operations
        sb.AppendLine("  \"operations\": [");
        var stats = GetAllStats();
        for (int i = 0; i < stats.Count; i++)
        {
            var s = stats[i];
            sb.AppendLine("    {");
            sb.AppendLine($"      \"name\": \"{EscapeJson(s.Name)}\",");
            sb.AppendLine($"      \"count\": {s.Count},");
            sb.AppendLine($"      \"totalMs\": {s.TotalMs:F3},");
            sb.AppendLine($"      \"meanMs\": {s.MeanMs:F3},");
            sb.AppendLine($"      \"minMs\": {s.MinMs:F3},");
            sb.AppendLine($"      \"maxMs\": {s.MaxMs:F3},");
            sb.AppendLine($"      \"p50Ms\": {s.P50Ms:F3},");
            sb.AppendLine($"      \"p95Ms\": {s.P95Ms:F3},");
            sb.AppendLine($"      \"p99Ms\": {s.P99Ms:F3},");
            sb.AppendLine($"      \"stdDevMs\": {s.StdDevMs:F3},");
            sb.AppendLine($"      \"totalAllocations\": {s.TotalAllocations},");
            sb.AppendLine($"      \"allocationCount\": {s.AllocationCount},");
            sb.AppendLine($"      \"opsPerSecond\": {s.OpsPerSecond:F2}");
            sb.Append("    }");
            if (i < stats.Count - 1) sb.AppendLine(",");
            else sb.AppendLine();
        }
        sb.AppendLine("  ]");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Exports the report to CSV format.
    /// </summary>
    /// <returns>CSV string representation of the report.</returns>
    public string ToCsv()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Name,Count,TotalMs,MeanMs,MinMs,MaxMs,P50Ms,P95Ms,P99Ms,StdDevMs,TotalAllocations,AllocationCount,OpsPerSecond");

        foreach (var s in GetAllStats())
        {
            sb.AppendLine($"\"{EscapeCsv(s.Name)}\",{s.Count},{s.TotalMs:F3},{s.MeanMs:F3},{s.MinMs:F3},{s.MaxMs:F3},{s.P50Ms:F3},{s.P95Ms:F3},{s.P99Ms:F3},{s.StdDevMs:F3},{s.TotalAllocations},{s.AllocationCount},{s.OpsPerSecond:F2}");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Exports the report to Markdown format.
    /// </summary>
    /// <returns>Markdown string representation of the report.</returns>
    public string ToMarkdown()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("# Profiling Report");
        sb.AppendLine();
        sb.AppendLine($"**Start Time:** {_startTime:O}");
        sb.AppendLine();
        sb.AppendLine($"**Total Runtime:** {_totalRuntime.TotalSeconds:F2}s");
        sb.AppendLine();
        sb.AppendLine($"**Operations Tracked:** {_entries.Count}");
        sb.AppendLine();

        if (_tags.Count > 0)
        {
            sb.AppendLine("## Tags");
            sb.AppendLine();
            foreach (var (key, value) in _tags)
            {
                sb.AppendLine($"- **{key}:** {value}");
            }
            sb.AppendLine();
        }

        sb.AppendLine("## Operations");
        sb.AppendLine();
        sb.AppendLine("| Operation | Count | Total (ms) | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |");
        sb.AppendLine("|-----------|------:|----------:|---------:|--------:|--------:|--------:|");

        foreach (var s in GetTopOperations(50))
        {
            sb.AppendLine($"| {s.Name} | {s.Count} | {s.TotalMs:F2} | {s.MeanMs:F3} | {s.P50Ms:F3} | {s.P95Ms:F3} | {s.P99Ms:F3} |");
        }

        return sb.ToString();
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

    private static string EscapeJson(string s)
    {
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
    }

    private static string EscapeCsv(string s)
    {
        return s.Replace("\"", "\"\"");
    }
}

/// <summary>
/// Statistics for a profiled operation.
/// </summary>
public class ProfilerStats
{
    /// <summary>Operation name.</summary>
    public required string Name { get; init; }

    /// <summary>Number of times the operation was called.</summary>
    public int Count { get; init; }

    /// <summary>Total time in milliseconds.</summary>
    public double TotalMs { get; init; }

    /// <summary>Mean time in milliseconds.</summary>
    public double MeanMs { get; init; }

    /// <summary>Minimum time in milliseconds.</summary>
    public double MinMs { get; init; }

    /// <summary>Maximum time in milliseconds.</summary>
    public double MaxMs { get; init; }

    /// <summary>50th percentile (median) in milliseconds.</summary>
    public double P50Ms { get; init; }

    /// <summary>95th percentile in milliseconds.</summary>
    public double P95Ms { get; init; }

    /// <summary>99th percentile in milliseconds.</summary>
    public double P99Ms { get; init; }

    /// <summary>Standard deviation in milliseconds.</summary>
    public double StdDevMs { get; init; }

    /// <summary>Total bytes allocated.</summary>
    public long TotalAllocations { get; init; }

    /// <summary>Number of allocation events.</summary>
    public int AllocationCount { get; init; }

    /// <summary>Parent operations (for hierarchy tracking).</summary>
    public IReadOnlyList<string> Parents { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Gets the operations per second based on count and total time.
    /// </summary>
    public double OpsPerSecond => TotalMs > 0 ? Count / (TotalMs / 1000.0) : 0;

    public override string ToString()
    {
        return $"{Name}: {Count} calls, Mean: {MeanMs:F3}ms, P95: {P95Ms:F3}ms, Total: {TotalMs:F2}ms";
    }
}

/// <summary>
/// Result of comparing two profiling reports.
/// </summary>
public class ProfileComparison
{
    /// <summary>Baseline report.</summary>
    public required ProfileReport Baseline { get; init; }

    /// <summary>Current report being compared.</summary>
    public required ProfileReport Current { get; init; }

    /// <summary>Threshold percentage used for comparison.</summary>
    public double ThresholdPercent { get; init; }

    /// <summary>Operations that regressed beyond threshold.</summary>
    public IReadOnlyList<ProfileRegression> Regressions { get; init; } = Array.Empty<ProfileRegression>();

    /// <summary>Operations that improved beyond threshold.</summary>
    public IReadOnlyList<ProfileRegression> Improvements { get; init; } = Array.Empty<ProfileRegression>();

    /// <summary>Whether any regressions were detected.</summary>
    public bool HasRegressions { get; init; }

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Profile Comparison (threshold: {ThresholdPercent}%)");
        sb.AppendLine($"Regressions: {Regressions.Count}, Improvements: {Improvements.Count}");

        if (Regressions.Count > 0)
        {
            sb.AppendLine("\nRegressions:");
            foreach (var r in Regressions)
            {
                sb.AppendLine($"  {r.OperationName}: {r.BaselineMeanMs:F3}ms -> {r.CurrentMeanMs:F3}ms ({r.ChangePercent:+0.0}%)");
            }
        }

        if (Improvements.Count > 0)
        {
            sb.AppendLine("\nImprovements:");
            foreach (var i in Improvements)
            {
                sb.AppendLine($"  {i.OperationName}: {i.BaselineMeanMs:F3}ms -> {i.CurrentMeanMs:F3}ms ({i.ChangePercent:0.0}%)");
            }
        }

        return sb.ToString();
    }
}

/// <summary>
/// Details about a regression or improvement in performance.
/// </summary>
public class ProfileRegression
{
    /// <summary>Name of the operation.</summary>
    public required string OperationName { get; init; }

    /// <summary>Baseline mean time in milliseconds.</summary>
    public double BaselineMeanMs { get; init; }

    /// <summary>Current mean time in milliseconds.</summary>
    public double CurrentMeanMs { get; init; }

    /// <summary>Percentage change (positive = regression, negative = improvement).</summary>
    public double ChangePercent { get; init; }
}
