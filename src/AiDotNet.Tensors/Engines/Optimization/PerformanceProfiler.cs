using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;

namespace AiDotNet.Tensors.Engines.Optimization
{
    /// <summary>
    /// Thread-safe performance profiler for tracking operation timings and statistics.
    /// Use this to measure and optimize tensor operations.
    /// </summary>
    public sealed class PerformanceProfiler
    {
        private static readonly Lazy<PerformanceProfiler> _instance =
            new Lazy<PerformanceProfiler>(() => new PerformanceProfiler());

        private readonly ConcurrentDictionary<string, OperationStats> _stats;

        /// <summary>
        /// Gets the singleton instance of the profiler
        /// </summary>
        public static PerformanceProfiler Instance => _instance.Value;

        /// <summary>
        /// Enable or disable profiling (disabled by default for production)
        /// </summary>
        public bool Enabled { get; set; }

        private PerformanceProfiler()
        {
            _stats = new ConcurrentDictionary<string, OperationStats>();
            Enabled = false;
        }

        /// <summary>
        /// Starts profiling an operation
        /// </summary>
        public IDisposable Profile(string operationName)
        {
            if (!Enabled)
                return DisposableHelper.Empty;

            return new ProfileScope(this, operationName);
        }

        /// <summary>
        /// Records a completed operation
        /// </summary>
        internal void RecordOperation(string operationName, long elapsedTicks, long memoryBytes = 0)
        {
            if (!Enabled)
                return;

            var updated = _stats.AddOrUpdate(
                operationName,
                _ => new OperationStats
                {
                    OperationName = operationName,
                    CallCount = 1,
                    TotalTicks = elapsedTicks,
                    MinTicks = elapsedTicks,
                    MaxTicks = elapsedTicks,
                    TotalMemoryBytes = memoryBytes
                },
                (_, existing) =>
                {
                    // Return new object to ensure thread-safety (avoid mutating existing object)
                    return new OperationStats
                    {
                        OperationName = existing.OperationName,
                        CallCount = existing.CallCount + 1,
                        TotalTicks = existing.TotalTicks + elapsedTicks,
                        MinTicks = Math.Min(existing.MinTicks, elapsedTicks),
                        MaxTicks = Math.Max(existing.MaxTicks, elapsedTicks),
                        TotalMemoryBytes = existing.TotalMemoryBytes + memoryBytes
                    };
                });

            _ = updated.CallCount;
        }

        /// <summary>
        /// Gets statistics for a specific operation
        /// </summary>
        public OperationStats? GetStats(string operationName)
        {
            return _stats.TryGetValue(operationName, out var stats) ? stats : null;
        }

        /// <summary>
        /// Gets all recorded statistics
        /// </summary>
        public OperationStats[] GetAllStats()
        {
            return _stats.Values.OrderByDescending(s => s.TotalMilliseconds).ToArray();
        }

        /// <summary>
        /// Clears all statistics
        /// </summary>
        public void Clear()
        {
            _stats.Clear();
        }

        /// <summary>
        /// Generates a performance report
        /// </summary>
        public string GenerateReport()
        {
            var stats = GetAllStats();
            if (stats.Length == 0)
                return "No profiling data available.";

            var report = new System.Text.StringBuilder();
            report.AppendLine("=== Performance Profile Report ===");
            report.AppendLine();
            report.AppendLine($"{"Operation",-40} {"Calls",10} {"Total (ms)",12} {"Avg (ms)",12} {"Min (ms)",12} {"Max (ms)",12} {"Memory (MB)",12}");
            report.AppendLine(new string('-', 120));

            foreach (var stat in stats)
            {
                report.AppendLine($"{stat.OperationName,-40} {stat.CallCount,10} {stat.TotalMilliseconds,12:F3} " +
                                $"{stat.AverageMilliseconds,12:F3} {stat.MinMilliseconds,12:F3} " +
                                $"{stat.MaxMilliseconds,12:F3} {stat.TotalMemoryMB,12:F2}");
            }

            report.AppendLine();
            report.AppendLine($"Total operations: {stats.Length}");
            report.AppendLine($"Total time: {stats.Sum(s => s.TotalMilliseconds):F3} ms");

            return report.ToString();
        }

        private class ProfileScope : IDisposable
        {
            private readonly PerformanceProfiler _profiler;
            private readonly string _operationName;
            private readonly Stopwatch _stopwatch;
            private readonly long _startMemory;

            public ProfileScope(PerformanceProfiler profiler, string operationName)
            {
                _profiler = profiler;
                _operationName = operationName;
#if NET6_0_OR_GREATER
                // Use per-thread allocation tracking for more accurate measurements
                _startMemory = GC.GetAllocatedBytesForCurrentThread();
#else
                // Fallback for .NET Framework - less accurate but functional
                _startMemory = GC.GetTotalMemory(false);
#endif
                _stopwatch = Stopwatch.StartNew();
            }

            public void Dispose()
            {
                _stopwatch.Stop();
#if NET6_0_OR_GREATER
                long endMemory = GC.GetAllocatedBytesForCurrentThread();
#else
                long endMemory = GC.GetTotalMemory(false);
#endif
                // Only report positive memory delta (allocation), ignore GC effects
                long memoryDelta = Math.Max(0, endMemory - _startMemory);

                _profiler.RecordOperation(_operationName, _stopwatch.ElapsedTicks, memoryDelta);
            }
        }

        private static class DisposableHelper
        {
            public static readonly IDisposable Empty = new EmptyDisposable();

            private class EmptyDisposable : IDisposable
            {
                public void Dispose() { }
            }
        }
    }

    /// <summary>
    /// Statistics for a profiled operation
    /// </summary>
    public class OperationStats
    {
        public string OperationName { get; set; } = string.Empty;
        public long CallCount { get; set; }
        public long TotalTicks { get; set; }
        public long MinTicks { get; set; }
        public long MaxTicks { get; set; }
        public long TotalMemoryBytes { get; set; }

        public double TotalMilliseconds => TotalTicks * 1000.0 / Stopwatch.Frequency;
        public double AverageMilliseconds => CallCount > 0 ? TotalMilliseconds / CallCount : 0;
        public double MinMilliseconds => MinTicks * 1000.0 / Stopwatch.Frequency;
        public double MaxMilliseconds => MaxTicks * 1000.0 / Stopwatch.Frequency;
        public double TotalMemoryMB => TotalMemoryBytes / (1024.0 * 1024.0);
        public double AverageMemoryMB => CallCount > 0 ? TotalMemoryMB / CallCount : 0;

        public double ThroughputOpsPerSecond => TotalMilliseconds > 0 ? CallCount / (TotalMilliseconds / 1000.0) : 0;
    }
}
