using System.Diagnostics;
using System.Runtime.InteropServices;

namespace AiDotNet.TrainingMonitoring.Resources;

/// <summary>
/// Monitors system resources (CPU, memory, GPU) during training.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> The ResourceMonitor tracks hardware utilization
/// to help you understand if your training is bottlenecked by resources.
///
/// Key metrics:
/// - CPU Usage: How much processing power is being used
/// - Memory Usage: RAM consumption and available memory
/// - GPU Usage: Graphics card utilization (if available)
/// - GPU Memory: VRAM consumption (if available)
///
/// Example usage:
/// <code>
/// using var monitor = new ResourceMonitor();
///
/// // Start monitoring with 1-second intervals
/// monitor.Start(TimeSpan.FromSeconds(1));
///
/// // Subscribe to updates
/// monitor.ResourceUpdated += (s, e) =&gt; {
///     Console.WriteLine($"CPU: {e.CpuPercent:F1}%");
///     Console.WriteLine($"Memory: {e.MemoryUsedMB:F0} MB");
/// };
///
/// // Or get current snapshot
/// var snapshot = monitor.GetSnapshot();
///
/// // Stop when done
/// monitor.Stop();
/// </code>
/// </remarks>
public class ResourceMonitor : IDisposable
{
    private readonly object _lock = new();
    private readonly Process _currentProcess;
    private readonly List<ResourceSnapshot> _history;
    private readonly int _maxHistorySize;
    private Timer? _monitorTimer;
    private DateTime _lastCpuTime;
    private TimeSpan _lastTotalProcessorTime;
    private bool _isRunning;
    private bool _isDisposed;

    /// <summary>
    /// Gets whether the monitor is currently running.
    /// </summary>
    public bool IsRunning => _isRunning;

    /// <summary>
    /// Gets or sets whether to attempt GPU monitoring.
    /// </summary>
    public bool MonitorGpu { get; set; } = true;

    /// <summary>
    /// Gets or sets the resource warning thresholds.
    /// </summary>
    public ResourceThresholds Thresholds { get; set; } = new();

    /// <summary>
    /// Event raised when resource metrics are updated.
    /// </summary>
    public event EventHandler<ResourceSnapshot>? ResourceUpdated;

    /// <summary>
    /// Event raised when a resource threshold is exceeded.
    /// </summary>
    public event EventHandler<ResourceWarningEventArgs>? ThresholdExceeded;

    /// <summary>
    /// Initializes a new instance of the ResourceMonitor class.
    /// </summary>
    /// <param name="maxHistorySize">Maximum number of snapshots to keep in history.</param>
    public ResourceMonitor(int maxHistorySize = 3600)
    {
        _maxHistorySize = maxHistorySize;
        _history = new List<ResourceSnapshot>();
        _currentProcess = Process.GetCurrentProcess();

        // Initialize timing for process CPU calculation
        _lastCpuTime = DateTime.UtcNow;
        _lastTotalProcessorTime = _currentProcess.TotalProcessorTime;
    }

    /// <summary>
    /// Starts the resource monitor.
    /// </summary>
    /// <param name="interval">How often to sample resources.</param>
    public void Start(TimeSpan? interval = null)
    {
        if (_isRunning)
            return;

        _isRunning = true;
        var monitorInterval = interval ?? TimeSpan.FromSeconds(1);
        _monitorTimer = new Timer(SampleResources, null, TimeSpan.Zero, monitorInterval);
    }

    /// <summary>
    /// Stops the resource monitor.
    /// </summary>
    public void Stop()
    {
        if (!_isRunning)
            return;

        _isRunning = false;
        _monitorTimer?.Dispose();
        _monitorTimer = null;
    }

    /// <summary>
    /// Gets the current resource snapshot.
    /// </summary>
    /// <returns>Current resource metrics.</returns>
    public ResourceSnapshot GetSnapshot()
    {
        var snapshot = new ResourceSnapshot
        {
            Timestamp = DateTime.UtcNow,
            CpuPercent = GetCpuUsage(),
            ProcessCpuPercent = GetProcessCpuUsage(),
            MemoryUsedMB = GetMemoryUsedMB(),
            MemoryTotalMB = GetTotalMemoryMB(),
            ProcessMemoryMB = GetProcessMemoryMB()
        };

        // Try to get GPU metrics if enabled
        if (MonitorGpu)
        {
            var gpuMetrics = GetGpuMetrics();
            if (gpuMetrics != null)
            {
                snapshot.GpuPercent = gpuMetrics.Value.utilization;
                snapshot.GpuMemoryUsedMB = gpuMetrics.Value.memoryUsedMB;
                snapshot.GpuMemoryTotalMB = gpuMetrics.Value.memoryTotalMB;
                snapshot.GpuAvailable = true;
            }
        }

        return snapshot;
    }

    /// <summary>
    /// Gets the resource history.
    /// </summary>
    /// <param name="limit">Maximum number of entries to return.</param>
    /// <returns>List of historical snapshots.</returns>
    public List<ResourceSnapshot> GetHistory(int? limit = null)
    {
        lock (_lock)
        {
            var result = limit.HasValue
                ? _history.TakeLast(limit.Value).ToList()
                : _history.ToList();
            return result;
        }
    }

    /// <summary>
    /// Gets average resource usage over the history period.
    /// </summary>
    /// <returns>Average metrics.</returns>
    public ResourceSnapshot GetAverage()
    {
        lock (_lock)
        {
            if (_history.Count == 0)
                return new ResourceSnapshot();

            return new ResourceSnapshot
            {
                Timestamp = DateTime.UtcNow,
                CpuPercent = _history.Average(s => s.CpuPercent),
                ProcessCpuPercent = _history.Average(s => s.ProcessCpuPercent),
                MemoryUsedMB = _history.Average(s => s.MemoryUsedMB),
                MemoryTotalMB = _history.LastOrDefault()?.MemoryTotalMB ?? 0,
                ProcessMemoryMB = _history.Average(s => s.ProcessMemoryMB),
                GpuPercent = _history.Where(s => s.GpuAvailable).Select(s => s.GpuPercent).DefaultIfEmpty(0).Average(),
                GpuMemoryUsedMB = _history.Where(s => s.GpuAvailable).Select(s => s.GpuMemoryUsedMB).DefaultIfEmpty(0).Average(),
                GpuMemoryTotalMB = _history.LastOrDefault()?.GpuMemoryTotalMB ?? 0,
                GpuAvailable = _history.Any(s => s.GpuAvailable)
            };
        }
    }

    /// <summary>
    /// Gets peak resource usage.
    /// </summary>
    /// <returns>Peak metrics.</returns>
    public ResourceSnapshot GetPeak()
    {
        lock (_lock)
        {
            if (_history.Count == 0)
                return new ResourceSnapshot();

            return new ResourceSnapshot
            {
                Timestamp = DateTime.UtcNow,
                CpuPercent = _history.Max(s => s.CpuPercent),
                ProcessCpuPercent = _history.Max(s => s.ProcessCpuPercent),
                MemoryUsedMB = _history.Max(s => s.MemoryUsedMB),
                MemoryTotalMB = _history.Max(s => s.MemoryTotalMB),
                ProcessMemoryMB = _history.Max(s => s.ProcessMemoryMB),
                GpuPercent = _history.Where(s => s.GpuAvailable).Select(s => s.GpuPercent).DefaultIfEmpty(0).Max(),
                GpuMemoryUsedMB = _history.Where(s => s.GpuAvailable).Select(s => s.GpuMemoryUsedMB).DefaultIfEmpty(0).Max(),
                GpuMemoryTotalMB = _history.Max(s => s.GpuMemoryTotalMB),
                GpuAvailable = _history.Any(s => s.GpuAvailable)
            };
        }
    }

    /// <summary>
    /// Clears the history.
    /// </summary>
    public void ClearHistory()
    {
        lock (_lock)
        {
            _history.Clear();
        }
    }

    private void SampleResources(object? state)
    {
        try
        {
            var snapshot = GetSnapshot();

            lock (_lock)
            {
                _history.Add(snapshot);

                // Trim history if needed
                while (_history.Count > _maxHistorySize)
                {
                    _history.RemoveAt(0);
                }
            }

            // Raise update event
            ResourceUpdated?.Invoke(this, snapshot);

            // Check thresholds
            CheckThresholds(snapshot);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ResourceMonitor] Error sampling resources: {ex.Message}");
        }
    }

    private void CheckThresholds(ResourceSnapshot snapshot)
    {
        if (Thresholds.CpuWarningPercent > 0 && snapshot.CpuPercent > Thresholds.CpuWarningPercent)
        {
            ThresholdExceeded?.Invoke(this, new ResourceWarningEventArgs
            {
                ResourceName = "CPU",
                CurrentValue = snapshot.CpuPercent,
                ThresholdValue = Thresholds.CpuWarningPercent,
                Snapshot = snapshot
            });
        }

        var memoryPercent = snapshot.MemoryTotalMB > 0 ? (snapshot.MemoryUsedMB / snapshot.MemoryTotalMB) * 100 : 0;
        if (Thresholds.MemoryWarningPercent > 0 && memoryPercent > Thresholds.MemoryWarningPercent)
        {
            ThresholdExceeded?.Invoke(this, new ResourceWarningEventArgs
            {
                ResourceName = "Memory",
                CurrentValue = memoryPercent,
                ThresholdValue = Thresholds.MemoryWarningPercent,
                Snapshot = snapshot
            });
        }

        if (snapshot.GpuAvailable)
        {
            if (Thresholds.GpuWarningPercent > 0 && snapshot.GpuPercent > Thresholds.GpuWarningPercent)
            {
                ThresholdExceeded?.Invoke(this, new ResourceWarningEventArgs
                {
                    ResourceName = "GPU",
                    CurrentValue = snapshot.GpuPercent,
                    ThresholdValue = Thresholds.GpuWarningPercent,
                    Snapshot = snapshot
                });
            }

            var gpuMemoryPercent = snapshot.GpuMemoryTotalMB > 0 ? (snapshot.GpuMemoryUsedMB / snapshot.GpuMemoryTotalMB) * 100 : 0;
            if (Thresholds.GpuMemoryWarningPercent > 0 && gpuMemoryPercent > Thresholds.GpuMemoryWarningPercent)
            {
                ThresholdExceeded?.Invoke(this, new ResourceWarningEventArgs
                {
                    ResourceName = "GPU Memory",
                    CurrentValue = gpuMemoryPercent,
                    ThresholdValue = Thresholds.GpuMemoryWarningPercent,
                    Snapshot = snapshot
                });
            }
        }
    }

    private double GetCpuUsage()
    {
        // Use process CPU as approximation of system CPU
        // This is cross-platform and doesn't require Windows-specific PerformanceCounter
        return GetProcessCpuUsage();
    }

    private double GetProcessCpuUsage()
    {
        try
        {
            var currentTime = DateTime.UtcNow;
            var currentCpuTime = _currentProcess.TotalProcessorTime;

            var cpuUsedMs = (currentCpuTime - _lastTotalProcessorTime).TotalMilliseconds;
            var totalMsPassed = (currentTime - _lastCpuTime).TotalMilliseconds;

            _lastTotalProcessorTime = currentCpuTime;
            _lastCpuTime = currentTime;

            if (totalMsPassed <= 0)
                return 0;

            var cpuPercent = (cpuUsedMs / (Environment.ProcessorCount * totalMsPassed)) * 100;
            return Math.Min(100, Math.Max(0, cpuPercent));
        }
        catch
        {
            return 0;
        }
    }

    private double GetMemoryUsedMB()
    {
        try
        {
            var gcMemory = GC.GetTotalMemory(false);
            return gcMemory / (1024.0 * 1024.0);
        }
        catch
        {
            return 0;
        }
    }

    private double GetTotalMemoryMB()
    {
        try
        {
#if NET6_0_OR_GREATER
            // Use GC to get available memory info (only available in .NET 6+)
            var memInfo = GC.GetGCMemoryInfo();
            return memInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0);
#else
            // Fallback for .NET Framework - estimate based on process info
            // Return 0 to indicate unknown (or use other OS-specific APIs)
            return 0;
#endif
        }
        catch
        {
            return 0;
        }
    }

    private double GetProcessMemoryMB()
    {
        try
        {
            _currentProcess.Refresh();
            return _currentProcess.WorkingSet64 / (1024.0 * 1024.0);
        }
        catch
        {
            return 0;
        }
    }

    private (double utilization, double memoryUsedMB, double memoryTotalMB)? GetGpuMetrics()
    {
        // Try NVIDIA GPU via nvidia-smi
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            try
            {
                var result = TryGetNvidiaMetrics();
                if (result.HasValue)
                    return result;
            }
            catch
            {
                // NVIDIA not available
            }
        }

        return null;
    }

    private (double utilization, double memoryUsedMB, double memoryTotalMB)? TryGetNvidiaMetrics()
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process == null)
                return null;

            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(1000);

            if (process.ExitCode != 0)
                return null;

            var parts = output.Trim().Split(',');
            if (parts.Length >= 3)
            {
                if (double.TryParse(parts[0].Trim(), out var utilization) &&
                    double.TryParse(parts[1].Trim(), out var memoryUsed) &&
                    double.TryParse(parts[2].Trim(), out var memoryTotal))
                {
                    return (utilization, memoryUsed, memoryTotal);
                }
            }
        }
        catch
        {
            // nvidia-smi not available
        }

        return null;
    }

    /// <summary>
    /// Disposes the resource monitor.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;
        Stop();

        _currentProcess.Dispose();
    }
}

/// <summary>
/// A snapshot of resource usage at a point in time.
/// </summary>
public class ResourceSnapshot
{
    /// <summary>
    /// Gets or sets the timestamp of this snapshot.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the system CPU usage percentage.
    /// </summary>
    public double CpuPercent { get; set; }

    /// <summary>
    /// Gets or sets the process CPU usage percentage.
    /// </summary>
    public double ProcessCpuPercent { get; set; }

    /// <summary>
    /// Gets or sets the system memory used in MB.
    /// </summary>
    public double MemoryUsedMB { get; set; }

    /// <summary>
    /// Gets or sets the total system memory in MB.
    /// </summary>
    public double MemoryTotalMB { get; set; }

    /// <summary>
    /// Gets or sets the process memory used in MB.
    /// </summary>
    public double ProcessMemoryMB { get; set; }

    /// <summary>
    /// Gets the memory usage percentage.
    /// </summary>
    public double MemoryPercent => MemoryTotalMB > 0 ? (MemoryUsedMB / MemoryTotalMB) * 100 : 0;

    /// <summary>
    /// Gets or sets whether a GPU is available.
    /// </summary>
    public bool GpuAvailable { get; set; }

    /// <summary>
    /// Gets or sets the GPU utilization percentage.
    /// </summary>
    public double GpuPercent { get; set; }

    /// <summary>
    /// Gets or sets the GPU memory used in MB.
    /// </summary>
    public double GpuMemoryUsedMB { get; set; }

    /// <summary>
    /// Gets or sets the total GPU memory in MB.
    /// </summary>
    public double GpuMemoryTotalMB { get; set; }

    /// <summary>
    /// Gets the GPU memory usage percentage.
    /// </summary>
    public double GpuMemoryPercent => GpuMemoryTotalMB > 0 ? (GpuMemoryUsedMB / GpuMemoryTotalMB) * 100 : 0;

    /// <summary>
    /// Returns a string representation of the snapshot.
    /// </summary>
    public override string ToString()
    {
        var result = $"CPU: {CpuPercent:F1}% | Memory: {MemoryUsedMB:F0}/{MemoryTotalMB:F0} MB ({MemoryPercent:F1}%)";

        if (GpuAvailable)
        {
            result += $" | GPU: {GpuPercent:F1}% | GPU Memory: {GpuMemoryUsedMB:F0}/{GpuMemoryTotalMB:F0} MB ({GpuMemoryPercent:F1}%)";
        }

        return result;
    }
}

/// <summary>
/// Resource warning thresholds.
/// </summary>
public class ResourceThresholds
{
    /// <summary>
    /// Gets or sets the CPU warning threshold (percentage).
    /// </summary>
    public double CpuWarningPercent { get; set; } = 90;

    /// <summary>
    /// Gets or sets the memory warning threshold (percentage).
    /// </summary>
    public double MemoryWarningPercent { get; set; } = 85;

    /// <summary>
    /// Gets or sets the GPU warning threshold (percentage).
    /// </summary>
    public double GpuWarningPercent { get; set; } = 95;

    /// <summary>
    /// Gets or sets the GPU memory warning threshold (percentage).
    /// </summary>
    public double GpuMemoryWarningPercent { get; set; } = 90;
}

/// <summary>
/// Event arguments for resource warnings.
/// </summary>
public class ResourceWarningEventArgs : EventArgs
{
    /// <summary>
    /// Gets or sets the name of the resource that exceeded the threshold.
    /// </summary>
    public string ResourceName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the current value.
    /// </summary>
    public double CurrentValue { get; set; }

    /// <summary>
    /// Gets or sets the threshold value.
    /// </summary>
    public double ThresholdValue { get; set; }

    /// <summary>
    /// Gets or sets the full resource snapshot.
    /// </summary>
    public ResourceSnapshot Snapshot { get; set; } = new();
}
