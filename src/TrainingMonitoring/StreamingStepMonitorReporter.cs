using System.Diagnostics;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using LogLevel = AiDotNet.Interfaces.LogLevel;

namespace AiDotNet.TrainingMonitoring;

/// <summary>
/// Reports step-level metrics and periodic resource usage from the streaming training loop to an
/// <see cref="ITrainingMonitor{T}"/>.
/// </summary>
/// <remarks>
/// Every <c>everyNSteps</c> optimizer steps it logs, at the global step: <c>train_loss</c>, <c>train_loss_avg</c>,
/// <c>learning_rate</c>, <c>step_ms</c>, <c>samples_per_second</c>, <c>tokens_per_second</c> (when tokens per sample
/// is known) and <c>grad_norm</c> (when the optimizer step computed one), then calls
/// <see cref="ITrainingMonitor{T}.UpdateProgress"/>. Times are raw <see cref="Stopwatch"/> readings over the window
/// since the previous report. The loss passed in is the host value the loop already read, so nothing here
/// synchronizes a device. Resource usage is sampled on a reporting step once the wall-time interval has elapsed.
/// </remarks>
/// <typeparam name="T">The numeric type of the monitor's metric values.</typeparam>
internal sealed class StreamingStepMonitorReporter<T>
{
    private const double BytesPerMegabyte = 1024.0 * 1024.0;

    private readonly ITrainingMonitor<T> _monitor;
    private readonly string _sessionId;
    private readonly int _everyNSteps;
    private readonly int? _tokensPerSample;
    private readonly long _resourceIntervalTicks;
    private readonly Func<double?>? _gradNormSource;
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    private long _windowStartTimestamp;
    private long _windowStartSamples;
    private long _windowStartStep;
    private double _windowLossSum;
    private int _windowLossCount;

    private long _lastResourceTimestamp;
    private TimeSpan _lastProcessorTime;
    private bool _resourceBaselineTaken;

    /// <summary>Creates a reporter for one monitor session.</summary>
    /// <param name="monitor">The monitor to report to.</param>
    /// <param name="sessionId">The monitor session the loop opened.</param>
    /// <param name="everyNSteps">Reporting interval in optimizer steps; must be positive.</param>
    /// <param name="tokensPerSample">Tokens per sample for <c>tokens_per_second</c>, or null.</param>
    /// <param name="resourceIntervalSeconds">Resource sampling interval in seconds; 0 disables it.</param>
    /// <param name="gradNormSource">Returns the last step's gradient norm, or null when none was computed.</param>
    /// <param name="startStep">The global step training resumes from.</param>
    /// <param name="totalSteps">The planned step count, or 0 when unknown.</param>
    public StreamingStepMonitorReporter(
        ITrainingMonitor<T> monitor,
        string sessionId,
        int everyNSteps,
        int? tokensPerSample,
        double resourceIntervalSeconds,
        Func<double?>? gradNormSource,
        long startStep,
        long totalSteps = 0)
    {
        TotalSteps = totalSteps;
        _monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
        _sessionId = sessionId ?? throw new ArgumentNullException(nameof(sessionId));
        if (everyNSteps <= 0) throw new ArgumentOutOfRangeException(nameof(everyNSteps));
        _everyNSteps = everyNSteps;
        _tokensPerSample = tokensPerSample;
        _resourceIntervalTicks = resourceIntervalSeconds > 0
            ? (long)(resourceIntervalSeconds * Stopwatch.Frequency)
            : 0L;
        _gradNormSource = gradNormSource;
        _windowStartStep = startStep;
        _windowStartTimestamp = Stopwatch.GetTimestamp();
        _lastResourceTimestamp = _windowStartTimestamp;
    }

    /// <summary>The planned step count passed to <see cref="ITrainingMonitor{T}.UpdateProgress"/>, or 0 when unknown.</summary>
    public long TotalSteps { get; }

    /// <summary>
    /// Records one completed optimizer step and reports when the step is a multiple of the interval.
    /// </summary>
    /// <param name="globalStep">The global step just completed (1-based, counted across epochs).</param>
    /// <param name="hasLoss">False when the step produced no loss (a distributed rank with no local samples).</param>
    /// <param name="stepLoss">The step's loss as already read to the host; ignored when <paramref name="hasLoss"/> is false.</param>
    /// <param name="totalSamplesSeen">Samples consumed by training so far (cumulative).</param>
    /// <param name="learningRate">The optimizer's current learning rate.</param>
    /// <param name="totalSteps">The planned step count, or 0 when unknown.</param>
    /// <param name="currentEpoch">The current epoch (1-based).</param>
    /// <param name="totalEpochs">The planned epoch count.</param>
    public void OnStep(
        long globalStep,
        bool hasLoss,
        T stepLoss,
        long totalSamplesSeen,
        double learningRate,
        long totalSteps,
        int currentEpoch,
        int totalEpochs)
    {
        if (hasLoss)
        {
            _windowLossSum += _numOps.ToDouble(stepLoss);
            _windowLossCount++;
        }

        if (globalStep % _everyNSteps != 0)
            return;

        long now = Stopwatch.GetTimestamp();
        long steps = globalStep - _windowStartStep;
        double seconds = (now - _windowStartTimestamp) / (double)Stopwatch.Frequency;
        long samples = totalSamplesSeen - _windowStartSamples;
        int step = ClampToInt(globalStep);

        try
        {
            if (hasLoss)
                _monitor.LogMetric(_sessionId, "train_loss", stepLoss, step);
            if (_windowLossCount > 0)
                _monitor.LogMetric(_sessionId, "train_loss_avg", _numOps.FromDouble(_windowLossSum / _windowLossCount), step);
            if (!double.IsNaN(learningRate))
                _monitor.LogMetric(_sessionId, "learning_rate", _numOps.FromDouble(learningRate), step);
            if (steps > 0 && seconds > 0)
            {
                _monitor.LogMetric(_sessionId, "step_ms", _numOps.FromDouble(seconds * 1000.0 / steps), step);
                _monitor.LogMetric(_sessionId, "samples_per_second", _numOps.FromDouble(samples / seconds), step);
                if (_tokensPerSample is int tokens)
                    _monitor.LogMetric(_sessionId, "tokens_per_second", _numOps.FromDouble(samples * (double)tokens / seconds), step);
            }

            if (_gradNormSource?.Invoke() is double gradNorm)
            {
                if (!double.IsNaN(gradNorm) && !double.IsInfinity(gradNorm))
                {
                    _monitor.LogMetric(_sessionId, "grad_norm", _numOps.FromDouble(gradNorm), step);
                }
                else
                {
                    // A non-finite norm is usually a step the optimizer's anomaly guard (or loss scaling) skipped.
                    // As a metric it would read as a training failure to monitors that stop on non-finite values,
                    // so it is reported as a warning instead.
                    _monitor.LogMessage(_sessionId, LogLevel.Warning,
                        $"Non-finite gradient norm ({gradNorm}) at step {globalStep}.");
                }
            }

            _monitor.UpdateProgress(_sessionId, step, ClampToInt(totalSteps), currentEpoch, totalEpochs);

            if (_resourceIntervalTicks > 0 && (!_resourceBaselineTaken || now - _lastResourceTimestamp >= _resourceIntervalTicks))
                ReportResourceUsage(now, step);
        }
        catch (Exception ex)
        {
            // A broken monitor must not abort training; surface it to operators instead.
            Trace.TraceWarning($"Training monitor step report failed at step {globalStep}: {ex}");
        }

        _windowStartTimestamp = Stopwatch.GetTimestamp();
        _windowStartSamples = totalSamplesSeen;
        _windowStartStep = globalStep;
        _windowLossSum = 0.0;
        _windowLossCount = 0;
    }

    /// <summary>
    /// Reports progress at the end of an epoch, so a monitor's stall detection sees the latest step even when the
    /// epoch ended between reporting steps.
    /// </summary>
    public void OnEpochEnd(long globalStep, int currentEpoch, int totalEpochs)
    {
        try
        {
            _monitor.UpdateProgress(_sessionId, ClampToInt(globalStep), ClampToInt(TotalSteps), currentEpoch, totalEpochs);
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"Training monitor progress report failed at step {globalStep}: {ex}");
        }
    }

    private void ReportResourceUsage(long now, int step)
    {
        // TotalProcessorTime is one GetProcessTimes call. Environment.WorkingSet reads this process's counters only;
        // Process.WorkingSet64 snapshots every process on the machine (Windows), which costs milliseconds.
        TimeSpan cpuTime;
        using (var process = Process.GetCurrentProcess())
            cpuTime = process.TotalProcessorTime;
        double workingSetMb = Environment.WorkingSet / BytesPerMegabyte;

        if (!_resourceBaselineTaken)
        {
            // The first sample only fixes the CPU-time baseline (a percentage needs a delta). Take it at the start
            // of the run so the first report covers a full interval.
            _resourceBaselineTaken = true;
            _lastProcessorTime = cpuTime;
            _lastResourceTimestamp = now;
            return;
        }

        double wallSeconds = (now - _lastResourceTimestamp) / (double)Stopwatch.Frequency;
        double cpuPercent = wallSeconds > 0
            ? (cpuTime - _lastProcessorTime).TotalSeconds / wallSeconds / Environment.ProcessorCount * 100.0
            : 0.0;
        _lastProcessorTime = cpuTime;
        _lastResourceTimestamp = now;

        var gpu = TryReadGpuMemory();
        // GPU memory goes out as metrics only: TrainingMonitor reads LogResourceUsage's gpuMemory as a percentage
        // (above 95 is a critical issue, which stops the streaming loop), so a value in MB would end the run. The
        // device's utilization is the percentage LogResourceUsage's gpuUsage takes.
        _monitor.LogResourceUsage(_sessionId, cpuPercent, workingSetMb, gpuUsage: gpu?.UtilizationPercent, gpuMemory: null);
        if (gpu is { } g)
        {
            _monitor.LogMetric(_sessionId, "gpu_memory_allocated_mb", _numOps.FromDouble(g.AllocatedMb), step);
            _monitor.LogMetric(_sessionId, "gpu_memory_reserved_mb", _numOps.FromDouble(g.ReservedMb), step);
            if (g.TotalMb > 0)
                _monitor.LogMetric(_sessionId, "gpu_memory_total_mb", _numOps.FromDouble(g.TotalMb), step);
            if (g.DeviceUsedMb is double deviceUsed)
                _monitor.LogMetric(_sessionId, "gpu_device_memory_used_mb", _numOps.FromDouble(deviceUsed), step);
            if (g.UtilizationPercent is double utilization)
                _monitor.LogMetric(_sessionId, "gpu_utilization_percent", _numOps.FromDouble(utilization), step);
            if (g.MemoryUtilizationPercent is double memoryUtilization)
                _monitor.LogMetric(_sessionId, "gpu_memory_utilization_percent", _numOps.FromDouble(memoryUtilization), step);
        }
    }

    /// <summary>
    /// Takes the resource baseline now (at the start of the run), so the first resource report after one interval
    /// has a CPU-time delta to report.
    /// </summary>
    public void TakeResourceBaseline()
    {
        if (_resourceIntervalTicks <= 0 || _resourceBaselineTaken) return;
        try
        {
            ReportResourceUsage(Stopwatch.GetTimestamp(), 0);
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"Training monitor resource baseline failed: {ex}");
        }
    }

    // The GPU allocator's own accounting (GpuMemoryStats) for the active GPU engine's backend; the device's memory in
    // use by every process (total minus free, as the driver reports it) and its total; and the device's kernel and
    // memory-traffic utilization over the driver's last sample period (NVML on CUDA). The device-wide figures are
    // null where the backend's driver does not report them.
    private static (double AllocatedMb, double ReservedMb, double TotalMb, double? DeviceUsedMb, double? UtilizationPercent,
        double? MemoryUtilizationPercent)? TryReadGpuMemory()
    {
        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpuEngine)
            return null;
        var backend = gpuEngine.GetBackend();
        if (backend is null)
            return null;
        var snapshot = GpuMemoryStats.Snapshot(backend);
        double totalMb = backend.GlobalMemoryBytes / BytesPerMegabyte;
        double? deviceUsedMb = null;
        if (GpuMemoryStats.TryGetDeviceMemory(backend, out long freeBytes, out long totalBytes))
        {
            deviceUsedMb = (totalBytes - freeBytes) / BytesPerMegabyte;
            totalMb = totalBytes / BytesPerMegabyte;
        }
        double? utilization = null, memoryUtilization = null;
        if (GpuUtilization.TryRead(backend, out int gpuPercent, out int memoryPercent))
        {
            utilization = gpuPercent;
            memoryUtilization = memoryPercent;
        }
        return (snapshot.AllocatedBytes / BytesPerMegabyte, snapshot.ReservedBytes / BytesPerMegabyte, totalMb,
            deviceUsedMb, utilization, memoryUtilization);
    }

    private static int ClampToInt(long value) =>
        value > int.MaxValue ? int.MaxValue : value < 0 ? 0 : (int)value;
}
