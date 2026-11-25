// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Diagnostics;

namespace AiDotNet.Tools.ModelPerfFixtureRunner;

/// <summary>
/// Measures the fixture process's own peak working set and peak private memory, and reports them on standard output
/// for the shard script.
/// </summary>
/// <remarks>
/// <para>
/// The shard script used to be the only observer: it polled the child every 250 ms. A fixture whose whole workload
/// runs in about a second gets four or five samples, so whether its transient peak was seen was a matter of timing.
/// UnifiedMultimodalNetwork measured 749 MB of peak private memory on seven nightly runs and 450 MB on one, with
/// identical allocations. The 450 MB reading then became the baseline, and the next normal run failed the 1.6x
/// memory gate.
/// </para>
/// <para>
/// The working-set peak comes from the OS high-water mark (VmHWM on Linux, PeakWorkingSetSize on Windows), which no
/// sampling interval can miss. Linux keeps no high-water mark for private memory, so a background thread samples it
/// every 5 ms from inside the process (the shard script polled every 250 ms from outside it). On Windows the peak
/// commit charge is also read directly.
/// </para>
/// </remarks>
internal sealed class ProcessPeakSampler : IDisposable
{
    /// <summary>The line prefix the shard script reads.</summary>
    internal const string Marker = "AIDOTNET_PROCESS_PEAKS";

    /// <summary>Sampling interval: short enough for a transient tensor, long enough to cost well under 1% of a core.</summary>
    private const int SampleIntervalMs = 5;

    private readonly Thread _thread;
    private volatile bool _stop;
    private long _peakPrivate;

    internal ProcessPeakSampler()
    {
        _thread = new Thread(Run) { IsBackground = true, Name = "process-peak-sampler", Priority = ThreadPriority.AboveNormal };
        _thread.Start();
    }

    private void Run()
    {
        using var process = Process.GetCurrentProcess();
        while (!_stop)
        {
            Sample(process);
            Thread.Sleep(SampleIntervalMs);
        }
    }

    private void Sample(Process process)
    {
        try
        {
            process.Refresh();
            long current = process.PrivateMemorySize64;
            if (current > Interlocked.Read(ref _peakPrivate)) Interlocked.Exchange(ref _peakPrivate, current);
        }
        catch (InvalidOperationException)
        {
            // The process information is unavailable only while the process is exiting.
        }
    }

    /// <summary>Stops sampling and writes the peaks on one line of standard output.</summary>
    public void Dispose()
    {
        _stop = true;
        _thread.Join();
        using var process = Process.GetCurrentProcess();
        Sample(process);
        long peakWorkingSet = process.PeakWorkingSet64;
        long peakPrivate = Interlocked.Read(ref _peakPrivate);
        if (OperatingSystem.IsWindows()) peakPrivate = Math.Max(peakPrivate, process.PeakPagedMemorySize64);
        Console.Out.WriteLine($"{Marker} {peakWorkingSet} {peakPrivate}");
        Console.Out.Flush();
    }
}
