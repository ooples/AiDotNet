// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

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
/// THE SAMPLER MUST NOT ALLOCATE OR SPEND CPU. The census reads allocation as <c>GC.GetTotalAllocatedBytes</c> and CPU
/// as the whole process's processor time, so a sampler thread is part of every fixture's measurement. The first
/// version refreshed a <see cref="Process"/> every 5 ms: that allocated ~1.5 MB per fixture-second (MixtureOfExperts
/// 16.8 -> 18.8-20.0 MB, GaussianSplatting 3.5 -> 4.6-5.0 MB) and added CPU, which let timing-only noise pass the
/// gate's CPU corroboration.
/// </para>
/// <para>
/// So: on Windows the OS keeps both high-water marks (PeakWorkingSetSize and PeakPagefileUsage, the peak commit
/// charge that <see cref="Process.PrivateMemorySize64"/> reports the current value of), read once at the end with no
/// thread at all. On Linux the working-set peak is VmHWM, also read once; private memory (VmData, what .NET reports
/// as PrivateMemorySize64 there) has no high-water mark, so a thread re-reads <c>/proc/self/status</c> every 5 ms
/// into one reused buffer through one open handle, parsing the bytes in place.
/// </para>
/// </remarks>
internal sealed class ProcessPeakSampler : IDisposable
{
    /// <summary>The line prefix the shard script reads.</summary>
    internal const string Marker = "AIDOTNET_PROCESS_PEAKS";

    /// <summary>Sampling interval for Linux private memory.</summary>
    private const int SampleIntervalMs = 5;

    private static readonly byte[] VmData = "VmData:"u8.ToArray();
    private static readonly byte[] VmHwm = "VmHWM:"u8.ToArray();

    private readonly Thread? _thread;
    private readonly SafeFileHandle? _status;
    private readonly byte[] _buffer = new byte[8192];
    private volatile bool _stop;
    private long _peakPrivate;

    internal ProcessPeakSampler()
    {
        if (!OperatingSystem.IsLinux()) return;
        try
        {
            _status = File.OpenHandle("/proc/self/status", FileMode.Open, FileAccess.Read);
        }
        catch (IOException)
        {
            return;
        }
        catch (UnauthorizedAccessException)
        {
            return;
        }

        _thread = new Thread(Run) { IsBackground = true, Name = "process-peak-sampler", Priority = ThreadPriority.AboveNormal };
        _thread.Start();
    }

    private void Run()
    {
        while (!_stop)
        {
            SampleLinux();
            Thread.Sleep(SampleIntervalMs);
        }
    }

    /// <summary>Reads VmData (and returns VmHWM) from /proc/self/status without allocating; kilobytes as bytes.</summary>
    private long SampleLinux()
    {
        if (_status is null) return 0;
        lock (_buffer)
        {
            int length = RandomAccess.Read(_status, _buffer, 0);
            var text = new ReadOnlySpan<byte>(_buffer, 0, length);
            long data = Field(text, VmData);
            if (data > Interlocked.Read(ref _peakPrivate)) Interlocked.Exchange(ref _peakPrivate, data);
            return Field(text, VmHwm);
        }
    }

    /// <summary>The value of a "Name:   1234 kB" line, in bytes; 0 when absent.</summary>
    private static long Field(ReadOnlySpan<byte> text, ReadOnlySpan<byte> name)
    {
        int at = text.IndexOf(name);
        if (at < 0) return 0;
        long kilobytes = 0;
        for (int i = at + name.Length; i < text.Length && text[i] != (byte)'\n'; i++)
        {
            byte c = text[i];
            if (c >= (byte)'0' && c <= (byte)'9') kilobytes = kilobytes * 10 + (c - '0');
        }

        return kilobytes * 1024;
    }

    /// <summary>Stops sampling and writes the peaks on one line of standard output.</summary>
    public void Dispose()
    {
        _stop = true;
        _thread?.Join();
        long peakWorkingSet;
        long peakPrivate;
        if (OperatingSystem.IsWindows() && TryWindowsPeaks(out peakWorkingSet, out peakPrivate))
        {
        }
        else if (_status is not null)
        {
            peakWorkingSet = SampleLinux();
            peakPrivate = Interlocked.Read(ref _peakPrivate);
            _status.Dispose();
        }
        else
        {
            using var process = Process.GetCurrentProcess();
            peakWorkingSet = process.PeakWorkingSet64;
            peakPrivate = process.PrivateMemorySize64;
        }

        Console.Out.WriteLine($"{Marker} {peakWorkingSet} {peakPrivate}");
        Console.Out.Flush();
    }

    private static bool TryWindowsPeaks(out long peakWorkingSet, out long peakPrivate)
    {
        var counters = new ProcessMemoryCounters { Size = (uint)Marshal.SizeOf<ProcessMemoryCounters>() };
        if (!GetProcessMemoryInfo(GetCurrentProcess(), ref counters, counters.Size))
        {
            peakWorkingSet = peakPrivate = 0;
            return false;
        }

        peakWorkingSet = (long)counters.PeakWorkingSetSize;
        peakPrivate = (long)counters.PeakPagefileUsage;
        return true;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessMemoryCounters
    {
        public uint Size;
        public uint PageFaultCount;
        public nuint PeakWorkingSetSize;
        public nuint WorkingSetSize;
        public nuint QuotaPeakPagedPoolUsage;
        public nuint QuotaPagedPoolUsage;
        public nuint QuotaPeakNonPagedPoolUsage;
        public nuint QuotaNonPagedPoolUsage;
        public nuint PagefileUsage;
        public nuint PeakPagefileUsage;
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentProcess();

    [DllImport("psapi.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetProcessMemoryInfo(IntPtr process, ref ProcessMemoryCounters counters, uint size);
}
