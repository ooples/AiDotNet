using System.Runtime.InteropServices;

namespace AiDotNet.ProgramSynthesis.Execution;

/// <summary>
/// Wraps a Windows job object that caps the memory of a sandboxed child process and terminates whatever is still
/// running in it when the handle closes.
/// </summary>
/// <remarks>
/// The type is Windows-only and every entry point is guarded, so on any other platform the factory returns
/// <c>null</c> and no interop is attempted. Failures are also non-fatal: a machine or container policy can forbid
/// job assignment, and a sandbox that still enforces its wall-clock limit and its output caps is far better than a
/// sandbox that refuses to run. Callers therefore treat a <c>null</c> result as "no memory cap available here" and
/// report that honestly rather than pretending the cap was applied.
/// </remarks>
internal sealed class WindowsJobObject : IDisposable
{
    private const int JobObjectExtendedLimitInformation = 9;
    private const uint JobObjectLimitProcessMemory = 0x0000_0100;
    private const uint JobObjectLimitJobMemory = 0x0000_0200;
    private const uint JobObjectLimitKillOnJobClose = 0x0000_2000;

    private IntPtr _handle;
    private bool _disposed;

    private WindowsJobObject(IntPtr handle) => _handle = handle;

    /// <summary>Creates a job object that limits committed memory and kills its members when disposed.</summary>
    /// <param name="memoryLimitBytes">The per-process and per-job commit limit in bytes; values below one are ignored.</param>
    /// <returns>The job object, or <c>null</c> when the platform is not Windows or the operating system refused.</returns>
    public static WindowsJobObject? TryCreate(long memoryLimitBytes)
    {
        if (!IsWindows() || memoryLimitBytes <= 0)
        {
            return null;
        }

        IntPtr handle;
        try
        {
            handle = CreateJobObject(IntPtr.Zero, null);
        }
        catch (DllNotFoundException)
        {
            return null;
        }
        catch (EntryPointNotFoundException)
        {
            return null;
        }

        if (handle == IntPtr.Zero)
        {
            return null;
        }

        var information = default(JobObjectExtendedLimitInformationNative);
        information.BasicLimitInformation.LimitFlags =
            JobObjectLimitProcessMemory | JobObjectLimitJobMemory | JobObjectLimitKillOnJobClose;
        information.ProcessMemoryLimit = new UIntPtr((ulong)memoryLimitBytes);
        information.JobMemoryLimit = new UIntPtr((ulong)memoryLimitBytes);

        int size = Marshal.SizeOf(typeof(JobObjectExtendedLimitInformationNative));
        IntPtr buffer = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(information, buffer, fDeleteOld: false);
            if (!SetInformationJobObject(handle, JobObjectExtendedLimitInformation, buffer, (uint)size))
            {
                CloseHandle(handle);
                return null;
            }
        }
        catch (EntryPointNotFoundException)
        {
            CloseHandle(handle);
            return null;
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }

        return new WindowsJobObject(handle);
    }

    /// <summary>Adds a running process to this job so the memory cap applies to it and to whatever it starts.</summary>
    /// <param name="processHandle">The native handle of the process to assign.</param>
    /// <returns><c>true</c> when the process joined the job.</returns>
    public bool TryAssign(IntPtr processHandle)
    {
        if (_disposed || _handle == IntPtr.Zero || processHandle == IntPtr.Zero)
        {
            return false;
        }

        try
        {
            return AssignProcessToJobObject(_handle, processHandle);
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    /// <summary>Closes the job handle, which terminates any process still running inside it.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_handle != IntPtr.Zero)
        {
            CloseHandle(_handle);
            _handle = IntPtr.Zero;
        }
    }

    private static bool IsWindows()
    {
#if NET5_0_OR_GREATER
        return OperatingSystem.IsWindows();
#else
        return Environment.OSVersion.Platform == PlatformID.Win32NT;
#endif
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr jobAttributes, string? name);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetInformationJobObject(
        IntPtr job,
        int informationClass,
        IntPtr information,
        uint informationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(IntPtr handle);

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCountersNative
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectBasicLimitInformationNative
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectExtendedLimitInformationNative
    {
        public JobObjectBasicLimitInformationNative BasicLimitInformation;
        public IoCountersNative IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }
}
