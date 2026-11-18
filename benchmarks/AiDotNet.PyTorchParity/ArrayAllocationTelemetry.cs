using System.Runtime.InteropServices;

/// <summary>Exact diagnostic CLR array callbacks, separate from framework storage requests.</summary>
internal static class ArrayAllocationTelemetry
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void SnapshotDelegate([Out] ulong[] values, uint length);

    private static readonly string? ProfilerPath = Environment.GetEnvironmentVariable("CORECLR_PROFILER_PATH");
    private static SnapshotDelegate? Snapshot;
    private static SnapshotDelegate? SnapshotV2;
    private static IntPtr Library;

    internal static object Read()
    {
        if (Environment.GetEnvironmentVariable("CORECLR_ENABLE_PROFILING") != "1" || string.IsNullOrEmpty(ProfilerPath))
            return new { coverage = "unavailable" };
        try
        {
            if (Snapshot is null)
            {
                Library = NativeLibrary.Load(ProfilerPath);
                Snapshot = Marshal.GetDelegateForFunctionPointer<SnapshotDelegate>(NativeLibrary.GetExport(Library, "ArrayProfilerSnapshot"));
                if (NativeLibrary.TryGetExport(Library, "ArrayProfilerSnapshotV2", out var extended))
                    SnapshotV2 = Marshal.GetDelegateForFunctionPointer<SnapshotDelegate>(extended);
            }
            if (SnapshotV2 is not null)
            {
                var extended = new ulong[NumericArrayAllocationSnapshot.Length];
                SnapshotV2(extended, (uint)extended.Length);
                var primitive = NumericArrayAllocationSnapshot.Describe(extended);
                // Both views come from the same native critical section. Original ABI1 consumers keep their seven slots.
                var legacy = new ulong[7];
                Array.Copy(extended, legacy, legacy.Length);
                return new { coverage = "clr_array_callbacks_v1", abi = 1, snapshot = legacy,
                    scope = "All managed arrays including headers and unknown-class objects conservatively included; unmanaged storage is outside this provider.",
                    numeric = primitive };
            }
            var values = new ulong[7];
            Snapshot(values, (uint)values.Length);
            return new { coverage = "clr_array_callbacks_v1", abi = 1, snapshot = values,
                scope = "All managed arrays including headers and unknown-class objects conservatively included; unmanaged storage is outside this provider." };
        }
        catch (Exception exception)
        {
            return new { coverage = "unavailable", reason = exception.GetType().Name + ": " + exception.Message };
        }
    }
}
