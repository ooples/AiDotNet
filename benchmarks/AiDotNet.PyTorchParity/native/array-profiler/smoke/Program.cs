using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

internal static unsafe class Program
{
    private static byte[]? Bytes;
    private static float[]? Floats;
    private static double[,]? Matrix;
    private static object? Wrapper;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AllocateArrays()
    {
        Bytes = new byte[32];
        Floats = new float[17];
        Matrix = new double[2, 3];
        Wrapper = new object();
    }

    private static int Main()
    {
        var path = Environment.GetEnvironmentVariable("CORECLR_PROFILER_PATH")
            ?? throw new InvalidOperationException("Launch with the native array profiler enabled.");
        var library = NativeLibrary.Load(path);
        var snapshot = (delegate* unmanaged[Cdecl]<ulong*, uint, void>)NativeLibrary.GetExport(library, "ArrayProfilerSnapshot");
        ulong* before = stackalloc ulong[7];
        ulong* after = stackalloc ulong[7];
        // Complete first-use JIT, class loading and shared-pool initialization before exact-count checks.
        AllocateArrays();
        var warm = ArrayPool<byte>.Shared.Rent(32);
        ArrayPool<byte>.Shared.Return(warm);
        snapshot(before, 7);
        AllocateArrays();
        snapshot(after, 7);
        ulong arrays = after[1] + after[3] - before[1] - before[3];
        ulong bytes = after[2] + after[4] - before[2] - before[4];
        bool freshPass = arrays == 3 && bytes >= 32 + 17 * 4 + 6 * 8 && after[5] == 0 && after[6] == 1;
        snapshot(before, 7);
        var borrowed = ArrayPool<byte>.Shared.Rent(32);
        ArrayPool<byte>.Shared.Return(borrowed);
        snapshot(after, 7);
        ulong borrowedFreshArrays = after[1] + after[3] - before[1] - before[3];
        bool pass = freshPass && borrowedFreshArrays == 0 && after[5] == 0;
        Console.WriteLine($"{{\"pass\":{pass.ToString().ToLowerInvariant()},\"fresh_arrays\":{arrays},\"fresh_full_object_bytes\":{bytes},\"pool_borrow_fresh_arrays\":{borrowedFreshArrays},\"sizing_failures\":{after[5]}}}");
        GC.KeepAlive(Bytes);
        GC.KeepAlive(Floats);
        GC.KeepAlive(Matrix);
        GC.KeepAlive(Wrapper);
        return pass ? 0 : 1;
    }
}
