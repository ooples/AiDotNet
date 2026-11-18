using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Newtonsoft.Json.Linq;

internal static unsafe class Program
{
    private static readonly object[] Held = new object[18];

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AllocateTypedArrays()
    {
        Held[0] = new bool[2]; Held[1] = new char[2]; Held[2] = new sbyte[2]; Held[3] = new byte[2];
        Held[4] = new short[2]; Held[5] = new ushort[2]; Held[6] = new int[2]; Held[7] = new uint[2];
        Held[8] = new long[2]; Held[9] = new ulong[2]; Held[10] = new float[2]; Held[11] = new double[2];
        Held[12] = new nint[2]; Held[13] = new nuint[2]; Held[14] = new float[0];
        Held[15] = new float[2, 3]; Held[16] = new decimal[2]; Held[17] = new object[2];
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidDataException(message);
    }

    private static ulong[] Fixture()
    {
        var values = new ulong[NumericArrayAllocationSnapshot.Length];
        values[1] = 4; values[2] = 204; values[3] = 1; values[4] = 24; values[6] = 1;
        values[7] = 2; values[8] = (ulong)values.Length;
        values[9] = 2; values[10] = 2; values[11] = 36; values[12] = 100;
        values[18] = 1; values[19] = 48; values[20] = 1; values[21] = 56;
        values[22] = 1; values[23] = 24;
        int first = NumericArrayAllocationSnapshot.BinsBegin + 10 * 32 * 3;
        values[first] = 1; values[first + 1] = 12; values[first + 2] = 36;
        values[first + 3] = 1; values[first + 4] = 24; values[first + 5] = 64;
        return values;
    }

    private static void Contracts()
    {
        var valid = NumericArrayAllocationSnapshot.Describe(Fixture());
        Require(valid.Value<bool>("candidate_upper_bound_available"), "Complete conservative fixture rejected.");
        Require(!valid.Value<bool>("exact_primitive_payload"), "Unknown classes cannot establish exact numeric classification.");
        Require(valid.Value<ulong>("fresh_candidate_payload_bytes_upper_bound") == 116
            && valid.Value<ulong>("fresh_candidate_count_upper_bound") == 4, "Conservative unsupported/unknown bound changed.");
        foreach (int slot in new[] { 5, 24, 25 })
        {
            var failed = Fixture(); failed[slot] = 1;
            Require(!NumericArrayAllocationSnapshot.Describe(failed).Value<bool>("candidate_upper_bound_available"), "Sizing/overflow failure accepted.");
        }
        var inconsistent = Fixture(); inconsistent[11]++;
        Require(!NumericArrayAllocationSnapshot.Describe(inconsistent).Value<bool>("candidate_upper_bound_available"), "Inconsistent bin payload accepted.");
        foreach (int slot in new[] { 13, 14, 15, 16 })
        {
            var failed = Fixture();
            int first = NumericArrayAllocationSnapshot.BinsBegin + 10 * 32 * 3;
            failed[first] = 0; failed[first + 1] = 0; failed[first + 2] = 0;
            failed[10] = 1; failed[11] = 24; failed[17] = 36; failed[slot] = 1;
            var receipt = NumericArrayAllocationSnapshot.Describe(failed);
            Require(receipt.Value<bool>("candidate_upper_bound_available") && !receipt.Value<bool>("exact_primitive_payload"), "Failed payload did not retain its full-object upper bound.");
            Require(receipt.Value<ulong>("fresh_candidate_payload_bytes_upper_bound") == 140, "Failed payload full object omitted.");
        }
        bool rejected = false;
        try { NumericArrayAllocationSnapshot.Describe(new ulong[7]); } catch (InvalidDataException) { rejected = true; }
        Require(rejected, "Legacy snapshot was treated as ABI2.");
        var wrongAbi = Fixture(); wrongAbi[7] = 3; rejected = false;
        try { NumericArrayAllocationSnapshot.Describe(wrongAbi); } catch (InvalidDataException) { rejected = true; }
        Require(rejected, "Unknown ABI accepted.");
        Console.WriteLine("{\"contracts_pass\":true,\"runtime_measurement\":false}");
    }

    private static void Native()
    {
        string path = Environment.GetEnvironmentVariable("CORECLR_PROFILER_PATH")
            ?? throw new InvalidOperationException("Native mode requires a launch-enabled profiler.");
        var library = NativeLibrary.Load(path);
        var snapshot = (delegate* unmanaged[Cdecl]<ulong*, uint, void>)NativeLibrary.GetExport(library, "ArrayProfilerSnapshotV2");
        const int length = NumericArrayAllocationSnapshot.Length;
        ulong* before = stackalloc ulong[length]; ulong* after = stackalloc ulong[length];
        AllocateTypedArrays();
        var warm = ArrayPool<byte>.Shared.Rent(32); ArrayPool<byte>.Shared.Return(warm);
        snapshot(before, length); AllocateTypedArrays(); snapshot(after, length);
        Require(after[7] == 2 && after[8] == length && after[6] == 1, "Native ABI/init mismatch.");
        Require(after[9] - before[9] == 16 && after[10] - before[10] == 16, "Expected fourteen primitive dtypes plus empty/rank2 arrays.");
        ulong expectedPayload = 24;
        int[] widths = [1, 2, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8, IntPtr.Size, UIntPtr.Size];
        for (int dtype = 0; dtype < widths.Length; dtype++)
        {
            int index = NumericArrayAllocationSnapshot.BinsBegin + dtype * 32 * 3;
            Require(after[index] - before[index] == (dtype == 10 ? 2UL : 1UL), "Primitive dtype/rank count mismatch.");
            Require(after[index + 1] - before[index + 1] == (ulong)(2 * widths[dtype]), "Primitive payload width mismatch.");
            expectedPayload += (ulong)(2 * widths[dtype]);
        }
        int matrix = NumericArrayAllocationSnapshot.BinsBegin + (10 * 32 + 1) * 3;
        Require(after[matrix] - before[matrix] == 1 && after[matrix + 1] - before[matrix + 1] == 24, "Rank2 payload mismatch.");
        Require(after[11] - before[11] == expectedPayload, "Aggregate primitive payload mismatch.");
        Require(after[18] - before[18] == 1 && after[20] - before[20] == 1, "Reference/unsupported arrays were not classified separately.");
        for (int slot = 13; slot <= 16; slot++) Require(after[slot] == 0, "Payload metadata failure.");
        Require(after[5] == 0 && after[22] == before[22] && after[24] == 0 && after[25] == 0, "Classification/sizing/counter failure.");
        snapshot(before, length);
        var borrowed = ArrayPool<byte>.Shared.Rent(32); ArrayPool<byte>.Shared.Return(borrowed);
        snapshot(after, length);
        Require(after[9] == before[9] && after[11] == before[11], "Warm pool borrow must not become a fresh primitive array.");
        Console.WriteLine($"{{\"native_pass\":true,\"primitive_arrays\":16,\"primitive_payload_bytes\":{expectedPayload},\"pool_borrow_fresh_primitive_arrays\":0}}");
        GC.KeepAlive(Held);
    }

    private static int Main(string[] args)
    {
        if (args.Length != 1) throw new ArgumentException("Use --contracts or --native.");
        if (args[0] == "--contracts") Contracts();
        else if (args[0] == "--native") Native();
        else throw new ArgumentException("Unknown smoke mode.");
        return 0;
    }
}
