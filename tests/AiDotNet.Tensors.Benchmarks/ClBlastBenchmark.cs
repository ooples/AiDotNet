#if !NET462
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmark comparing CLBlast (industry-standard OpenCL BLAS) vs our OpenCL implementation.
/// This is an apples-to-apples comparison on AMD GPUs.
/// </summary>
public static class ClBlastBenchmark
{
    // CLBlast P/Invoke bindings
    private const string ClBlastDll = "clblast.dll";

    public enum CLBlastStatusCode
    {
        Success = 0,
        OpenCLCompilerNotAvailable = -3,
        TempBufferAllocFailure = -2,
        OpenCLOutOfResources = -5,
        OpenCLOutOfHostMemory = -6,
        OpenCLBuildProgramFailure = -11,
        InvalidValue = -30,
        InvalidCommandQueue = -36,
        InvalidMemObject = -38,
        InvalidBinary = -42,
        InvalidBuildOptions = -43,
        InvalidProgram = -44,
        InvalidProgramExecutable = -45,
        InvalidKernelName = -46,
        InvalidKernelDefinition = -47,
        InvalidKernel = -48,
        InvalidArgIndex = -49,
        InvalidArgValue = -50,
        InvalidArgSize = -51,
        InvalidKernelArgs = -52,
        InvalidLocalNumDimensions = -53,
        InvalidLocalThreadsTotal = -54,
        InvalidLocalThreadsDim = -55,
        InvalidGlobalOffset = -56,
        InvalidEventWaitList = -57,
        InvalidEvent = -58,
        InvalidOperation = -59,
        InvalidBufferSize = -61,
        InvalidGlobalWorkSize = -63,
        NotImplemented = -1024,
        InvalidMatrixA = -1022,
        InvalidMatrixB = -1021,
        InvalidMatrixC = -1020,
        InvalidVectorX = -1019,
        InvalidVectorY = -1018,
        InvalidDimension = -1017,
        InvalidLeadDimA = -1016,
        InvalidLeadDimB = -1015,
        InvalidLeadDimC = -1014,
        InvalidIncrementX = -1013,
        InvalidIncrementY = -1012,
        InsufficientMemoryA = -1011,
        InsufficientMemoryB = -1010,
        InsufficientMemoryC = -1009,
        InsufficientMemoryX = -1008,
        InsufficientMemoryY = -1007,
        InvalidBatchCount = -1006,
        InvalidOverrideKernel = -1005,
        MissingOverrideParameter = -1004,
        InvalidLocalMemUsage = -1003,
        NoHalfPrecision = -1002,
        NoDoublePrecision = -1001,
        InvalidVectorScalar = -1000,
        InsufficientMemoryScalar = -999,
        DatabaseError = -2048,
        UnknownError = -4096,
        UnexpectedError = -8192
    }

    private static int[] GetSizesFromEnv(string name, int[] defaultSizes)
    {
        var value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(value))
            return defaultSizes;

        var parts = value.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var sizes = new List<int>(parts.Length);
        foreach (var part in parts)
        {
            if (int.TryParse(part.Trim(), out var size) && size > 0)
                sizes.Add(size);
        }

        return sizes.Count > 0 ? sizes.ToArray() : defaultSizes;
    }

    private static bool GetEnvBool(string name)
    {
        var value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(value))
            return false;

        return value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
               value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
               value.Equals("yes", StringComparison.OrdinalIgnoreCase) ||
               value.Equals("on", StringComparison.OrdinalIgnoreCase);
    }

    public enum CLBlastLayout
    {
        RowMajor = 101,
        ColMajor = 102
    }

    public enum CLBlastTranspose
    {
        No = 111,
        Yes = 112,
        Conjugate = 113
    }

    [DllImport(ClBlastDll, CallingConvention = CallingConvention.Cdecl)]
    private static extern CLBlastStatusCode CLBlastSgemm(
        CLBlastLayout layout,
        CLBlastTranspose a_transpose,
        CLBlastTranspose b_transpose,
        UIntPtr m, UIntPtr n, UIntPtr k,
        float alpha,
        IntPtr a_buffer, UIntPtr a_offset, UIntPtr a_ld,
        IntPtr b_buffer, UIntPtr b_offset, UIntPtr b_ld,
        float beta,
        IntPtr c_buffer, UIntPtr c_offset, UIntPtr c_ld,
        ref IntPtr queue,
        IntPtr eventPtr);

    private static readonly Lazy<bool> ClblastAvailable = new Lazy<bool>(CheckClblastAvailable);

    public static bool IsClBlastAvailable => ClblastAvailable.Value;

    private static bool CheckClblastAvailable()
    {
        try
        {
            // Try to load the DLL
            var handle = NativeLibrary.Load(ClBlastDll);
            NativeLibrary.Free(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }

    public static void Run()
    {
        Console.WriteLine("=== CLBlast vs AiDotNet OpenCL Benchmark ===");
        Console.WriteLine("Apples-to-apples comparison on AMD GPU");
        Console.WriteLine();

        // Check availability
        Console.WriteLine($"CLBlast Available: {IsClBlastAvailable}");
        Console.WriteLine($"AiDotNet OpenCL Available: {OpenClContext.IsAvailable}");
        Console.WriteLine();

        if (!OpenClContext.IsAvailable)
        {
            Console.WriteLine("ERROR: OpenCL not available - cannot run benchmark.");
            return;
        }

        // Print device info
        using var context = new OpenClContext();
        Console.WriteLine($"Device: {context.DeviceName}");
        Console.WriteLine($"Vendor: {context.DeviceVendor}");
        Console.WriteLine($"Compute Units: {context.MaxComputeUnits}");
        Console.WriteLine($"Global Memory: {context.GlobalMemSize / (1024.0 * 1024 * 1024):F2} GB");
        Console.WriteLine($"Max Work Group Size: {context.MaxWorkGroupSize}");
        Console.WriteLine();

        // Test sizes (override via AIDOTNET_CLBLAST_SIZES="256,512")
        int[] sizes = GetSizesFromEnv("AIDOTNET_CLBLAST_SIZES", new[] { 256, 512, 1024, 2048, 4096 });
        bool skipDense = GetEnvBool("AIDOTNET_CLBLAST_SKIP_DENSE");

        Console.WriteLine("Matrix Multiplication (C = A x B, square matrices):");
        Console.WriteLine("Size       |  CLBlast (GFLOPS)  |  AiDotNet (GFLOPS)  |  Ratio");
        Console.WriteLine(new string('-', 70));

        using var matmul = new OpenClMatMul(context);

        foreach (int size in sizes)
        {
            BenchmarkSize(context, matmul, size);
        }

        if (!skipDense)
        {
            Console.WriteLine();
            Console.WriteLine("DenseLayer-style (batch=64, in=768, out=3072):");
            BenchmarkDenseLayer(context, matmul, 64, 768, 3072);

            Console.WriteLine();
            Console.WriteLine("Large matrix (batch=128, in=4096, out=4096):");
            BenchmarkDenseLayer(context, matmul, 128, 4096, 4096);
        }

        using var backend = new OpenClBackend();
        if (backend.IsAvailable)
        {
            Console.WriteLine();
            Console.WriteLine("DirectGpu TUNED (OpenClBackend) vs CLBlast:");
            Console.WriteLine("Size       |  CLBlast (GFLOPS)  |  AiDotNet (GFLOPS)  |  Ratio");
            Console.WriteLine(new string('-', 70));

            foreach (int size in sizes)
            {
                BenchmarkSizeTuned(context, backend, size);
            }

            if (!skipDense)
            {
                Console.WriteLine();
                Console.WriteLine("DenseLayer-style TUNED (batch=64, in=768, out=3072):");
                BenchmarkDenseLayerTuned(context, backend, 64, 768, 3072);

                Console.WriteLine();
                Console.WriteLine("Large matrix TUNED (batch=128, in=4096, out=4096):");
                BenchmarkDenseLayerTuned(context, backend, 128, 4096, 4096);
            }
        }
    }

    private static void BenchmarkSizeTuned(OpenClContext context, OpenClBackend backend, int size)
    {
        int m = size, k = size, n = size;
        double flops = 2.0 * m * n * k;

        // Generate test data
        var random = new Random(42);
        var A = new float[m * k];
        var B = new float[k * n];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < B.Length; i++) B[i] = (float)(random.NextDouble() * 2 - 1);

        // Benchmark CLBlast
        double clblastGflops = 0;
        if (IsClBlastAvailable)
        {
            clblastGflops = BenchmarkClBlast(context, A, B, m, k, n, flops);
        }

        // Benchmark AiDotNet TUNED OpenCL
        double aidotnetGflops = BenchmarkAiDotNetTuned(backend, A, B, m, k, n, flops);

        // Calculate ratio
        string ratio = (clblastGflops > 0 && aidotnetGflops > 0)
            ? $"{clblastGflops / aidotnetGflops:F2}x"
            : "N/A";

        string clblastStr = clblastGflops > 0 ? $"{clblastGflops,12:F1}" : "N/A".PadLeft(12);
        string aidotnetStr = aidotnetGflops > 0 ? $"{aidotnetGflops,12:F1}" : "FAILED".PadLeft(12);

        Console.WriteLine($"{size,5}x{size,-4} |  {clblastStr}      |  {aidotnetStr}       |  {ratio}");
    }

    private static void BenchmarkDenseLayerTuned(OpenClContext context, OpenClBackend backend, int batch, int inputSize, int outputSize)
    {
        Console.WriteLine($"Batch={batch}, InputSize={inputSize}, OutputSize={outputSize}");

        int m = batch, k = inputSize, n = outputSize;
        double flops = 2.0 * m * n * k;

        var random = new Random(42);
        var input = new float[m * k];
        var weights = new float[k * n];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 0.1 - 0.05);

        double clblastGflops = 0;
        if (IsClBlastAvailable)
        {
            clblastGflops = BenchmarkClBlast(context, input, weights, m, k, n, flops);
        }

        double aidotnetGflops = BenchmarkAiDotNetTuned(backend, input, weights, m, k, n, flops);

        Console.WriteLine($"  CLBlast:  {(clblastGflops > 0 ? $"{clblastGflops:F1} GFLOPS" : "N/A")}");
        Console.WriteLine($"  AiDotNet TUNED: {(aidotnetGflops > 0 ? $"{aidotnetGflops:F1} GFLOPS" : "FAILED")}");

        if (clblastGflops > 0 && aidotnetGflops > 0)
        {
            double ratio = clblastGflops / aidotnetGflops;
            if (ratio > 1)
                Console.WriteLine($"  CLBlast is {ratio:F2}x faster");
            else
                Console.WriteLine($"  AiDotNet is {1/ratio:F2}x faster!");
        }
    }

    private static void BenchmarkSize(OpenClContext context, OpenClMatMul matmul, int size)
    {
        int m = size, k = size, n = size;
        double flops = 2.0 * m * n * k;

        // Generate test data
        var random = new Random(42);
        var A = new float[m * k];
        var B = new float[k * n];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < B.Length; i++) B[i] = (float)(random.NextDouble() * 2 - 1);

        // Benchmark CLBlast
        double clblastGflops = 0;
        if (IsClBlastAvailable)
        {
            clblastGflops = BenchmarkClBlast(context, A, B, m, k, n, flops);
        }

        // Benchmark AiDotNet OpenCL
        double aidotnetGflops = BenchmarkAiDotNet(matmul, A, B, m, k, n, flops);

        // Calculate ratio
        string ratio = (clblastGflops > 0 && aidotnetGflops > 0)
            ? $"{clblastGflops / aidotnetGflops:F2}x"
            : "N/A";

        string clblastStr = clblastGflops > 0 ? $"{clblastGflops,12:F1}" : "N/A".PadLeft(12);
        string aidotnetStr = aidotnetGflops > 0 ? $"{aidotnetGflops,12:F1}" : "FAILED".PadLeft(12);

        Console.WriteLine($"{size,5}x{size,-4} |  {clblastStr}      |  {aidotnetStr}       |  {ratio}");
    }

    private static double BenchmarkClBlast(OpenClContext context, float[] A, float[] B, int m, int k, int n, double flops)
    {
        try
        {
            // Create OpenCL buffers
            using var bufferA = new OpenClBuffer<float>(context, A, OpenClNative.ClMemFlags.ReadOnly);
            using var bufferB = new OpenClBuffer<float>(context, B, OpenClNative.ClMemFlags.ReadOnly);
            using var bufferC = new OpenClBuffer<float>(context, m * n, OpenClNative.ClMemFlags.ReadWrite);

            IntPtr queue = context.CommandQueue;

            // Warmup
            for (int i = 0; i < 3; i++)
            {
                var status = CLBlastSgemm(
                    CLBlastLayout.RowMajor,
                    CLBlastTranspose.No,
                    CLBlastTranspose.No,
                    (UIntPtr)m, (UIntPtr)n, (UIntPtr)k,
                    1.0f,
                    bufferA.Handle, UIntPtr.Zero, (UIntPtr)k,
                    bufferB.Handle, UIntPtr.Zero, (UIntPtr)n,
                    0.0f,
                    bufferC.Handle, UIntPtr.Zero, (UIntPtr)n,
                    ref queue,
                    IntPtr.Zero);

                if (status != CLBlastStatusCode.Success)
                {
                    Console.WriteLine($"CLBlast warmup failed: {status}");
                    return 0;
                }
                context.Finish();
            }

            // Benchmark
            const int iterations = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                CLBlastSgemm(
                    CLBlastLayout.RowMajor,
                    CLBlastTranspose.No,
                    CLBlastTranspose.No,
                    (UIntPtr)m, (UIntPtr)n, (UIntPtr)k,
                    1.0f,
                    bufferA.Handle, UIntPtr.Zero, (UIntPtr)k,
                    bufferB.Handle, UIntPtr.Zero, (UIntPtr)n,
                    0.0f,
                    bufferC.Handle, UIntPtr.Zero, (UIntPtr)n,
                    ref queue,
                    IntPtr.Zero);
                context.Finish();
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            return (flops / seconds) / 1e9;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"CLBlast error: {ex.Message}");
            return 0;
        }
    }

    private static double BenchmarkAiDotNet(OpenClMatMul matmul, float[] A, float[] B, int m, int k, int n, double flops)
    {
        try
        {
            // Warmup
            for (int i = 0; i < 3; i++)
            {
                var result = matmul.MatMulFloat(A, m, k, B, n);
                if (result == null)
                {
                    Console.WriteLine($"AiDotNet warmup failed: {matmul.LastError}");
                    return 0;
                }
            }

            // Benchmark
            const int iterations = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                matmul.MatMulFloat(A, m, k, B, n);
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            return (flops / seconds) / 1e9;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"AiDotNet error: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Benchmark using DirectGpuEngine which uses the auto-tuned GEMM kernels from the database.
    /// </summary>
    private static double BenchmarkAiDotNetTuned(OpenClBackend backend, float[] A, float[] B, int m, int k, int n, double flops)
    {
        try
        {
            // Create GPU buffers
            using var bufA = backend.AllocateBuffer(A);
            using var bufB = backend.AllocateBuffer(B);
            using var bufC = backend.AllocateBuffer(m * n);

            // Warmup
            for (int i = 0; i < 3; i++)
            {
                backend.Gemm(bufA, bufB, bufC, m, n, k, 1.0f, 0.0f);
            }
            backend.Synchronize();

            // Benchmark
            const int iterations = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                backend.Gemm(bufA, bufB, bufC, m, n, k, 1.0f, 0.0f);
            }
            backend.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            return (flops / seconds) / 1e9;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"AiDotNet tuned error: {ex.Message}");
            return 0;
        }
    }

    private static void BenchmarkDenseLayer(OpenClContext context, OpenClMatMul matmul, int batch, int inputSize, int outputSize)
    {
        Console.WriteLine($"Batch={batch}, InputSize={inputSize}, OutputSize={outputSize}");

        int m = batch, k = inputSize, n = outputSize;
        double flops = 2.0 * m * n * k;

        var random = new Random(42);
        var input = new float[m * k];
        var weights = new float[k * n];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 0.1 - 0.05);

        double clblastGflops = 0;
        if (IsClBlastAvailable)
        {
            clblastGflops = BenchmarkClBlast(context, input, weights, m, k, n, flops);
        }

        double aidotnetGflops = BenchmarkAiDotNet(matmul, input, weights, m, k, n, flops);

        Console.WriteLine($"  CLBlast:  {(clblastGflops > 0 ? $"{clblastGflops:F1} GFLOPS" : "N/A")}");
        Console.WriteLine($"  AiDotNet: {(aidotnetGflops > 0 ? $"{aidotnetGflops:F1} GFLOPS" : "FAILED")}");

        if (clblastGflops > 0 && aidotnetGflops > 0)
        {
            double ratio = clblastGflops / aidotnetGflops;
            if (ratio > 1)
                Console.WriteLine($"  CLBlast is {ratio:F2}x faster");
            else
                Console.WriteLine($"  AiDotNet is {1/ratio:F2}x faster!");
        }
    }
}
#endif
