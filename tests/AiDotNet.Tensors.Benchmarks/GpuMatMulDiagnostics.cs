using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Benchmarks.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Provides diagnostic tools and performance benchmarks for GPU-based matrix multiplication.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> GPUs are incredibly fast at math, but they can sometimes be 
/// tricky to set up. This class helps us make sure the GPU is giving the exact same 
/// answers as the CPU, and measures exactly how much faster it is (the speedup).</para>
/// </remarks>
public static class GpuMatMulDiagnostics
{
    /// <summary>
    /// Runs a series of GPU correctness and performance tests.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method runs the actual tests. It first checks if the 
    /// GPU is even available, then it tests various math problem sizes to ensure accuracy 
    /// and measure throughput in GFLOPS (billions of operations per second).</para>
    /// </remarks>
    public static void Run()
    {
        Console.WriteLine("=== GPU Matrix Multiply Diagnostics ===");

        if (!AiDotNetEngine.AutoDetectAndConfigureGpu())
        {
            Console.WriteLine("GPU not available; skipping GPU diagnostics.");
            return;
        }

        var gpuEngine = AiDotNetEngine.Current;
        var cpuEngine = new CpuEngine();
        Console.WriteLine($"Engine: {gpuEngine.Name}");
        Console.WriteLine();

        var random = RandomHelper.CreateSeededRandom(123);
        int[] correctnessSizes = { 128, 256, 512, 1024 };

        foreach (int size in correctnessSizes)
        {
            var a = BenchmarkHelper.CreateRandomMatrix(size, size, random);
            var b = BenchmarkHelper.CreateRandomMatrix(size, size, random);

            Console.WriteLine($"Correctness check (safe): {size}x{size}");
            SetKernelMode(forceSafe: true, forceUnsafe: false);
            var safeStats = CompareInternal(cpuEngine, gpuEngine, a, b);
            PrintStats(safeStats);

            if (size <= 512)
            {
                Console.WriteLine($"Correctness check (unsafe): {size}x{size}");
                SetKernelMode(forceSafe: false, forceUnsafe: true);
                var unsafeStats = CompareInternal(cpuEngine, gpuEngine, a, b);
                PrintStats(unsafeStats);
            }

            Console.WriteLine();
        }

        SetKernelMode(forceSafe: false, forceUnsafe: false);

        int[] perfSizes = { 1024, 2048 };
        foreach (int perfSize in perfSizes)
        {
            var perfA = BenchmarkHelper.CreateRandomMatrix(perfSize, perfSize, random);
            var perfB = BenchmarkHelper.CreateRandomMatrix(perfSize, perfSize, random);

            Console.WriteLine($"GPU performance: {perfSize}x{perfSize}");
            
            // Warmup pass
            _ = gpuEngine.MatrixMultiply(perfA, perfB);

            int perfIterations = perfSize >= 2048 ? 2 : 3;
            double totalMs = 0;
            for (int i = 0; i < perfIterations; i++)
            {
                var sw = Stopwatch.StartNew();
                _ = gpuEngine.MatrixMultiply(perfA, perfB);
                sw.Stop();
                totalMs += sw.Elapsed.TotalMilliseconds;
                Console.WriteLine($"  Iteration {i + 1}: {sw.Elapsed.TotalMilliseconds:F2}ms");
            }

            double avgMs = totalMs / perfIterations;
            double seconds = avgMs / 1000.0;
            double flops = 2.0 * perfSize * perfSize * perfSize;
            double gflops = flops / seconds / 1_000_000_000.0;
            Console.WriteLine($"  Average: {avgMs:F2}ms  Throughput: {gflops:F2} GFLOPS");
            Console.WriteLine();
        }
    }

    private static (double maxError, double avgError, int nonFiniteCount) CompareInternal(
        CpuEngine cpuEngine,
        IEngine gpuEngine,
        Matrix<float> a,
        Matrix<float> b)
    {
        var cpuResult = cpuEngine.MatrixMultiply(a, b);
        var gpuResult = gpuEngine.MatrixMultiply(a, b);

        return BenchmarkHelper.Compare(cpuResult, gpuResult);
    }

    private static void PrintStats((double maxError, double avgError, int nonFiniteCount) stats)
    {
        Console.WriteLine($"  Max error: {stats.maxError:E3}");
        Console.WriteLine($"  Avg error: {stats.avgError:E3}");
        Console.WriteLine($"  Non-finite GPU values: {stats.nonFiniteCount}");
    }

    private static void SetKernelMode(bool forceSafe, bool forceUnsafe)
    {
        Environment.SetEnvironmentVariable("AIDOTNET_GEMM_SAFE", forceSafe ? "1" : "0");
        Environment.SetEnvironmentVariable("AIDOTNET_GEMM_UNSAFE", forceUnsafe ? "1" : "0");
    }
}