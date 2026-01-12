using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

public static class GpuMatMulDiagnostics
{
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
            var a = CreateRandomMatrix(size, size, random);
            var b = CreateRandomMatrix(size, size, random);

            Console.WriteLine($"Correctness check (safe): {size}x{size}");
            SetKernelMode(forceSafe: true, forceUnsafe: false);
            var safeStats = Compare(cpuEngine, gpuEngine, a, b);
            PrintStats(safeStats);

            if (size <= 512)
            {
                Console.WriteLine($"Correctness check (unsafe): {size}x{size}");
                SetKernelMode(forceSafe: false, forceUnsafe: true);
                var unsafeStats = Compare(cpuEngine, gpuEngine, a, b);
                PrintStats(unsafeStats);
            }

            Console.WriteLine();
        }

        SetKernelMode(forceSafe: false, forceUnsafe: false);

        int[] perfSizes = { 1024, 2048 };
        foreach (int perfSize in perfSizes)
        {
            var perfA = CreateRandomMatrix(perfSize, perfSize, random);
            var perfB = CreateRandomMatrix(perfSize, perfSize, random);

            Console.WriteLine($"GPU performance: {perfSize}x{perfSize}");
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

    private static Matrix<float> CreateRandomMatrix(int rows, int cols, Random random)
    {
        var matrix = new Matrix<float>(rows, cols);
        var data = matrix.AsWritableSpan();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)((random.NextDouble() * 2.0) - 1.0);
        }
        return matrix;
    }

    private static (double maxError, double avgError, int nonFiniteCount) Compare(
        CpuEngine cpuEngine,
        IEngine gpuEngine,
        Matrix<float> a,
        Matrix<float> b)
    {
        var cpuResult = cpuEngine.MatrixMultiply(a, b);
        var gpuResult = gpuEngine.MatrixMultiply(a, b);

        var cpuSpan = cpuResult.AsSpan();
        var gpuSpan = gpuResult.AsSpan();

        double maxError = 0;
        double sumError = 0;
        int count = 0;
        int nonFiniteCount = 0;

        for (int i = 0; i < gpuSpan.Length; i++)
        {
            float gpuValue = gpuSpan[i];
            if (float.IsNaN(gpuValue) || float.IsInfinity(gpuValue))
            {
                nonFiniteCount++;
                continue;
            }

            double error = Math.Abs(cpuSpan[i] - gpuValue);
            sumError += error;
            if (error > maxError)
                maxError = error;
            count++;
        }

        double avgError = count > 0 ? sumError / count : double.NaN;
        return (maxError, avgError, nonFiniteCount);
    }

    private static void PrintStats((double maxError, double avgError, int nonFiniteCount) stats)
    {
        Console.WriteLine($"  Max error: {stats.maxError:E3}");
        Console.WriteLine($"  Avg error: {stats.avgError:E3}");
        Console.WriteLine($"  Non-finite GPU values: {stats.nonFiniteCount}");
    }

    private static void SetKernelMode(bool forceSafe, bool forceUnsafe)
    {
        Environment.SetEnvironmentVariable("AIDOTNET_GEMM_SAFE", forceSafe ? "1" : null);
        Environment.SetEnvironmentVariable("AIDOTNET_GEMM_UNSAFE", forceUnsafe ? "1" : null);
    }
}
