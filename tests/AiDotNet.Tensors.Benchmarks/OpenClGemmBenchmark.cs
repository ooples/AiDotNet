#if !NET462
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmark for OpenCL GEMM on AMD/Intel GPUs.
/// Target: EXCEED competitors - 30,000+ GFLOPS on high-end AMD GPUs.
/// </summary>
public static class OpenClGemmBenchmark
{
    public static void Run()
    {
        Console.WriteLine("=== OpenCL GEMM Benchmark (AMD/Intel GPUs) ===");
        Console.WriteLine();
        Console.WriteLine("Target: EXCEED PyTorch/cuBLAS performance on non-NVIDIA hardware");
        Console.WriteLine();

        // Print system info
        Console.WriteLine(OpenClInfo.GetSystemInfo());

        if (!OpenClContext.IsAvailable)
        {
            Console.WriteLine("OpenCL not available - no compatible GPU found.");
            return;
        }

        try
        {
            using var context = new OpenClContext();
            Console.WriteLine($"Using device: {context.DeviceName}");
            Console.WriteLine($"Vendor: {context.DeviceVendor}");
            Console.WriteLine($"Compute Units: {context.MaxComputeUnits}");
            Console.WriteLine($"Global Memory: {context.GlobalMemSize / (1024.0 * 1024 * 1024):F2} GB");
            Console.WriteLine($"Local Memory: {context.LocalMemSize / 1024.0:F0} KB");
            Console.WriteLine();

            // Test various sizes typical for neural network layers
            int[] sizes = { 256, 512, 1024, 2048, 4096 };

            Console.WriteLine("Matrix Multiplication (C = A × B, square matrices):");
            Console.WriteLine("Size       |  OpenCL (GFLOPS)  |  Status");
            Console.WriteLine(new string('-', 55));

            using var matmul = new OpenClMatMul(context);

            Console.WriteLine($"CLBlast Available: {ClBlastNative.IsAvailable}");
            Console.WriteLine($"Using CLBlast: {matmul.UsingClBlast}");
            Console.WriteLine();

            foreach (int size in sizes)
            {
                BenchmarkSize(matmul, size);
            }

            Console.WriteLine();
            Console.WriteLine("DenseLayer-style operation (batch=64, in=768, out=3072):");
            BenchmarkDenseLayerStyle(matmul);

            Console.WriteLine();
            Console.WriteLine("Large matrix test (batch=128, in=4096, out=4096):");
            BenchmarkLargeMatrix(matmul);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    private static void BenchmarkSize(OpenClMatMul matmul, int size)
    {
        int m = size, k = size, n = size;

        // Generate test data
        var random = new Random(42);
        var A = new float[m * k];
        var B = new float[k * n];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < B.Length; i++) B[i] = (float)(random.NextDouble() * 2 - 1);

        // FLOPS for GEMM: 2 * m * n * k (multiply-add per output element)
        double flops = 2.0 * m * n * k;

        try
        {
            // Warmup
            for (int i = 0; i < 3; i++)
            {
                var result = matmul.MatMulFloat(A, m, k, B, n);
                if (result == null)
                {
                    string error = matmul.LastError ?? "Unknown error";
                    Console.WriteLine($"{size,5}x{size,-4} |  FAILED           |  {error}");
                    return;
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
            double gflops = (flops / seconds) / 1e9;

            string status = GetPerformanceStatus(gflops, size);
            Console.WriteLine($"{size,5}x{size,-4} |  {gflops,12:F1}     |  {status}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"{size,5}x{size,-4} |  ERROR            |  {ex.Message}");
        }
    }

    private static void BenchmarkDenseLayerStyle(OpenClMatMul matmul)
    {
        // Typical transformer layer dimensions
        int batch = 64;
        int inputSize = 768;
        int outputSize = 3072;

        Console.WriteLine($"Batch={batch}, InputSize={inputSize}, OutputSize={outputSize}");

        var random = new Random(42);

        // input: [batch, inputSize]
        var input = new float[batch * inputSize];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);

        // weights: [inputSize, outputSize]
        var weights = new float[inputSize * outputSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 0.1 - 0.05);

        // FLOPS for matmul
        double flops = 2.0 * batch * inputSize * outputSize;

        try
        {
            // Warmup
            for (int i = 0; i < 3; i++)
            {
                matmul.DenseForward(input, weights, batch, inputSize, outputSize);
            }

            const int iterations = 50;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                matmul.DenseForward(input, weights, batch, inputSize, outputSize);
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            double gflops = (flops / seconds) / 1e9;

            Console.WriteLine($"OpenCL: {gflops:F1} GFLOPS");

            // Performance analysis
            Console.WriteLine();
            Console.WriteLine("Performance Analysis vs Competitors:");
            Console.WriteLine($"  PyTorch/TorchSharp baseline: ~10,000 GFLOPS");
            Console.WriteLine($"  cuBLAS peak: ~30,000 GFLOPS");
            Console.WriteLine($"  Our target: EXCEED 30,000 GFLOPS");

            if (gflops > 30000)
            {
                Console.WriteLine($"  ★★★ EXCEEDING cuBLAS at {gflops:F0} GFLOPS - MISSION ACCOMPLISHED! ★★★");
            }
            else if (gflops > 10000)
            {
                Console.WriteLine($"  ★★ EXCEEDING PyTorch at {gflops:F0} GFLOPS - Good progress!");
            }
            else if (gflops > 1000)
            {
                Console.WriteLine($"  ★ {gflops:F0} GFLOPS - Room for improvement, but beating ILGPU");
            }
            else
            {
                Console.WriteLine($"  {gflops:F0} GFLOPS - Needs optimization work");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"OpenCL error: {ex.Message}");
        }
    }

    private static void BenchmarkLargeMatrix(OpenClMatMul matmul)
    {
        int batch = 128;
        int size = 4096;

        Console.WriteLine($"Batch={batch}, Size={size}x{size}");

        var random = new Random(42);

        var input = new float[batch * size];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);

        var weights = new float[size * size];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 0.1 - 0.05);

        double flops = 2.0 * batch * size * size;

        try
        {
            // Warmup
            for (int i = 0; i < 2; i++)
            {
                matmul.DenseForward(input, weights, batch, size, size);
            }

            const int iterations = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                matmul.DenseForward(input, weights, batch, size, size);
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            double gflops = (flops / seconds) / 1e9;

            Console.WriteLine($"OpenCL: {gflops:F1} GFLOPS");
            Console.WriteLine($"Memory bandwidth utilized: {(batch * size + size * size + batch * size) * 4.0 / seconds / 1e9:F1} GB/s");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"OpenCL error: {ex.Message}");
        }
    }

    private static string GetPerformanceStatus(double gflops, int size)
    {
        // Expected GFLOPS varies by matrix size (smaller matrices have more overhead)
        double expectedPeak = size switch
        {
            256 => 5000,
            512 => 10000,
            1024 => 15000,
            2048 => 20000,
            4096 => 25000,
            _ => 10000
        };

        if (gflops > expectedPeak * 1.2)
            return "★★★ EXCEEDING TARGET";
        else if (gflops > expectedPeak * 0.8)
            return "★★ ON TARGET";
        else if (gflops > expectedPeak * 0.5)
            return "★ ACCEPTABLE";
        else if (gflops > 100)
            return "NEEDS OPTIMIZATION";
        else
            return "BASELINE";
    }
}
#endif
