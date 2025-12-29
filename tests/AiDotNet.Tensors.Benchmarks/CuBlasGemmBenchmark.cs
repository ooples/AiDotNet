#if !NET462
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmark comparing cuBLAS GEMM vs ILGPU for matrix multiplication.
/// Target: ~30,000 GFLOPS for cuBLAS vs ~52-86 GFLOPS for ILGPU.
/// </summary>
public static class CuBlasGemmBenchmark
{
    public static void Run()
    {
        Console.WriteLine("=== cuBLAS vs ILGPU GEMM Benchmark ===");
        Console.WriteLine();

        // Check availability
        Console.WriteLine($"cuBLAS Available: {CuBlasMatMul.IsAvailable}");
        Console.WriteLine($"ILGPU Engine: {Engine.Default.Name}");
        Console.WriteLine();

        if (!CuBlasMatMul.IsAvailable)
        {
            Console.WriteLine("cuBLAS not available - NVIDIA drivers may not be installed.");
            Console.WriteLine("Will only run ILGPU benchmarks.");
            Console.WriteLine();
        }

        // Test various sizes typical for neural network layers
        int[] sizes = { 256, 512, 1024, 2048, 4096 };

        Console.WriteLine("Matrix Multiplication (C = A × B, square matrices):");
        Console.WriteLine("Size       |  cuBLAS (GFLOPS)  |  ILGPU (GFLOPS)  |  Speedup");
        Console.WriteLine(new string('-', 65));

        foreach (int size in sizes)
        {
            BenchmarkSize(size);
        }

        Console.WriteLine();
        Console.WriteLine("DenseLayer-style operation (batch=64, in=768, out=3072):");
        BenchmarkDenseLayerStyle();
    }

    private static void BenchmarkSize(int size)
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

        // Benchmark cuBLAS
        double cublasGflops = BenchmarkCublas(A, m, k, B, k, n, flops);

        // Benchmark ILGPU
        double ilgpuGflops = BenchmarkIlgpu(A, m, k, B, n, flops);

        // Calculate speedup
        string speedup = (cublasGflops > 0 && ilgpuGflops > 0)
            ? $"{cublasGflops / ilgpuGflops:F1}x"
            : "N/A";

        Console.WriteLine($"{size,5}x{size,-4} |  {cublasGflops,12:F1}     |  {ilgpuGflops,12:F1}    |  {speedup}");
    }

    private static double BenchmarkCublas(float[] A, int m, int k, float[] B, int kB, int n, double flops)
    {
        if (!CuBlasMatMul.IsAvailable) return 0;

        try
        {
            using var cublas = new CuBlasMatMul();

            // Warmup
            for (int i = 0; i < 3; i++)
            {
                var result = cublas.MatMulFloat(A, m, k, B, kB, n);
                if (result == null) return 0;
            }

            // Benchmark
            const int iterations = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                cublas.MatMulFloat(A, m, k, B, kB, n);
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            return (flops / seconds) / 1e9; // GFLOPS
        }
        catch (Exception ex)
        {
            Console.WriteLine($"cuBLAS error: {ex.Message}");
            return 0;
        }
    }

    private static double BenchmarkIlgpu(float[] A, int m, int k, float[] B, int n, double flops)
    {
        try
        {
            // Create tensors (Tensor constructor takes Vector<T>, not float[])
            var tensorA = new Tensor<float>(new int[] { m, k }, new Vector<float>(A));
            var tensorB = new Tensor<float>(new int[] { k, n }, new Vector<float>(B));

            // Warmup
            for (int i = 0; i < 3; i++)
            {
                var result = Engine.Default.TensorMatMul(tensorA, tensorB);
            }

            // Benchmark
            const int iterations = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                Engine.Default.TensorMatMul(tensorA, tensorB);
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            return (flops / seconds) / 1e9; // GFLOPS
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ILGPU error: {ex.Message}");
            return 0;
        }
    }

    private static void BenchmarkDenseLayerStyle()
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

        // weights transposed: [inputSize, outputSize]
        var weightsT = new float[inputSize * outputSize];
        for (int i = 0; i < weightsT.Length; i++) weightsT[i] = (float)(random.NextDouble() * 0.1 - 0.05);

        // FLOPS for matmul
        double flops = 2.0 * batch * inputSize * outputSize;

        // cuBLAS benchmark
        double cublasGflops = 0;
        if (CuBlasMatMul.IsAvailable)
        {
            try
            {
                using var cublas = new CuBlasMatMul();

                // Warmup
                for (int i = 0; i < 3; i++)
                {
                    cublas.MatMulFloat(input, batch, inputSize, weightsT, inputSize, outputSize);
                }

                const int iterations = 50;
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                {
                    cublas.MatMulFloat(input, batch, inputSize, weightsT, inputSize, outputSize);
                }
                sw.Stop();

                double seconds = sw.Elapsed.TotalSeconds / iterations;
                cublasGflops = (flops / seconds) / 1e9;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"cuBLAS error: {ex.Message}");
            }
        }

        // ILGPU benchmark
        double ilgpuGflops = 0;
        try
        {
            var tensorInput = new Tensor<float>(new int[] { batch, inputSize }, new Vector<float>(input));
            var tensorWeightsT = new Tensor<float>(new int[] { inputSize, outputSize }, new Vector<float>(weightsT));

            // Warmup
            for (int i = 0; i < 3; i++)
            {
                Engine.Default.TensorMatMul(tensorInput, tensorWeightsT);
            }

            const int iterations = 50;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                Engine.Default.TensorMatMul(tensorInput, tensorWeightsT);
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            ilgpuGflops = (flops / seconds) / 1e9;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ILGPU error: {ex.Message}");
        }

        string speedup = (cublasGflops > 0 && ilgpuGflops > 0)
            ? $"{cublasGflops / ilgpuGflops:F1}x"
            : "N/A";

        Console.WriteLine($"cuBLAS: {cublasGflops:F1} GFLOPS");
        Console.WriteLine($"ILGPU:  {ilgpuGflops:F1} GFLOPS");
        Console.WriteLine($"Speedup: {speedup}");

        // Performance analysis
        Console.WriteLine();
        Console.WriteLine("Performance Analysis:");
        Console.WriteLine($"  Target: ~30,000 GFLOPS (cuBLAS theoretical)");
        Console.WriteLine($"  PyTorch/TorchSharp baseline: ~10,000 GFLOPS");
        Console.WriteLine($"  Current ILGPU ceiling: ~52-86 GFLOPS");

        if (cublasGflops > 1000)
        {
            Console.WriteLine($"  ✓ cuBLAS achieving {cublasGflops:F0} GFLOPS - SUCCESS!");
        }
        else if (cublasGflops > 100)
        {
            Console.WriteLine($"  ! cuBLAS at {cublasGflops:F0} GFLOPS - Good but room for improvement");
        }
        else
        {
            Console.WriteLine($"  ✗ cuBLAS at {cublasGflops:F0} GFLOPS - Needs investigation");
        }
    }
}
#endif
