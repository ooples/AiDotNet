#if !NET462
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmark comparing cuBLAS GEMM vs DirectGpu for matrix multiplication.
/// Target: ~30,000 GFLOPS for cuBLAS.
/// </summary>
public static class CuBlasGemmBenchmark
{
    public static void Run()
    {
        Console.WriteLine("=== cuBLAS vs DirectGpu GEMM Benchmark ===");
        Console.WriteLine();

        // Check availability
        using var directEngine = new DirectGpuTensorEngine();
        Console.WriteLine($"cuBLAS Available: {CuBlasMatMul.IsAvailable}");
        Console.WriteLine($"DirectGpu Engine: {directEngine.Name}");
        Console.WriteLine($"DirectGpu Available: {directEngine.SupportsGpu}");
        Console.WriteLine();

        if (!CuBlasMatMul.IsAvailable)
        {
            Console.WriteLine("cuBLAS not available - NVIDIA drivers may not be installed.");
            Console.WriteLine("Will only run DirectGpu benchmarks if available.");
            Console.WriteLine();
        }
        else if (!directEngine.SupportsGpu)
        {
            Console.WriteLine("DirectGpu not available - will only run cuBLAS benchmarks.");
            Console.WriteLine();
        }

        // Test various sizes typical for neural network layers
        int[] sizes = { 256, 512, 1024, 2048, 4096 };

        Console.WriteLine("Matrix Multiplication (C = A × B, square matrices):");
        Console.WriteLine("Size       |  cuBLAS (GFLOPS)  |  DirectGpu (GFLOPS)  |  Speedup");
        Console.WriteLine(new string('-', 65));

        foreach (int size in sizes)
        {
            BenchmarkSize(directEngine, size);
        }

        Console.WriteLine();
        Console.WriteLine("DenseLayer-style operation (batch=64, in=768, out=3072):");
        BenchmarkDenseLayerStyle(directEngine);
    }

    private static void BenchmarkSize(DirectGpuTensorEngine directEngine, int size)
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

        // Benchmark DirectGpu
        double directGpuGflops = BenchmarkDirectGpu(directEngine, A, m, k, B, n, flops);

        // Calculate speedup
        string speedup = (cublasGflops > 0 && directGpuGflops > 0)
            ? $"{cublasGflops / directGpuGflops:F1}x"
            : "N/A";

        Console.WriteLine($"{size,5}x{size,-4} |  {cublasGflops,12:F1}     |  {directGpuGflops,12:F1}    |  {speedup}");
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

    private static double BenchmarkDirectGpu(DirectGpuTensorEngine directEngine, float[] A, int m, int k, float[] B, int n, double flops)
    {
        try
        {
            if (!directEngine.SupportsGpu)
                return 0;

            // Create tensors (Tensor constructor takes Vector<T>, not float[])
            var tensorA = new Tensor<float>(new int[] { m, k }, new Vector<float>(A));
            var tensorB = new Tensor<float>(new int[] { k, n }, new Vector<float>(B));

            // Warmup
            for (int i = 0; i < 3; i++)
            {
                var result = directEngine.TensorMatMul(tensorA, tensorB);
            }

            // Benchmark
            const int iterations = 10;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                directEngine.TensorMatMul(tensorA, tensorB);
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            return (flops / seconds) / 1e9; // GFLOPS
        }
        catch (Exception ex)
        {
            Console.WriteLine($"DirectGpu error: {ex.Message}");
            return 0;
        }
    }

    private static void BenchmarkDenseLayerStyle(DirectGpuTensorEngine directEngine)
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

        // DirectGpu benchmark
        double directGpuGflops = 0;
        try
        {
            if (!directEngine.SupportsGpu)
            {
                throw new InvalidOperationException("DirectGpu not available.");
            }

            var tensorInput = new Tensor<float>(new int[] { batch, inputSize }, new Vector<float>(input));
            var tensorWeightsT = new Tensor<float>(new int[] { inputSize, outputSize }, new Vector<float>(weightsT));

            // Warmup
            for (int i = 0; i < 3; i++)
            {
                directEngine.TensorMatMul(tensorInput, tensorWeightsT);
            }

            const int iterations = 50;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                directEngine.TensorMatMul(tensorInput, tensorWeightsT);
            }
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds / iterations;
            directGpuGflops = (flops / seconds) / 1e9;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"DirectGpu error: {ex.Message}");
        }

        string speedup = (cublasGflops > 0 && directGpuGflops > 0)
            ? $"{cublasGflops / directGpuGflops:F1}x"
            : "N/A";

        Console.WriteLine($"cuBLAS: {cublasGflops:F1} GFLOPS");
        Console.WriteLine($"DirectGpu:  {directGpuGflops:F1} GFLOPS");
        Console.WriteLine($"Speedup: {speedup}");

        // Performance analysis
        Console.WriteLine();
        Console.WriteLine("Performance Analysis:");
        Console.WriteLine($"  Target: ~30,000 GFLOPS (cuBLAS theoretical)");
        Console.WriteLine($"  PyTorch/TorchSharp baseline: ~10,000 GFLOPS");

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
