using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.Benchmarks;

public static class CpuMatMulDiagnostics
{
    public static void Run()
    {
        Console.WriteLine("=== CPU Matrix Multiply Diagnostics ===");
        Console.WriteLine($"ProcessorCount: {Environment.ProcessorCount}");
        Console.WriteLine(TensorPrimitivesCore.GetHardwareAccelerationInfo());
        Console.WriteLine();

        var engine = new CpuEngine();
        int[] sizes = GetSizes();
        const int warmupIterations = 1;

        foreach (int size in sizes)
        {
            Console.WriteLine($"{size}x{size} MatMul");
            Console.WriteLine(new string('-', 48));

            var random = RandomHelper.CreateSeededRandom(42);
            var a = CreateRandomMatrix(size, size, random);
            var b = CreateRandomMatrix(size, size, random);

            for (int i = 0; i < warmupIterations; i++)
            {
                _ = engine.MatrixMultiply(a, b);
            }

            int benchmarkIterations = size >= 2048 ? 2 : 3;
            double totalMs = 0;
            double checksum = 0;
            for (int i = 0; i < benchmarkIterations; i++)
            {
                var sw = Stopwatch.StartNew();
                var result = engine.MatrixMultiply(a, b);
                sw.Stop();

                totalMs += sw.Elapsed.TotalMilliseconds;
                checksum += result[0, 0];
                Console.WriteLine($"  Iteration {i + 1}: {sw.Elapsed.TotalMilliseconds:F2}ms");
            }

            double avgMs = totalMs / benchmarkIterations;
            double seconds = avgMs / 1000.0;
            double flops = 2.0 * size * size * size;
            double gflops = flops / seconds / 1_000_000_000.0;
            Console.WriteLine($"  Average: {avgMs:F2}ms  Throughput: {gflops:F2} GFLOPS  Checksum: {checksum:F3}");
            Console.WriteLine();
        }
    }

    private static int[] GetSizes()
    {
        string? sizesEnv = Environment.GetEnvironmentVariable("AIDOTNET_CPU_MATMUL_SIZES");
        if (!string.IsNullOrWhiteSpace(sizesEnv))
        {
            var tokens = sizesEnv.Split(new[] { ',', ';', ' ', '|' }, StringSplitOptions.RemoveEmptyEntries);
            var sizes = new List<int>();
            foreach (var token in tokens)
            {
                if (int.TryParse(token, out int size) && size > 0)
                    sizes.Add(size);
            }

            if (sizes.Count > 0)
                return sizes.ToArray();
        }

        return new[] { 256, 512, 1024, 2048 };
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
}
