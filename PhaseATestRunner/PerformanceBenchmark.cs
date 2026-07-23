using System.Diagnostics;
using AiDotNet.Engines;
using AiDotNet.LinearAlgebra;
using AiDotNet.Prototypes;
using AiDotNet.Tensors.Helpers;

namespace PhaseATestRunner;

/// <summary>
/// Comprehensive performance benchmark comparing:
/// 1. Original Vector (CPU only)
/// 2. PrototypeVector with CPU engine
/// 3. PrototypeVector with GPU engine
/// </summary>
public static class PerformanceBenchmark
{
    public static void RunComparison()
    {
        Console.WriteLine("================================================================================");
        Console.WriteLine("PERFORMANCE BENCHMARK: Original Vector vs PrototypeVector");
        Console.WriteLine("================================================================================");
        Console.WriteLine();

        int[] sizes = { 1000, 10000, 100000, 1000000 };
        int iterations = 100;

        Console.WriteLine($"Test Configuration:");
        Console.WriteLine($"  Iterations per size: {iterations}");
        Console.WriteLine($"  Operations tested: Add, Multiply, Dot Product");
        Console.WriteLine();

        foreach (var size in sizes)
        {
            Console.WriteLine($"Vector Size: {size:N0} elements");
            Console.WriteLine("─".PadRight(80, '─'));

            BenchmarkSize(size, iterations);
            Console.WriteLine();
        }

        Console.WriteLine("================================================================================");
        Console.WriteLine("BENCHMARK COMPLETE");
        Console.WriteLine("================================================================================");
    }

    private static void BenchmarkSize(int size, int iterations)
    {
        // Prepare test data
        var data1 = new float[size];
        var data2 = new float[size];
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < size; i++)
        {
            data1[i] = (float)random.NextDouble();
            data2[i] = (float)random.NextDouble();
        }

        // Original Vector
        var originalVec1 = new Vector<float>(data1);
        var originalVec2 = new Vector<float>(data2);

        // PrototypeVector (will be used with different engines)
        var protoVec1 = PrototypeVector<float>.FromArray(data1);
        var protoVec2 = PrototypeVector<float>.FromArray(data2);

        // ═══════════════════════════════════════════════════════════════════════════════
        // Benchmark 1: Vector Addition
        // ═══════════════════════════════════════════════════════════════════════════════

        // Original Vector (CPU only)
        var originalAddTime = BenchmarkOriginalAdd(originalVec1, originalVec2, iterations);

        // PrototypeVector with CPU engine
        AiDotNetEngine.ResetToCpu();
        var protoCpuAddTime = BenchmarkPrototypeAdd(protoVec1, protoVec2, iterations);

        // PrototypeVector with GPU engine (if available)
        double protoGpuAddTime = -1;
        if (AiDotNetEngine.AutoDetectAndConfigureGpu())
        {
            protoGpuAddTime = BenchmarkPrototypeAdd(protoVec1, protoVec2, iterations);
        }

        // ═══════════════════════════════════════════════════════════════════════════════
        // Benchmark 2: Scalar Multiplication
        // ═══════════════════════════════════════════════════════════════════════════════

        // Original Vector
        var originalMulTime = BenchmarkOriginalScalarMultiply(originalVec1, 2.5f, iterations);

        // PrototypeVector with CPU
        AiDotNetEngine.ResetToCpu();
        var protoCpuMulTime = BenchmarkPrototypeScalarMultiply(protoVec1, 2.5f, iterations);

        // PrototypeVector with GPU
        double protoGpuMulTime = -1;
        if (AiDotNetEngine.AutoDetectAndConfigureGpu())
        {
            protoGpuMulTime = BenchmarkPrototypeScalarMultiply(protoVec1, 2.5f, iterations);
        }


        // ═══════════════════════════════════════════════════════════════════════════════
        // Display Results
        // ═══════════════════════════════════════════════════════════════════════════════

        Console.WriteLine("VECTOR ADDITION:");
        Console.WriteLine($"  Original Vector (CPU):       {originalAddTime:F3} ms");
        Console.WriteLine($"  PrototypeVector (CPU):       {protoCpuAddTime:F3} ms  ({(originalAddTime / protoCpuAddTime):F2}x)");
        if (protoGpuAddTime > 0)
            Console.WriteLine($"  PrototypeVector (GPU):       {protoGpuAddTime:F3} ms  ({(originalAddTime / protoGpuAddTime):F2}x)");
        else
            Console.WriteLine($"  PrototypeVector (GPU):       N/A (no GPU)");
        Console.WriteLine();

        Console.WriteLine("SCALAR MULTIPLICATION:");
        Console.WriteLine($"  Original Vector (CPU):       {originalMulTime:F3} ms");
        Console.WriteLine($"  PrototypeVector (CPU):       {protoCpuMulTime:F3} ms  ({(originalMulTime / protoCpuMulTime):F2}x)");
        if (protoGpuMulTime > 0)
            Console.WriteLine($"  PrototypeVector (GPU):       {protoGpuMulTime:F3} ms  ({(originalMulTime / protoGpuMulTime):F2}x)");
        else
            Console.WriteLine($"  PrototypeVector (GPU):       N/A (no GPU)");
        Console.WriteLine();
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // Original Vector Benchmarks
    // ═══════════════════════════════════════════════════════════════════════════════════

    private static double BenchmarkOriginalAdd(Vector<float> a, Vector<float> b, int iterations)
    {
        // Warmup
        for (int i = 0; i < 3; i++)
            _ = a + b;

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            _ = a + b;
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / iterations;
    }

    private static double BenchmarkOriginalScalarMultiply(Vector<float> a, float scalar, int iterations)
    {
        // Warmup
        for (int i = 0; i < 3; i++)
            _ = a * scalar;

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            _ = a * scalar;
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / iterations;
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    // PrototypeVector Benchmarks
    // ═══════════════════════════════════════════════════════════════════════════════════

    private static double BenchmarkPrototypeAdd(PrototypeVector<float> a, PrototypeVector<float> b, int iterations)
    {
        // Warmup
        for (int i = 0; i < 3; i++)
            _ = a.Add(b);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            _ = a.Add(b);
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / iterations;
    }

    private static double BenchmarkPrototypeScalarMultiply(PrototypeVector<float> a, float scalar, int iterations)
    {
        // Warmup
        for (int i = 0; i < 3; i++)
            _ = a.Multiply(scalar);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            _ = a.Multiply(scalar);
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / iterations;
    }
}
