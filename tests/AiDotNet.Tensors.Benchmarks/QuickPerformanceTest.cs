using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Operators;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Quick performance validation test to verify SIMD speedup without full BenchmarkDotNet overhead.
/// </summary>
public static class QuickPerformanceTest
{
    public static void Run()
    {
        Console.WriteLine("=== Quick Performance Validation ===");
        Console.WriteLine();
        Console.WriteLine($"Hardware Acceleration Status:");
        Console.WriteLine(TensorPrimitivesCore.GetHardwareAccelerationInfo());
        Console.WriteLine();

        int[] sizes = { 1000, 10000, 100000 };

        foreach (int size in sizes)
        {
            Console.WriteLine($"Array Size: {size:N0} elements");
            Console.WriteLine(new string('-', 60));

            TestSinDouble(size);
            TestSinFloat(size);
            TestCosDouble(size);
            TestCosFloat(size);

            Console.WriteLine();
        }
    }

    private static void TestSinDouble(int size)
    {
        var random = new Random(42);
        var input = new double[size];
        var outputScalar = new double[size];
        var outputSimd = new double[size];

        for (int i = 0; i < size; i++)
        {
            input[i] = (random.NextDouble() * 2 - 1) * Math.PI;
        }

        // Warmup
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < size; j++) outputScalar[j] = Math.Sin(input[j]);
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(input.AsSpan(), outputSimd.AsSpan());
        }

        // Scalar benchmark
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < size; j++)
            {
                outputScalar[j] = Math.Sin(input[j]);
            }
        }
        sw.Stop();
        double scalarTime = sw.Elapsed.TotalMilliseconds;

        // SIMD benchmark
        sw.Restart();
        for (int i = 0; i < 100; i++)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(
                input.AsSpan(),
                outputSimd.AsSpan());
        }
        sw.Stop();
        double simdTime = sw.Elapsed.TotalMilliseconds;

        double speedup = scalarTime / simdTime;
        Console.WriteLine($"  Sin(double):  Scalar={scalarTime:F2}ms  SIMD={simdTime:F2}ms  Speedup={speedup:F2}x");
    }

    private static void TestSinFloat(int size)
    {
        var random = new Random(42);
        var input = new float[size];
        var outputScalar = new float[size];
        var outputSimd = new float[size];

        for (int i = 0; i < size; i++)
        {
            input[i] = (float)((random.NextDouble() * 2 - 1) * Math.PI);
        }

        // Warmup
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < size; j++) outputScalar[j] = MathF.Sin(input[j]);
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(input.AsSpan(), outputSimd.AsSpan());
        }

        // Scalar benchmark
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < size; j++)
            {
                outputScalar[j] = MathF.Sin(input[j]);
            }
        }
        sw.Stop();
        double scalarTime = sw.Elapsed.TotalMilliseconds;

        // SIMD benchmark
        sw.Restart();
        for (int i = 0; i < 100; i++)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(
                input.AsSpan(),
                outputSimd.AsSpan());
        }
        sw.Stop();
        double simdTime = sw.Elapsed.TotalMilliseconds;

        double speedup = scalarTime / simdTime;
        Console.WriteLine($"  Sin(float):   Scalar={scalarTime:F2}ms  SIMD={simdTime:F2}ms  Speedup={speedup:F2}x");
    }

    private static void TestCosDouble(int size)
    {
        var random = new Random(42);
        var input = new double[size];
        var outputScalar = new double[size];
        var outputSimd = new double[size];

        for (int i = 0; i < size; i++)
        {
            input[i] = (random.NextDouble() * 2 - 1) * Math.PI;
        }

        // Warmup
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < size; j++) outputScalar[j] = Math.Cos(input[j]);
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(input.AsSpan(), outputSimd.AsSpan());
        }

        // Scalar benchmark
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < size; j++)
            {
                outputScalar[j] = Math.Cos(input[j]);
            }
        }
        sw.Stop();
        double scalarTime = sw.Elapsed.TotalMilliseconds;

        // SIMD benchmark
        sw.Restart();
        for (int i = 0; i < 100; i++)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(
                input.AsSpan(),
                outputSimd.AsSpan());
        }
        sw.Stop();
        double simdTime = sw.Elapsed.TotalMilliseconds;

        double speedup = scalarTime / simdTime;
        Console.WriteLine($"  Cos(double):  Scalar={scalarTime:F2}ms  SIMD={simdTime:F2}ms  Speedup={speedup:F2}x");
    }

    private static void TestCosFloat(int size)
    {
        var random = new Random(42);
        var input = new float[size];
        var outputScalar = new float[size];
        var outputSimd = new float[size];

        for (int i = 0; i < size; i++)
        {
            input[i] = (float)((random.NextDouble() * 2 - 1) * Math.PI);
        }

        // Warmup
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < size; j++) outputScalar[j] = MathF.Cos(input[j]);
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(input.AsSpan(), outputSimd.AsSpan());
        }

        // Scalar benchmark
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < size; j++)
            {
                outputScalar[j] = MathF.Cos(input[j]);
            }
        }
        sw.Stop();
        double scalarTime = sw.Elapsed.TotalMilliseconds;

        // SIMD benchmark
        sw.Restart();
        for (int i = 0; i < 100; i++)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(
                input.AsSpan(),
                outputSimd.AsSpan());
        }
        sw.Stop();
        double simdTime = sw.Elapsed.TotalMilliseconds;

        double speedup = scalarTime / simdTime;
        Console.WriteLine($"  Cos(float):   Scalar={scalarTime:F2}ms  SIMD={simdTime:F2}ms  Speedup={speedup:F2}x");
    }
}
