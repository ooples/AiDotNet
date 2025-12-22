using System;
using System.Runtime.Intrinsics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.Operators;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks comparing scalar vs SIMD performance for trigonometric operations.
/// </summary>
/// <remarks>
/// Target: Achieve 8-12x speedup for SIMD operations vs scalar Math.Sin/Cos.
/// This benchmark measures operations on arrays of various sizes to demonstrate
/// the performance characteristics of SIMD acceleration.
/// </remarks>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[MarkdownExporter]
public class TrigonometricOperatorBenchmarks
{
    private double[] _inputDouble = null!;
    private float[] _inputFloat = null!;
    private double[] _outputDouble = null!;
    private float[] _outputFloat = null!;

    [Params(100, 1000, 10000)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Initialize input arrays with random values in [-π, π]
        _inputDouble = new double[N];
        _inputFloat = new float[N];
        _outputDouble = new double[N];
        _outputFloat = new float[N];

        for (int i = 0; i < N; i++)
        {
            _inputDouble[i] = (random.NextDouble() * 2 - 1) * Math.PI;
            _inputFloat[i] = (float)((random.NextDouble() * 2 - 1) * Math.PI);
        }
    }

    #region Sine Benchmarks - Double

    [Benchmark(Description = "Sin (Scalar Math.Sin) - Double", Baseline = true)]
    public void SinScalarDouble()
    {
        for (int i = 0; i < N; i++)
        {
            _outputDouble[i] = Math.Sin(_inputDouble[i]);
        }
    }

    [Benchmark(Description = "Sin (SIMD Operator) - Double")]
    public void SinSimdDouble()
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(
            _inputDouble.AsSpan(),
            _outputDouble.AsSpan());
    }

    #endregion

    #region Sine Benchmarks - Float

    [Benchmark(Description = "Sin (Scalar MathF.Sin) - Float")]
    public void SinScalarFloat()
    {
        for (int i = 0; i < N; i++)
        {
            _outputFloat[i] = MathF.Sin(_inputFloat[i]);
        }
    }

    [Benchmark(Description = "Sin (SIMD Operator) - Float")]
    public void SinSimdFloat()
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(
            _inputFloat.AsSpan(),
            _outputFloat.AsSpan());
    }

    #endregion

    #region Cosine Benchmarks - Double

    [Benchmark(Description = "Cos (Scalar Math.Cos) - Double")]
    public void CosScalarDouble()
    {
        for (int i = 0; i < N; i++)
        {
            _outputDouble[i] = Math.Cos(_inputDouble[i]);
        }
    }

    [Benchmark(Description = "Cos (SIMD Operator) - Double")]
    public void CosSimdDouble()
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(
            _inputDouble.AsSpan(),
            _outputDouble.AsSpan());
    }

    #endregion

    #region Cosine Benchmarks - Float

    [Benchmark(Description = "Cos (Scalar MathF.Cos) - Float")]
    public void CosScalarFloat()
    {
        for (int i = 0; i < N; i++)
        {
            _outputFloat[i] = MathF.Cos(_inputFloat[i]);
        }
    }

    [Benchmark(Description = "Cos (SIMD Operator) - Float")]
    public void CosSimdFloat()
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(
            _inputFloat.AsSpan(),
            _outputFloat.AsSpan());
    }

    #endregion
}
