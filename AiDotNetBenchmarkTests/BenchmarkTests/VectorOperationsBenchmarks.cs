using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Accord.Math;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Vector operations comparing AiDotNet vs Accord.NET
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class VectorOperationsBenchmarks
{
    [Params(100, 1000, 10000)]
    public int Size { get; set; }

    private Vector<double> _aiVectorA = null!;
    private Vector<double> _aiVectorB = null!;
    private double[] _accordVectorA = null!;
    private double[] _accordVectorB = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize AiDotNet vectors
        _aiVectorA = new Vector<double>(Size);
        _aiVectorB = new Vector<double>(Size);

        // Initialize Accord.NET arrays
        _accordVectorA = new double[Size];
        _accordVectorB = new double[Size];

        // Fill with random data
        for (int i = 0; i < Size; i++)
        {
            var value = random.NextDouble();
            _aiVectorA[i] = value;
            _accordVectorA[i] = value;

            value = random.NextDouble();
            _aiVectorB[i] = value;
            _accordVectorB[i] = value;
        }
    }

    #region Dot Product

    [Benchmark]
    public double AiDotNet_DotProduct()
    {
        return _aiVectorA.DotProduct(_aiVectorB);
    }

    [Benchmark(Baseline = true)]
    public double AccordNet_DotProduct()
    {
        return _accordVectorA.Dot(_accordVectorB);
    }

    #endregion

    #region Vector Norms

    [Benchmark]
    public double AiDotNet_EuclideanNorm()
    {
        return _aiVectorA.Norm();
    }

    [Benchmark]
    public double AccordNet_EuclideanNorm()
    {
        return _accordVectorA.Euclidean();
    }

    [Benchmark]
    public double AiDotNet_ManhattanNorm()
    {
        return _aiVectorA.ManhattanNorm();
    }

    [Benchmark]
    public double AccordNet_ManhattanNorm()
    {
        return _accordVectorA.Manhattan();
    }

    #endregion

    #region Vector Addition

    [Benchmark]
    public Vector<double> AiDotNet_VectorAdd()
    {
        return _aiVectorA.Add(_aiVectorB);
    }

    [Benchmark]
    public double[] AccordNet_VectorAdd()
    {
        return _accordVectorA.Add(_accordVectorB);
    }

    #endregion

    #region Scalar Multiplication

    [Benchmark]
    public Vector<double> AiDotNet_ScalarMultiply()
    {
        return _aiVectorA.Multiply(2.5);
    }

    [Benchmark]
    public double[] AccordNet_ScalarMultiply()
    {
        return _accordVectorA.Multiply(2.5);
    }

    #endregion

    #region Element-wise Operations

    [Benchmark]
    public Vector<double> AiDotNet_ElementWiseMultiply()
    {
        return _aiVectorA.MultiplyElementWise(_aiVectorB);
    }

    [Benchmark]
    public double[] AccordNet_ElementWiseMultiply()
    {
        return _accordVectorA.ElementwiseMultiply(_accordVectorB);
    }

    #endregion

    #region Distance Metrics

    [Benchmark]
    public double AiDotNet_EuclideanDistance()
    {
        return _aiVectorA.EuclideanDistance(_aiVectorB);
    }

    [Benchmark]
    public double AccordNet_EuclideanDistance()
    {
        return Accord.Math.Distance.Euclidean(_accordVectorA, _accordVectorB);
    }

    [Benchmark]
    public double AiDotNet_ManhattanDistance()
    {
        return _aiVectorA.ManhattanDistance(_aiVectorB);
    }

    [Benchmark]
    public double AccordNet_ManhattanDistance()
    {
        return Accord.Math.Distance.Manhattan(_accordVectorA, _accordVectorB);
    }

    #endregion
}
