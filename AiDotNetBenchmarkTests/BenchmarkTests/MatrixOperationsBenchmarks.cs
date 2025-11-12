using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Accord.Math;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Matrix operations comparing AiDotNet vs Accord.NET
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class MatrixOperationsBenchmarks
{
    [Params(10, 100, 500)]
    public int Size { get; set; }

    private Matrix<double> _aiMatrixA = null!;
    private Matrix<double> _aiMatrixB = null!;
    private double[,] _accordMatrixA = null!;
    private double[,] _accordMatrixB = null!;
    private Vector<double> _aiVector = null!;
    private double[] _accordVector = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize AiDotNet matrices
        _aiMatrixA = new Matrix<double>(Size, Size);
        _aiMatrixB = new Matrix<double>(Size, Size);
        _aiVector = new Vector<double>(Size);

        // Initialize Accord.NET matrices
        _accordMatrixA = new double[Size, Size];
        _accordMatrixB = new double[Size, Size];
        _accordVector = new double[Size];

        // Fill with random data
        for (int i = 0; i < Size; i++)
        {
            _aiVector[i] = random.NextDouble();
            _accordVector[i] = random.NextDouble();

            for (int j = 0; j < Size; j++)
            {
                var value = random.NextDouble();
                _aiMatrixA[i, j] = value;
                _accordMatrixA[i, j] = value;

                value = random.NextDouble();
                _aiMatrixB[i, j] = value;
                _accordMatrixB[i, j] = value;
            }
        }
    }

    #region Matrix Multiplication

    [Benchmark]
    public Matrix<double> AiDotNet_MatrixMultiply()
    {
        return _aiMatrixA.Multiply(_aiMatrixB);
    }

    [Benchmark(Baseline = true)]
    public double[,] AccordNet_MatrixMultiply()
    {
        return _accordMatrixA.Dot(_accordMatrixB);
    }

    #endregion

    #region Matrix-Vector Multiplication

    [Benchmark]
    public Vector<double> AiDotNet_MatrixVectorMultiply()
    {
        return _aiMatrixA.Multiply(_aiVector);
    }

    [Benchmark]
    public double[] AccordNet_MatrixVectorMultiply()
    {
        return _accordMatrixA.Dot(_accordVector);
    }

    #endregion

    #region Matrix Addition

    [Benchmark]
    public Matrix<double> AiDotNet_MatrixAdd()
    {
        return _aiMatrixA.Add(_aiMatrixB);
    }

    [Benchmark]
    public double[,] AccordNet_MatrixAdd()
    {
        return _accordMatrixA.Add(_accordMatrixB);
    }

    #endregion

    #region Matrix Transpose

    [Benchmark]
    public Matrix<double> AiDotNet_Transpose()
    {
        return _aiMatrixA.Transpose();
    }

    [Benchmark]
    public double[,] AccordNet_Transpose()
    {
        return _accordMatrixA.Transpose();
    }

    #endregion

    #region Element-wise Operations

    [Benchmark]
    public Matrix<double> AiDotNet_ElementWiseMultiply()
    {
        return _aiMatrixA.MultiplyElementWise(_aiMatrixB);
    }

    [Benchmark]
    public double[,] AccordNet_ElementWiseMultiply()
    {
        return _accordMatrixA.ElementwiseMultiply(_accordMatrixB);
    }

    #endregion

    #region Matrix Determinant

    [Benchmark]
    public double AiDotNet_Determinant()
    {
        return _aiMatrixA.Determinant();
    }

    [Benchmark]
    public double AccordNet_Determinant()
    {
        return _accordMatrixA.Determinant();
    }

    #endregion

    #region Matrix Inverse

    [Benchmark]
    public Matrix<double> AiDotNet_Inverse()
    {
        return _aiMatrixA.Inverse();
    }

    [Benchmark]
    public double[,] AccordNet_Inverse()
    {
        return _accordMatrixA.Inverse();
    }

    #endregion
}
