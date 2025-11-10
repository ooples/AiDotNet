using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Accord.Math.Decompositions;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Matrix Decomposition methods comparing AiDotNet vs Accord.NET
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class MatrixDecompositionBenchmarks
{
    [Params(10, 50, 100)]
    public int Size { get; set; }

    private Matrix<double> _aiMatrix = null!;
    private Matrix<double> _aiSymmetricMatrix = null!;
    private Matrix<double> _aiPositiveDefiniteMatrix = null!;
    private double[,] _accordMatrix = null!;
    private double[,] _accordSymmetricMatrix = null!;
    private double[,] _accordPositiveDefiniteMatrix = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize general matrices
        _aiMatrix = new Matrix<double>(Size, Size);
        _accordMatrix = new double[Size, Size];

        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                var value = random.NextDouble() * 10 - 5;
                _aiMatrix[i, j] = value;
                _accordMatrix[i, j] = value;
            }
        }

        // Initialize symmetric matrices
        _aiSymmetricMatrix = new Matrix<double>(Size, Size);
        _accordSymmetricMatrix = new double[Size, Size];

        for (int i = 0; i < Size; i++)
        {
            for (int j = i; j < Size; j++)
            {
                var value = random.NextDouble() * 10;
                _aiSymmetricMatrix[i, j] = value;
                _aiSymmetricMatrix[j, i] = value;
                _accordSymmetricMatrix[i, j] = value;
                _accordSymmetricMatrix[j, i] = value;
            }
        }

        // Initialize positive definite matrices (A^T * A is always positive definite)
        var tempMatrix = new Matrix<double>(Size, Size);
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                tempMatrix[i, j] = random.NextDouble();
            }
        }
        _aiPositiveDefiniteMatrix = tempMatrix.Transpose().Multiply(tempMatrix);

        // Add small diagonal to ensure positive definiteness
        for (int i = 0; i < Size; i++)
        {
            _aiPositiveDefiniteMatrix[i, i] += Size * 0.1;
        }

        _accordPositiveDefiniteMatrix = new double[Size, Size];
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                _accordPositiveDefiniteMatrix[i, j] = _aiPositiveDefiniteMatrix[i, j];
            }
        }
    }

    #region SVD Decomposition

    [Benchmark]
    public (Matrix<double> U, Vector<double> S, Matrix<double> V) AiDotNet_SVD()
    {
        var svd = new SvdDecomposition<double>();
        svd.Decompose(_aiMatrix);
        return (svd.U, svd.S, svd.V);
    }

    [Benchmark(Baseline = true)]
    public (double[,] U, double[] S, double[,] V) AccordNet_SVD()
    {
        var svd = new SingularValueDecomposition(_accordMatrix);
        return (svd.LeftSingularVectors, svd.Diagonal, svd.RightSingularVectors);
    }

    #endregion

    #region QR Decomposition

    [Benchmark]
    public (Matrix<double> Q, Matrix<double> R) AiDotNet_QR()
    {
        var qr = new QrDecomposition<double>();
        qr.Decompose(_aiMatrix);
        return (qr.Q, qr.R);
    }

    [Benchmark]
    public (double[,] Q, double[,] R) AccordNet_QR()
    {
        var qr = new QrDecomposition(_accordMatrix);
        return (qr.OrthogonalFactor, qr.UpperTriangularFactor);
    }

    #endregion

    #region LU Decomposition

    [Benchmark]
    public (Matrix<double> L, Matrix<double> U) AiDotNet_LU()
    {
        var lu = new LuDecomposition<double>();
        lu.Decompose(_aiMatrix);
        return (lu.L, lu.U);
    }

    [Benchmark]
    public (double[,] L, double[,] U) AccordNet_LU()
    {
        var lu = new LuDecomposition(_accordMatrix);
        return (lu.LowerTriangularFactor, lu.UpperTriangularFactor);
    }

    #endregion

    #region Cholesky Decomposition

    [Benchmark]
    public Matrix<double> AiDotNet_Cholesky()
    {
        var cholesky = new CholeskyDecomposition<double>();
        cholesky.Decompose(_aiPositiveDefiniteMatrix);
        return cholesky.L;
    }

    [Benchmark]
    public double[,] AccordNet_Cholesky()
    {
        var cholesky = new CholeskyDecomposition(_accordPositiveDefiniteMatrix);
        return cholesky.LeftTriangularFactor;
    }

    #endregion

    #region Eigen Decomposition

    [Benchmark]
    public (Vector<double> values, Matrix<double> vectors) AiDotNet_Eigen()
    {
        var eigen = new EigenDecomposition<double>();
        eigen.Decompose(_aiSymmetricMatrix);
        return (eigen.RealEigenvalues, eigen.Eigenvectors);
    }

    [Benchmark]
    public (double[] values, double[,] vectors) AccordNet_Eigen()
    {
        var eigen = new EigenvalueDecomposition(_accordSymmetricMatrix);
        return (eigen.RealEigenvalues, eigen.Eigenvectors);
    }

    #endregion
}
