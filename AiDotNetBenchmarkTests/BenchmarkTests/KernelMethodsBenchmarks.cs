using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Accord.Statistics.Kernels;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Kernel methods comparing AiDotNet vs Accord.NET
/// Tests kernel computation performance for various kernel types
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class KernelMethodsBenchmarks
{
    [Params(10, 100, 1000)]
    public int VectorSize { get; set; }

    [Params(50, 200)]
    public int DatasetSize { get; set; }

    private Vector<double> _aiVectorA = null!;
    private Vector<double> _aiVectorB = null!;
    private Matrix<double> _aiDataset = null!;

    private double[] _accordVectorA = null!;
    private double[] _accordVectorB = null!;
    private double[][] _accordDataset = null!;

    // AiDotNet kernels
    private GaussianKernel<double> _aiGaussian = null!;
    private LinearKernel<double> _aiLinear = null!;
    private PolynomialKernel<double> _aiPolynomial = null!;
    private SigmoidKernel<double> _aiSigmoid = null!;
    private LaplacianKernel<double> _aiLaplacian = null!;

    // Accord.NET kernels
    private Gaussian _accordGaussian = null!;
    private Linear _accordLinear = null!;
    private Polynomial _accordPolynomial = null!;
    private Sigmoid _accordSigmoid = null!;
    private Laplacian _accordLaplacian = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize single vectors
        _aiVectorA = new Vector<double>(VectorSize);
        _aiVectorB = new Vector<double>(VectorSize);
        _accordVectorA = new double[VectorSize];
        _accordVectorB = new double[VectorSize];

        for (int i = 0; i < VectorSize; i++)
        {
            var valueA = random.NextDouble() * 2 - 1;
            var valueB = random.NextDouble() * 2 - 1;

            _aiVectorA[i] = valueA;
            _aiVectorB[i] = valueB;
            _accordVectorA[i] = valueA;
            _accordVectorB[i] = valueB;
        }

        // Initialize datasets for kernel matrix computation
        _aiDataset = new Matrix<double>(DatasetSize, VectorSize);
        _accordDataset = new double[DatasetSize][];

        for (int i = 0; i < DatasetSize; i++)
        {
            _accordDataset[i] = new double[VectorSize];
            for (int j = 0; j < VectorSize; j++)
            {
                var value = random.NextDouble() * 2 - 1;
                _aiDataset[i, j] = value;
                _accordDataset[i][j] = value;
            }
        }

        // Initialize AiDotNet kernels
        _aiGaussian = new GaussianKernel<double>(sigma: 1.0);
        _aiLinear = new LinearKernel<double>();
        _aiPolynomial = new PolynomialKernel<double>(degree: 3, constant: 1.0);
        _aiSigmoid = new SigmoidKernel<double>(alpha: 1.0, constant: 0.0);
        _aiLaplacian = new LaplacianKernel<double>(sigma: 1.0);

        // Initialize Accord.NET kernels
        _accordGaussian = new Gaussian(sigma: 1.0);
        _accordLinear = new Linear();
        _accordPolynomial = new Polynomial(degree: 3, constant: 1.0);
        _accordSigmoid = new Sigmoid(alpha: 1.0, constant: 0.0);
        _accordLaplacian = new Laplacian(sigma: 1.0);
    }

    #region Gaussian (RBF) Kernel

    [Benchmark(Baseline = true)]
    public double AiDotNet_Gaussian_Compute()
    {
        return _aiGaussian.Compute(_aiVectorA, _aiVectorB);
    }

    [Benchmark]
    public double AccordNet_Gaussian_Compute()
    {
        return _accordGaussian.Function(_accordVectorA, _accordVectorB);
    }

    [Benchmark]
    public Matrix<double> AiDotNet_Gaussian_KernelMatrix()
    {
        var kernel = new Matrix<double>(DatasetSize, DatasetSize);
        for (int i = 0; i < DatasetSize; i++)
        {
            var rowI = _aiDataset.GetRow(i);
            for (int j = 0; j < DatasetSize; j++)
            {
                var rowJ = _aiDataset.GetRow(j);
                kernel[i, j] = _aiGaussian.Compute(rowI, rowJ);
            }
        }
        return kernel;
    }

    [Benchmark]
    public double[,] AccordNet_Gaussian_KernelMatrix()
    {
        return _accordGaussian.ToJagged(_accordDataset);
    }

    #endregion

    #region Linear Kernel

    [Benchmark]
    public double AiDotNet_Linear_Compute()
    {
        return _aiLinear.Compute(_aiVectorA, _aiVectorB);
    }

    [Benchmark]
    public double AccordNet_Linear_Compute()
    {
        return _accordLinear.Function(_accordVectorA, _accordVectorB);
    }

    #endregion

    #region Polynomial Kernel

    [Benchmark]
    public double AiDotNet_Polynomial_Compute()
    {
        return _aiPolynomial.Compute(_aiVectorA, _aiVectorB);
    }

    [Benchmark]
    public double AccordNet_Polynomial_Compute()
    {
        return _accordPolynomial.Function(_accordVectorA, _accordVectorB);
    }

    #endregion

    #region Sigmoid Kernel

    [Benchmark]
    public double AiDotNet_Sigmoid_Compute()
    {
        return _aiSigmoid.Compute(_aiVectorA, _aiVectorB);
    }

    [Benchmark]
    public double AccordNet_Sigmoid_Compute()
    {
        return _accordSigmoid.Function(_accordVectorA, _accordVectorB);
    }

    #endregion

    #region Laplacian Kernel

    [Benchmark]
    public double AiDotNet_Laplacian_Compute()
    {
        return _aiLaplacian.Compute(_aiVectorA, _aiVectorB);
    }

    [Benchmark]
    public double AccordNet_Laplacian_Compute()
    {
        return _accordLaplacian.Function(_accordVectorA, _accordVectorB);
    }

    #endregion

    #region Batch Kernel Computations

    [Benchmark]
    public double AiDotNet_Gaussian_BatchCompute()
    {
        double sum = 0;
        for (int i = 0; i < DatasetSize; i++)
        {
            var rowI = _aiDataset.GetRow(i);
            sum += _aiGaussian.Compute(rowI, _aiVectorA);
        }
        return sum;
    }

    [Benchmark]
    public double AccordNet_Gaussian_BatchCompute()
    {
        double sum = 0;
        for (int i = 0; i < DatasetSize; i++)
        {
            sum += _accordGaussian.Function(_accordDataset[i], _accordVectorA);
        }
        return sum;
    }

    #endregion
}
