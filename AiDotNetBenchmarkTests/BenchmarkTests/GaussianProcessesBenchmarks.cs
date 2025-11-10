using AiDotNet.GaussianProcesses;
using AiDotNet.LinearAlgebra;
using AiDotNet.Kernels;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Gaussian Process methods
/// Tests training and prediction performance for GP regression
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class GaussianProcessesBenchmarks
{
    [Params(100, 500, 1000)]
    public int TrainSize { get; set; }

    [Params(5, 20)]
    public int FeatureDim { get; set; }

    private Matrix<double> _trainX = null!;
    private Vector<double> _trainY = null!;
    private Matrix<double> _testX = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize training data
        _trainX = new Matrix<double>(TrainSize, FeatureDim);
        _trainY = new Vector<double>(TrainSize);

        for (int i = 0; i < TrainSize; i++)
        {
            double target = 0;
            for (int j = 0; j < FeatureDim; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                _trainX[i, j] = value;
                target += Math.Sin(value) * (j + 1);
            }
            _trainY[i] = target + random.NextGaussian() * 0.1;
        }

        // Initialize test data
        _testX = new Matrix<double>(50, FeatureDim);
        for (int i = 0; i < 50; i++)
        {
            for (int j = 0; j < FeatureDim; j++)
            {
                _testX[i, j] = random.NextDouble() * 10 - 5;
            }
        }
    }

    #region Standard Gaussian Process

    [Benchmark(Baseline = true)]
    public StandardGaussianProcess<double> GP_Standard_Fit()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var gp = new StandardGaussianProcess<double>(kernel, noise: 0.1);
        gp.Fit(_trainX, _trainY);
        return gp;
    }

    [Benchmark]
    public (Vector<double> mean, Vector<double> variance) GP_Standard_Predict()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var gp = new StandardGaussianProcess<double>(kernel, noise: 0.1);
        gp.Fit(_trainX, _trainY);
        return gp.Predict(_testX);
    }

    [Benchmark]
    public Vector<double> GP_Standard_PredictMean()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var gp = new StandardGaussianProcess<double>(kernel, noise: 0.1);
        gp.Fit(_trainX, _trainY);
        return gp.PredictMean(_testX);
    }

    #endregion

    #region Sparse Gaussian Process

    [Benchmark]
    public SparseGaussianProcess<double> GP_Sparse_Fit()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var gp = new SparseGaussianProcess<double>(kernel, numInducingPoints: Math.Min(100, TrainSize / 2), noise: 0.1);
        gp.Fit(_trainX, _trainY);
        return gp;
    }

    [Benchmark]
    public Vector<double> GP_Sparse_Predict()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var gp = new SparseGaussianProcess<double>(kernel, numInducingPoints: Math.Min(100, TrainSize / 2), noise: 0.1);
        gp.Fit(_trainX, _trainY);
        return gp.PredictMean(_testX);
    }

    #endregion

    #region Multi-Output Gaussian Process

    [Benchmark]
    public MultiOutputGaussianProcess<double> GP_MultiOutput_Fit()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var gp = new MultiOutputGaussianProcess<double>(kernel, numOutputs: 2, noise: 0.1);

        // Create multi-output targets
        var multiY = new Matrix<double>(TrainSize, 2);
        for (int i = 0; i < TrainSize; i++)
        {
            multiY[i, 0] = _trainY[i];
            multiY[i, 1] = _trainY[i] * 0.5;
        }

        gp.Fit(_trainX, multiY);
        return gp;
    }

    #endregion

    #region Different Kernel Functions

    [Benchmark]
    public Vector<double> GP_Matern_Predict()
    {
        var kernel = new MaternKernel<double>(nu: 2.5, lengthScale: 1.0);
        var gp = new StandardGaussianProcess<double>(kernel, noise: 0.1);
        gp.Fit(_trainX, _trainY);
        return gp.PredictMean(_testX);
    }

    [Benchmark]
    public Vector<double> GP_Polynomial_Predict()
    {
        var kernel = new PolynomialKernel<double>(degree: 3, constant: 1.0);
        var gp = new StandardGaussianProcess<double>(kernel, noise: 0.1);
        gp.Fit(_trainX, _trainY);
        return gp.PredictMean(_testX);
    }

    #endregion

    #region Hyperparameter Optimization

    [Benchmark]
    public double GP_LogMarginalLikelihood()
    {
        var kernel = new GaussianKernel<double>(sigma: 1.0);
        var gp = new StandardGaussianProcess<double>(kernel, noise: 0.1);
        gp.Fit(_trainX, _trainY);
        return gp.LogMarginalLikelihood();
    }

    #endregion
}

// Extension for Gaussian random number generation
public static class RandomExtensions
{
    public static double NextGaussian(this Random random, double mean = 0, double stdDev = 1)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}
