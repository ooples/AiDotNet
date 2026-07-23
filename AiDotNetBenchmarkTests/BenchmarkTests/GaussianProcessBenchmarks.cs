using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Gaussian Process and Kernel operations.
/// Tests construction, fit, and prediction performance for various GP and kernel types.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class GaussianProcessBenchmarks
{
    [Params(50, 200)]
    public int NumTrainingPoints { get; set; }

    [Params(1, 5)]
    public int NumDimensions { get; set; }

    private Matrix<double> _trainingX = null!;
    private Vector<double> _trainingY = null!;
    private Vector<double> _testPoint = null!;
    private Matrix<double> _testMatrix = null!;

    private GaussianKernel<double> _rbfKernel = null!;
    private MaternKernel<double> _maternKernel = null!;
    private RationalQuadraticKernel<double> _rqKernel = null!;

    private StandardGaussianProcess<double> _standardGp = null!;
    private SparseGaussianProcess<double> _sparseGp = null!;
    private Vector<double>[] _testPoints = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Using seeded Random for reproducible benchmark data
        var random = RandomHelper.CreateSeededRandom(42);

        // Initialize training data
        _trainingX = new Matrix<double>(NumTrainingPoints, NumDimensions);
        _trainingY = new Vector<double>(NumTrainingPoints);

        for (int i = 0; i < NumTrainingPoints; i++)
        {
            double sum = 0;
            for (int j = 0; j < NumDimensions; j++)
            {
                double val = random.NextDouble() * 10;
                _trainingX[i, j] = val;
                sum += val;
            }
            // y = sin(sum(x)) + noise
            _trainingY[i] = Math.Sin(sum) + random.NextDouble() * 0.1;
        }

        // Initialize test point
        _testPoint = new Vector<double>(NumDimensions);
        for (int j = 0; j < NumDimensions; j++)
        {
            _testPoint[j] = random.NextDouble() * 10;
        }

        // Test matrix for batch prediction (as array of vectors for iteration)
        _testMatrix = new Matrix<double>(10, NumDimensions);
        _testPoints = new Vector<double>[10];
        for (int i = 0; i < 10; i++)
        {
            _testPoints[i] = new Vector<double>(NumDimensions);
            for (int j = 0; j < NumDimensions; j++)
            {
                double val = random.NextDouble() * 10;
                _testMatrix[i, j] = val;
                _testPoints[i][j] = val;
            }
        }

        // Initialize kernels
        _rbfKernel = new GaussianKernel<double>(1.0);
        _maternKernel = new MaternKernel<double>(2.5, 1.0);
        _rqKernel = new RationalQuadraticKernel<double>(1.0, 1.0);

        // Initialize and train GPs
        _standardGp = new StandardGaussianProcess<double>(_rbfKernel);
        _standardGp.Fit(_trainingX, _trainingY);

        // SparseGaussianProcess auto-selects inducing points (min of data points or 100)
        _sparseGp = new SparseGaussianProcess<double>(_rbfKernel);
        _sparseGp.Fit(_trainingX, _trainingY);
    }

    #region Kernel Benchmarks

    [Benchmark]
    public double RBFKernel_Calculate()
    {
        return _rbfKernel.Calculate(_testPoint, _testPoint);
    }

    [Benchmark]
    public double MaternKernel_Calculate()
    {
        return _maternKernel.Calculate(_testPoint, _testPoint);
    }

    [Benchmark]
    public double RationalQuadraticKernel_Calculate()
    {
        return _rqKernel.Calculate(_testPoint, _testPoint);
    }

    #endregion

    #region Standard GP Benchmarks

    [Benchmark(Baseline = true)]
    public void StandardGP_Fit()
    {
        var gp = new StandardGaussianProcess<double>(_rbfKernel);
        gp.Fit(_trainingX, _trainingY);
    }

    [Benchmark]
    public (double, double) StandardGP_Predict()
    {
        return _standardGp.Predict(_testPoint);
    }

    [Benchmark]
    public (double mean, double variance)[] StandardGP_PredictBatch()
    {
        var results = new (double mean, double variance)[_testPoints.Length];
        for (int i = 0; i < _testPoints.Length; i++)
        {
            results[i] = _standardGp.Predict(_testPoints[i]);
        }
        return results;
    }

    #endregion

    #region Sparse GP Benchmarks

    [Benchmark]
    public void SparseGP_Fit()
    {
        // SparseGaussianProcess auto-selects inducing points (min of data points or 100)
        var gp = new SparseGaussianProcess<double>(_rbfKernel);
        gp.Fit(_trainingX, _trainingY);
    }

    [Benchmark]
    public (double, double) SparseGP_Predict()
    {
        return _sparseGp.Predict(_testPoint);
    }

    #endregion
}

/// <summary>
/// Benchmarks for specialized kernel operations including grid kernels and RFF.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class SpecializedKernelBenchmarks
{
    [Params(10, 20)]
    public int GridSize { get; set; }

    [Params(100, 500)]
    public int NumRFFFeatures { get; set; }

    private double[][] _gridCoordinates = null!;
    private GridKernel<double> _gridKernel = null!;
    private RFFKernel<double> _rffKernel = null!;
    private CosineKernel<double> _cosineKernel = null!;
    private SpectralDeltaKernel<double> _spectralKernel = null!;

    private Vector<double> _testPoint2D = null!;
    private Vector<double> _fullGridVector = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Initialize 2D grid coordinates
        var gridX = Enumerable.Range(0, GridSize).Select(i => (double)i).ToArray();
        var gridY = Enumerable.Range(0, GridSize).Select(i => (double)i).ToArray();
        _gridCoordinates = new[] { gridX, gridY };

        // Initialize kernels
        _gridKernel = GridKernel<double>.WithRBF(_gridCoordinates, lengthscales: new[] { 1.0, 1.0 });
        _gridKernel.Precompute();

        _rffKernel = new RFFKernel<double>(numFeatures: NumRFFFeatures, inputDim: 2);

        _cosineKernel = new CosineKernel<double>();

        _spectralKernel = SpectralDeltaKernel<double>.FromPeriod(period: 7.0);

        // Test point
        _testPoint2D = new Vector<double>(new double[] { GridSize / 2.0, GridSize / 2.0 });

        // Full grid vector for Kronecker multiply
        int totalPoints = GridSize * GridSize;
        _fullGridVector = new Vector<double>(totalPoints);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < totalPoints; i++)
        {
            _fullGridVector[i] = random.NextDouble();
        }
    }

    #region Grid Kernel Benchmarks

    [Benchmark(Baseline = true)]
    public double GridKernel_Calculate()
    {
        return _gridKernel.Calculate(_testPoint2D, _testPoint2D);
    }

    [Benchmark]
    public Vector<double> GridKernel_KroneckerMultiply()
    {
        return _gridKernel.KroneckerMultiply(_fullGridVector);
    }

    [Benchmark]
    public double GridKernel_LogDeterminant()
    {
        return _gridKernel.LogDeterminant();
    }

    #endregion

    #region RFF Kernel Benchmarks

    [Benchmark]
    public double RFFKernel_Calculate()
    {
        return _rffKernel.Calculate(_testPoint2D, _testPoint2D);
    }

    [Benchmark]
    public double[] RFFKernel_GetFeatures()
    {
        return _rffKernel.GetFeatures(_testPoint2D);
    }

    #endregion

    #region Other Kernel Benchmarks

    [Benchmark]
    public double CosineKernel_Calculate()
    {
        return _cosineKernel.Calculate(_testPoint2D, _testPoint2D);
    }

    [Benchmark]
    public double SpectralDeltaKernel_Calculate()
    {
        return _spectralKernel.Calculate(_testPoint2D, _testPoint2D);
    }

    [Benchmark]
    public double SpectralDeltaKernel_GetPSD()
    {
        return _spectralKernel.GetPowerSpectralDensity(1.0);
    }

    #endregion
}

/// <summary>
/// Benchmarks for MCMC-based GP inference.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80)]
public class MCMCGPBenchmarks
{
    [Params(20, 50)]
    public int NumDataPoints { get; set; }

    [Params(50, 100)]
    public int NumMCMCSamples { get; set; }

    private Matrix<double> _trainingX = null!;
    private Vector<double> _trainingY = null!;
    private Vector<double> _testPoint = null!;
    private GaussianKernel<double> _kernel = null!;
    private GPWithMCMC<double> _trainedMcmcGp = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Initialize training data
        _trainingX = new Matrix<double>(NumDataPoints, 1);
        _trainingY = new Vector<double>(NumDataPoints);

        for (int i = 0; i < NumDataPoints; i++)
        {
            double x = i * 10.0 / NumDataPoints;
            _trainingX[i, 0] = x;
            _trainingY[i] = Math.Sin(x) + random.NextDouble() * 0.1;
        }

        _testPoint = new Vector<double>(new double[] { 5.0 });
        _kernel = new GaussianKernel<double>(1.0);

        // Pre-train one GP for prediction benchmarks
        _trainedMcmcGp = new GPWithMCMC<double>(_kernel, numSamples: NumMCMCSamples, burnIn: 20, seed: 42);
        _trainedMcmcGp.Fit(_trainingX, _trainingY);
    }

    [Benchmark(Baseline = true)]
    public void MCMCGP_Fit()
    {
        var gp = new GPWithMCMC<double>(_kernel, numSamples: NumMCMCSamples, burnIn: 20, seed: 42);
        gp.Fit(_trainingX, _trainingY);
    }

    [Benchmark]
    public (double, double) MCMCGP_Predict()
    {
        return _trainedMcmcGp.Predict(_testPoint);
    }

    [Benchmark]
    public Dictionary<string, (double, double)> MCMCGP_GetPosteriorStats()
    {
        return _trainedMcmcGp.GetPosteriorStatistics();
    }
}

/// <summary>
/// Benchmarks for Beta likelihood computations.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class BetaLikelihoodBenchmarks
{
    [Params(100, 1000)]
    public int NumPoints { get; set; }

    private Vector<double> _y = null!;
    private Vector<double> _f = null!;
    private BetaLikelihood<double> _likelihood = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _y = new Vector<double>(NumPoints);
        _f = new Vector<double>(NumPoints);

        for (int i = 0; i < NumPoints; i++)
        {
            _y[i] = 0.01 + random.NextDouble() * 0.98; // Values in (0, 1)
            _f[i] = random.NextDouble() * 4 - 2; // Latent values
        }

        _likelihood = new BetaLikelihood<double>(precision: 10.0);
    }

    [Benchmark(Baseline = true)]
    public Vector<double> BetaLikelihood_GetMeans()
    {
        return _likelihood.GetMeans(_f);
    }

    [Benchmark]
    public double BetaLikelihood_LogLikelihood()
    {
        return _likelihood.LogLikelihood(_y, _f);
    }

    [Benchmark]
    public Vector<double> BetaLikelihood_Gradient()
    {
        return _likelihood.LogLikelihoodGradient(_y, _f);
    }

    [Benchmark]
    public Vector<double> BetaLikelihood_Hessian()
    {
        return _likelihood.LogLikelihoodHessianDiag(_y, _f);
    }

    [Benchmark]
    public (double, double) BetaLikelihood_PredictiveMoments()
    {
        return _likelihood.PredictiveMoments(0.0, 1.0);
    }
}
