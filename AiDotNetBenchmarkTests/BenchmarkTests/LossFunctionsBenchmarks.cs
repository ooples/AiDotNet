using AiDotNet.LossFunctions;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for all Loss Functions in AiDotNet
/// Tests both loss calculation and derivative computation
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class LossFunctionsBenchmarks
{
    [Params(100, 1000, 10000)]
    public int Size { get; set; }

    private Vector<double> _predicted = null!;
    private Vector<double> _actual = null!;
    private Vector<double> _binaryPredicted = null!;
    private Vector<double> _binaryActual = null!;

    // Regression loss functions
    private MeanSquaredErrorLoss<double> _mse = null!;
    private MeanAbsoluteErrorLoss<double> _mae = null!;
    private RootMeanSquaredErrorLoss<double> _rmse = null!;
    private HuberLoss<double> _huber = null!;
    private QuantileLoss<double> _quantile = null!;
    private LogCoshLoss<double> _logCosh = null!;

    // Classification loss functions
    private BinaryCrossEntropyLoss<double> _bce = null!;
    private CrossEntropyLoss<double> _crossEntropy = null!;
    private FocalLoss<double> _focal = null!;
    private HingeLoss<double> _hinge = null!;

    // Other loss functions
    private CosineSimilarityLoss<double> _cosine = null!;
    private DiceLoss<double> _dice = null!;
    private JaccardLoss<double> _jaccard = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize regression vectors
        _predicted = new Vector<double>(Size);
        _actual = new Vector<double>(Size);

        for (int i = 0; i < Size; i++)
        {
            _predicted[i] = random.NextDouble() * 10;
            _actual[i] = random.NextDouble() * 10;
        }

        // Initialize binary classification vectors (values between 0 and 1)
        _binaryPredicted = new Vector<double>(Size);
        _binaryActual = new Vector<double>(Size);

        for (int i = 0; i < Size; i++)
        {
            _binaryPredicted[i] = random.NextDouble();
            _binaryActual[i] = random.Next(2); // 0 or 1
        }

        // Initialize loss functions
        _mse = new MeanSquaredErrorLoss<double>();
        _mae = new MeanAbsoluteErrorLoss<double>();
        _rmse = new RootMeanSquaredErrorLoss<double>();
        _huber = new HuberLoss<double>();
        _quantile = new QuantileLoss<double>();
        _logCosh = new LogCoshLoss<double>();

        _bce = new BinaryCrossEntropyLoss<double>();
        _crossEntropy = new CrossEntropyLoss<double>();
        _focal = new FocalLoss<double>();
        _hinge = new HingeLoss<double>();

        _cosine = new CosineSimilarityLoss<double>();
        _dice = new DiceLoss<double>();
        _jaccard = new JaccardLoss<double>();
    }

    #region Mean Squared Error

    [Benchmark(Baseline = true)]
    public double MSE_CalculateLoss()
    {
        return _mse.CalculateLoss(_predicted, _actual);
    }

    [Benchmark]
    public Vector<double> MSE_CalculateDerivative()
    {
        return _mse.CalculateDerivative(_predicted, _actual);
    }

    #endregion

    #region Mean Absolute Error

    [Benchmark]
    public double MAE_CalculateLoss()
    {
        return _mae.CalculateLoss(_predicted, _actual);
    }

    [Benchmark]
    public Vector<double> MAE_CalculateDerivative()
    {
        return _mae.CalculateDerivative(_predicted, _actual);
    }

    #endregion

    #region Root Mean Squared Error

    [Benchmark]
    public double RMSE_CalculateLoss()
    {
        return _rmse.CalculateLoss(_predicted, _actual);
    }

    [Benchmark]
    public Vector<double> RMSE_CalculateDerivative()
    {
        return _rmse.CalculateDerivative(_predicted, _actual);
    }

    #endregion

    #region Huber Loss

    [Benchmark]
    public double Huber_CalculateLoss()
    {
        return _huber.CalculateLoss(_predicted, _actual);
    }

    [Benchmark]
    public Vector<double> Huber_CalculateDerivative()
    {
        return _huber.CalculateDerivative(_predicted, _actual);
    }

    #endregion

    #region Quantile Loss

    [Benchmark]
    public double Quantile_CalculateLoss()
    {
        return _quantile.CalculateLoss(_predicted, _actual);
    }

    [Benchmark]
    public Vector<double> Quantile_CalculateDerivative()
    {
        return _quantile.CalculateDerivative(_predicted, _actual);
    }

    #endregion

    #region LogCosh Loss

    [Benchmark]
    public double LogCosh_CalculateLoss()
    {
        return _logCosh.CalculateLoss(_predicted, _actual);
    }

    [Benchmark]
    public Vector<double> LogCosh_CalculateDerivative()
    {
        return _logCosh.CalculateDerivative(_predicted, _actual);
    }

    #endregion

    #region Binary Cross Entropy

    [Benchmark]
    public double BCE_CalculateLoss()
    {
        return _bce.CalculateLoss(_binaryPredicted, _binaryActual);
    }

    [Benchmark]
    public Vector<double> BCE_CalculateDerivative()
    {
        return _bce.CalculateDerivative(_binaryPredicted, _binaryActual);
    }

    #endregion

    #region Cross Entropy

    [Benchmark]
    public double CrossEntropy_CalculateLoss()
    {
        return _crossEntropy.CalculateLoss(_binaryPredicted, _binaryActual);
    }

    [Benchmark]
    public Vector<double> CrossEntropy_CalculateDerivative()
    {
        return _crossEntropy.CalculateDerivative(_binaryPredicted, _binaryActual);
    }

    #endregion

    #region Focal Loss

    [Benchmark]
    public double Focal_CalculateLoss()
    {
        return _focal.CalculateLoss(_binaryPredicted, _binaryActual);
    }

    [Benchmark]
    public Vector<double> Focal_CalculateDerivative()
    {
        return _focal.CalculateDerivative(_binaryPredicted, _binaryActual);
    }

    #endregion

    #region Hinge Loss

    [Benchmark]
    public double Hinge_CalculateLoss()
    {
        return _hinge.CalculateLoss(_binaryPredicted, _binaryActual);
    }

    [Benchmark]
    public Vector<double> Hinge_CalculateDerivative()
    {
        return _hinge.CalculateDerivative(_binaryPredicted, _binaryActual);
    }

    #endregion

    #region Cosine Similarity Loss

    [Benchmark]
    public double Cosine_CalculateLoss()
    {
        return _cosine.CalculateLoss(_predicted, _actual);
    }

    [Benchmark]
    public Vector<double> Cosine_CalculateDerivative()
    {
        return _cosine.CalculateDerivative(_predicted, _actual);
    }

    #endregion

    #region Dice Loss

    [Benchmark]
    public double Dice_CalculateLoss()
    {
        return _dice.CalculateLoss(_binaryPredicted, _binaryActual);
    }

    [Benchmark]
    public Vector<double> Dice_CalculateDerivative()
    {
        return _dice.CalculateDerivative(_binaryPredicted, _binaryActual);
    }

    #endregion

    #region Jaccard Loss

    [Benchmark]
    public double Jaccard_CalculateLoss()
    {
        return _jaccard.CalculateLoss(_binaryPredicted, _binaryActual);
    }

    [Benchmark]
    public Vector<double> Jaccard_CalculateDerivative()
    {
        return _jaccard.CalculateDerivative(_binaryPredicted, _binaryActual);
    }

    #endregion
}
