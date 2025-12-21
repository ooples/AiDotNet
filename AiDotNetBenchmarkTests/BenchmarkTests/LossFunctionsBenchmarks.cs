using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for all Loss Functions in AiDotNet
/// Tests both loss calculation and derivative computation
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class LossFunctionsBenchmarks
{
    [Params(100, 1000, 10000)]
    public int Size { get; set; }

    private Vector<double> _predicted = null!;
    private Vector<double> _actual = null!;
    private Vector<double> _binaryPredicted = null!;
    private Vector<double> _binaryActual = null!;
    private Vector<double> _hingeBinaryActual = null!;  // {-1, +1} labels for hinge loss
    private Vector<double> _softmaxPredicted = null!;
    private Vector<double> _oneHotActual = null!;

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
        // Using seeded Random for reproducible benchmark data - not used for security purposes
        var random = RandomHelper.CreateSeededRandom(42); // NOSONAR S2245 - benchmarks don't need cryptographic randomness

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
        _hingeBinaryActual = new Vector<double>(Size);  // {-1, +1} labels for hinge loss

        for (int i = 0; i < Size; i++)
        {
            _binaryPredicted[i] = random.NextDouble();
            int binaryLabel = random.Next(2); // 0 or 1
            _binaryActual[i] = binaryLabel;
            // Convert 0/1 to -1/+1 for hinge loss using integer comparison (not float equality)
            _hingeBinaryActual[i] = binaryLabel switch { 0 => -1.0, _ => 1.0 };
        }

        // Initialize multi-class softmax vectors (proper probability distributions)
        // Using pairs of values that sum to 1.0 (2-class softmax output)
        _softmaxPredicted = new Vector<double>(Size);
        _oneHotActual = new Vector<double>(Size);

        for (int i = 0; i < Size; i += 2)
        {
            // Generate softmax output (values that sum to 1)
            double p = random.NextDouble();
            _softmaxPredicted[i] = p;
            if (i + 1 < Size)
            {
                _softmaxPredicted[i + 1] = 1.0 - p;
            }

            // Generate one-hot encoded actual (0 or 1, exactly one class active per pair)
            int classIndex = random.Next(2);
            _oneHotActual[i] = classIndex == 0 ? 1.0 : 0.0;
            if (i + 1 < Size)
            {
                _oneHotActual[i + 1] = classIndex == 1 ? 1.0 : 0.0;
            }
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

    #region Cross Entropy (Multi-class)

    [Benchmark]
    public double CrossEntropy_CalculateLoss()
    {
        // Use softmax predicted and one-hot encoded actual for proper multi-class classification
        return _crossEntropy.CalculateLoss(_softmaxPredicted, _oneHotActual);
    }

    [Benchmark]
    public Vector<double> CrossEntropy_CalculateDerivative()
    {
        // Use softmax predicted and one-hot encoded actual for proper multi-class classification
        return _crossEntropy.CalculateDerivative(_softmaxPredicted, _oneHotActual);
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
        // Hinge loss expects {-1, +1} labels for max(0, 1 - y*f(x)) formula
        return _hinge.CalculateLoss(_binaryPredicted, _hingeBinaryActual);
    }

    [Benchmark]
    public Vector<double> Hinge_CalculateDerivative()
    {
        // Hinge loss expects {-1, +1} labels for max(0, 1 - y*f(x)) formula
        return _hinge.CalculateDerivative(_binaryPredicted, _hingeBinaryActual);
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
