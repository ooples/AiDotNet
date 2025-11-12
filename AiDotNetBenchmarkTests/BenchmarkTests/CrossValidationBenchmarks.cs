using AiDotNet.CrossValidators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Cross-Validation methods
/// Tests performance of different cross-validation strategies
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class CrossValidationBenchmarks
{
    [Params(500, 2000)]
    public int SampleCount { get; set; }

    [Params(5, 10)]
    public int FeatureCount { get; set; }

    [Params(3, 5)]
    public int Folds { get; set; }

    private Matrix<double> _X = null!;
    private Vector<double> _y = null!;
    private Vector<int> _groups = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize training data
        _X = new Matrix<double>(SampleCount, FeatureCount);
        _y = new Vector<double>(SampleCount);
        _groups = new Vector<int>(SampleCount);

        for (int i = 0; i < SampleCount; i++)
        {
            double target = 0;
            for (int j = 0; j < FeatureCount; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                _X[i, j] = value;
                target += value * (j + 1);
            }
            _y[i] = target + random.NextDouble() * 2;
            _groups[i] = i / (SampleCount / 5); // 5 groups
        }
    }

    #region K-Fold Cross Validation

    [Benchmark(Baseline = true)]
    public double KFold_CrossValidate()
    {
        var cv = new KFoldCrossValidator<double>(k: Folds);
        var model = new SimpleRegression<double>();

        return cv.CrossValidate(model, _X, _y);
    }

    #endregion

    #region Stratified K-Fold

    [Benchmark]
    public double StratifiedKFold_CrossValidate()
    {
        var cv = new StratifiedKFoldCrossValidator<double>(k: Folds);
        var model = new SimpleRegression<double>();

        return cv.CrossValidate(model, _X, _y);
    }

    #endregion

    #region Group K-Fold

    [Benchmark]
    public double GroupKFold_CrossValidate()
    {
        var cv = new GroupKFoldCrossValidator<double>(k: Folds);
        var model = new SimpleRegression<double>();

        return cv.CrossValidate(model, _X, _y, _groups);
    }

    #endregion

    #region Leave-One-Out Cross Validation

    [Benchmark]
    public double LeaveOneOut_CrossValidate_Small()
    {
        // Use smaller dataset for LOO
        int smallSize = 100;
        var xSmall = new Matrix<double>(smallSize, FeatureCount);
        var ySmall = new Vector<double>(smallSize);

        for (int i = 0; i < smallSize; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                xSmall[i, j] = _X[i, j];
            }
            ySmall[i] = _y[i];
        }

        var cv = new LeaveOneOutCrossValidator<double>();
        var model = new SimpleRegression<double>();

        return cv.CrossValidate(model, xSmall, ySmall);
    }

    #endregion

    #region Monte Carlo Cross Validation

    [Benchmark]
    public double MonteCarlo_CrossValidate()
    {
        var cv = new MonteCarloValidator<double>(iterations: 10, testSplit: 0.2);
        var model = new SimpleRegression<double>();

        return cv.CrossValidate(model, _X, _y);
    }

    #endregion

    #region Time Series Cross Validation

    [Benchmark]
    public double TimeSeries_CrossValidate()
    {
        var cv = new TimeSeriesCrossValidator<double>(k: Folds);
        var model = new SimpleRegression<double>();

        return cv.CrossValidate(model, _X, _y);
    }

    #endregion

    #region Nested Cross Validation

    [Benchmark]
    public double Nested_CrossValidate()
    {
        var outerCV = new KFoldCrossValidator<double>(k: 3);
        var innerCV = new KFoldCrossValidator<double>(k: 3);
        var nestedCV = new NestedCrossValidator<double>(outerCV, innerCV);
        var model = new SimpleRegression<double>();

        return nestedCV.CrossValidate(model, _X, _y);
    }

    #endregion
}
