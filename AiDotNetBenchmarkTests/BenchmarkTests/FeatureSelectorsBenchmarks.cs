using AiDotNet.FeatureSelectors;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Feature Selection methods
/// Tests performance of various feature selection algorithms
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class FeatureSelectorsBenchmarks
{
    [Params(1000, 5000)]
    public int SampleCount { get; set; }

    [Params(50, 100)]
    public int FeatureCount { get; set; }

    [Params(10, 20)]
    public int SelectedFeatures { get; set; }

    private Matrix<double> _X = null!;
    private Vector<double> _y = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize data with some informative and some random features
        _X = new Matrix<double>(SampleCount, FeatureCount);
        _y = new Vector<double>(SampleCount);

        for (int i = 0; i < SampleCount; i++)
        {
            double target = 0;

            // First 20% of features are highly informative
            int informativeCount = FeatureCount / 5;
            for (int j = 0; j < informativeCount; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                _X[i, j] = value;
                target += value * (j + 1);
            }

            // Remaining features are mostly noise
            for (int j = informativeCount; j < FeatureCount; j++)
            {
                _X[i, j] = random.NextDouble() * 2 - 1;
            }

            _y[i] = target + random.NextDouble() * 2;
        }
    }

    #region Variance Threshold

    [Benchmark(Baseline = true)]
    public Matrix<double> FeatureSelector_VarianceThreshold()
    {
        var selector = new VarianceThresholdSelector<double>(threshold: 0.1);
        selector.Fit(_X);
        return selector.Transform(_X);
    }

    #endregion

    #region Univariate Feature Selection

    [Benchmark]
    public Matrix<double> FeatureSelector_SelectKBest()
    {
        var selector = new SelectKBestSelector<double>(k: SelectedFeatures);
        selector.Fit(_X, _y);
        return selector.Transform(_X);
    }

    [Benchmark]
    public Matrix<double> FeatureSelector_SelectPercentile()
    {
        var selector = new SelectPercentileSelector<double>(percentile: 50);
        selector.Fit(_X, _y);
        return selector.Transform(_X);
    }

    #endregion

    #region Recursive Feature Elimination

    [Benchmark]
    public Matrix<double> FeatureSelector_RFE()
    {
        var selector = new RecursiveFeatureEliminationSelector<double>(
            numFeaturesToSelect: SelectedFeatures
        );
        selector.Fit(_X, _y);
        return selector.Transform(_X);
    }

    #endregion

    #region Mutual Information

    [Benchmark]
    public Matrix<double> FeatureSelector_MutualInformation()
    {
        var selector = new MutualInformationSelector<double>(k: SelectedFeatures);
        selector.Fit(_X, _y);
        return selector.Transform(_X);
    }

    #endregion

    #region L1-Based Feature Selection

    [Benchmark]
    public Matrix<double> FeatureSelector_L1Based()
    {
        var selector = new L1BasedFeatureSelector<double>(threshold: 0.1);
        selector.Fit(_X, _y);
        return selector.Transform(_X);
    }

    #endregion

    #region Tree-Based Feature Importance

    [Benchmark]
    public Matrix<double> FeatureSelector_TreeBased()
    {
        var selector = new TreeBasedFeatureSelector<double>(
            numFeaturesToSelect: SelectedFeatures
        );
        selector.Fit(_X, _y);
        return selector.Transform(_X);
    }

    #endregion

    #region Sequential Feature Selection

    [Benchmark]
    public Matrix<double> FeatureSelector_SequentialForward()
    {
        var selector = new SequentialForwardSelector<double>(
            numFeaturesToSelect: Math.Min(10, SelectedFeatures)
        );
        selector.Fit(_X, _y);
        return selector.Transform(_X);
    }

    [Benchmark]
    public Matrix<double> FeatureSelector_SequentialBackward()
    {
        var selector = new SequentialBackwardSelector<double>(
            numFeaturesToSelect: Math.Min(10, SelectedFeatures)
        );
        selector.Fit(_X, _y);
        return selector.Transform(_X);
    }

    #endregion

    #region Get Feature Importance Scores

    [Benchmark]
    public Vector<double> FeatureSelector_GetImportanceScores()
    {
        var selector = new SelectKBestSelector<double>(k: SelectedFeatures);
        selector.Fit(_X, _y);
        return selector.GetFeatureScores();
    }

    #endregion
}
