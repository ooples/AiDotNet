using AiDotNet.DataProcessor;
using AiDotNet.LinearAlgebra;
using AiDotNet.Normalizers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Data Preprocessing operations
/// Tests performance of data loading, cleaning, and transformation
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class DataPreprocessingBenchmarks
{
    [Params(5000, 20000)]
    public int SampleCount { get; set; }

    [Params(20, 50)]
    public int FeatureCount { get; set; }

    private Matrix<double> _rawData = null!;
    private Matrix<double> _dataWithMissing = null!;
    private Matrix<double> _dataWithOutliers = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize raw data
        _rawData = new Matrix<double>(SampleCount, FeatureCount);
        _dataWithMissing = new Matrix<double>(SampleCount, FeatureCount);
        _dataWithOutliers = new Matrix<double>(SampleCount, FeatureCount);

        for (int i = 0; i < SampleCount; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                double value = random.NextDouble() * 100;
                _rawData[i, j] = value;

                // Add missing values (10% chance)
                if (random.NextDouble() < 0.1)
                {
                    _dataWithMissing[i, j] = double.NaN;
                }
                else
                {
                    _dataWithMissing[i, j] = value;
                }

                // Add outliers (5% chance)
                if (random.NextDouble() < 0.05)
                {
                    _dataWithOutliers[i, j] = value * 10; // Extreme outlier
                }
                else
                {
                    _dataWithOutliers[i, j] = value;
                }
            }
        }
    }

    #region Data Preprocessing Pipeline

    [Benchmark(Baseline = true)]
    public Matrix<double> Preprocessing_FullPipeline()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();

        // 1. Handle missing values
        var cleaned = preprocessor.ImputeMissingValues(_dataWithMissing, strategy: "mean");

        // 2. Remove outliers
        var withoutOutliers = preprocessor.RemoveOutliers(cleaned, method: "iqr");

        // 3. Normalize
        var normalizer = new ZScoreNormalizer<double>();
        var normalized = normalizer.FitTransform(withoutOutliers);

        return normalized;
    }

    #endregion

    #region Missing Value Imputation

    [Benchmark]
    public Matrix<double> Preprocessing_ImputeMean()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.ImputeMissingValues(_dataWithMissing, strategy: "mean");
    }

    [Benchmark]
    public Matrix<double> Preprocessing_ImputeMedian()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.ImputeMissingValues(_dataWithMissing, strategy: "median");
    }

    [Benchmark]
    public Matrix<double> Preprocessing_ImputeMode()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.ImputeMissingValues(_dataWithMissing, strategy: "mode");
    }

    #endregion

    #region Outlier Detection and Removal

    [Benchmark]
    public Matrix<double> Preprocessing_RemoveOutliersIQR()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.RemoveOutliers(_dataWithOutliers, method: "iqr");
    }

    [Benchmark]
    public Matrix<double> Preprocessing_RemoveOutliersZScore()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.RemoveOutliers(_dataWithOutliers, method: "zscore");
    }

    #endregion

    #region Data Splitting

    [Benchmark]
    public (Matrix<double> train, Matrix<double> test) Preprocessing_TrainTestSplit()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.TrainTestSplit(_rawData, testSize: 0.2);
    }

    [Benchmark]
    public (Matrix<double> train, Matrix<double> val, Matrix<double> test) Preprocessing_TrainValTestSplit()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.TrainValidationTestSplit(_rawData, validationSize: 0.15, testSize: 0.15);
    }

    #endregion

    #region Data Shuffling

    [Benchmark]
    public Matrix<double> Preprocessing_Shuffle()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.Shuffle(_rawData);
    }

    #endregion

    #region Data Balancing

    [Benchmark]
    public Matrix<double> Preprocessing_Oversample()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();

        // Create imbalanced data
        int minoritySize = SampleCount / 10;
        var labels = new Vector<int>(SampleCount);
        for (int i = 0; i < SampleCount; i++)
        {
            labels[i] = i < minoritySize ? 1 : 0;
        }

        return preprocessor.Oversample(_rawData, labels);
    }

    [Benchmark]
    public Matrix<double> Preprocessing_Undersample()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();

        // Create imbalanced data
        int minoritySize = SampleCount / 10;
        var labels = new Vector<int>(SampleCount);
        for (int i = 0; i < SampleCount; i++)
        {
            labels[i] = i < minoritySize ? 1 : 0;
        }

        return preprocessor.Undersample(_rawData, labels);
    }

    #endregion

    #region Feature Engineering

    [Benchmark]
    public Matrix<double> Preprocessing_PolynomialFeatures()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();

        // Use subset of features for polynomial expansion
        int subsetSize = Math.Min(5, FeatureCount);
        var subset = new Matrix<double>(SampleCount, subsetSize);
        for (int i = 0; i < SampleCount; i++)
        {
            for (int j = 0; j < subsetSize; j++)
            {
                subset[i, j] = _rawData[i, j];
            }
        }

        return preprocessor.CreatePolynomialFeatures(subset, degree: 2);
    }

    [Benchmark]
    public Matrix<double> Preprocessing_InteractionFeatures()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();

        // Use subset of features for interactions
        int subsetSize = Math.Min(10, FeatureCount);
        var subset = new Matrix<double>(SampleCount, subsetSize);
        for (int i = 0; i < SampleCount; i++)
        {
            for (int j = 0; j < subsetSize; j++)
            {
                subset[i, j] = _rawData[i, j];
            }
        }

        return preprocessor.CreateInteractionFeatures(subset);
    }

    #endregion

    #region Binning

    [Benchmark]
    public Matrix<double> Preprocessing_Binning()
    {
        var preprocessor = new DefaultDataPreprocessor<double>();
        return preprocessor.BinFeatures(_rawData, numBins: 10);
    }

    #endregion
}
