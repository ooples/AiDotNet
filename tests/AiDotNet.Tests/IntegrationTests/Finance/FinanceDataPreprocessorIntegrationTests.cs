using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Data;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// Integration tests for FinancialPreprocessor, MarketDataProvider, and MarketDataPoint.
/// Tests golden reference values, normalization correctness, windowing, and edge cases.
/// </summary>
public class FinanceDataPreprocessorIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region MarketDataPoint Tests

    [Fact]
    public void MarketDataPoint_ConstructsWithCorrectValues()
    {
        var ts = new DateTime(2024, 1, 15, 9, 30, 0);
        var point = new MarketDataPoint<double>(ts, 100.0, 105.0, 98.0, 103.0, 50000.0);

        Assert.Equal(ts, point.Timestamp);
        Assert.Equal(100.0, point.Open);
        Assert.Equal(105.0, point.High);
        Assert.Equal(98.0, point.Low);
        Assert.Equal(103.0, point.Close);
        Assert.Equal(50000.0, point.Volume);
    }

    [Fact]
    public void MarketDataPoint_Float_ConstructsCorrectly()
    {
        var ts = new DateTime(2024, 6, 1);
        var point = new MarketDataPoint<float>(ts, 10.5f, 11.0f, 10.0f, 10.8f, 1000.0f);

        Assert.Equal(10.5f, point.Open);
        Assert.Equal(11.0f, point.High);
        Assert.Equal(10.0f, point.Low);
        Assert.Equal(10.8f, point.Close);
        Assert.Equal(1000.0f, point.Volume);
    }

    #endregion

    #region MarketDataProvider Tests

    [Fact]
    public void MarketDataProvider_Add_IncreasesCount()
    {
        var provider = new MarketDataProvider<double>();

        Assert.Equal(0, provider.Count);

        provider.Add(CreatePoint(0, 100));
        Assert.Equal(1, provider.Count);

        provider.Add(CreatePoint(1, 101));
        Assert.Equal(2, provider.Count);
    }

    [Fact]
    public void MarketDataProvider_AddRange_AddsAllPoints()
    {
        var provider = new MarketDataProvider<double>();
        var points = CreateSeries(10, 100.0, 1.0);

        provider.AddRange(points);

        Assert.Equal(10, provider.Count);
    }

    [Fact]
    public void MarketDataProvider_Clear_ResetsCount()
    {
        var provider = new MarketDataProvider<double>();
        provider.AddRange(CreateSeries(5, 100.0, 1.0));

        Assert.Equal(5, provider.Count);

        provider.Clear();
        Assert.Equal(0, provider.Count);
    }

    [Fact]
    public void MarketDataProvider_GetAll_ReturnsAllPoints()
    {
        var provider = new MarketDataProvider<double>();
        var points = CreateSeries(3, 100.0, 1.0);
        provider.AddRange(points);

        var all = provider.GetAll();

        Assert.Equal(3, all.Count);
        Assert.Equal(100.0, all[0].Close);
        Assert.Equal(101.0, all[1].Close);
        Assert.Equal(102.0, all[2].Close);
    }

    [Fact]
    public void MarketDataProvider_GetRange_FiltersCorrectly()
    {
        var provider = new MarketDataProvider<double>();
        var baseDate = new DateTime(2024, 1, 1);
        for (int i = 0; i < 10; i++)
        {
            provider.Add(new MarketDataPoint<double>(
                baseDate.AddDays(i), 100.0 + i, 105.0 + i, 95.0 + i, 100.0 + i, 1000.0));
        }

        var range = provider.GetRange(baseDate.AddDays(3), baseDate.AddDays(6));

        Assert.Equal(4, range.Count); // days 3, 4, 5, 6
        Assert.Equal(103.0, range[0].Close);
        Assert.Equal(106.0, range[3].Close);
    }

    [Fact]
    public void MarketDataProvider_GetRange_EmptyWhenNoMatch()
    {
        var provider = new MarketDataProvider<double>();
        var baseDate = new DateTime(2024, 1, 1);
        provider.AddRange(CreateSeries(5, 100.0, 1.0));

        var range = provider.GetRange(new DateTime(2025, 1, 1), new DateTime(2025, 12, 31));

        Assert.Equal(0, range.Count);
    }

    [Fact]
    public void MarketDataProvider_GetWindow_ReturnsCorrectSlice()
    {
        var provider = new MarketDataProvider<double>();
        provider.AddRange(CreateSeries(10, 100.0, 1.0));

        var window = provider.GetWindow(2, 3);

        Assert.Equal(3, window.Count);
        Assert.Equal(102.0, window[0].Close);
        Assert.Equal(103.0, window[1].Close);
        Assert.Equal(104.0, window[2].Close);
    }

    [Fact]
    public void MarketDataProvider_GetWindow_ClampedAtEnd()
    {
        var provider = new MarketDataProvider<double>();
        provider.AddRange(CreateSeries(5, 100.0, 1.0));

        // Request beyond end
        var window = provider.GetWindow(3, 10);

        Assert.Equal(2, window.Count); // only indices 3 and 4
    }

    [Fact]
    public void MarketDataProvider_GetWindow_InvalidIndex_Throws()
    {
        var provider = new MarketDataProvider<double>();
        provider.AddRange(CreateSeries(5, 100.0, 1.0));

        Assert.Throws<ArgumentOutOfRangeException>(() => provider.GetWindow(-1, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => provider.GetWindow(5, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => provider.GetWindow(0, 0));
    }

    [Fact]
    public void MarketDataProvider_ToTensor_WithVolume_GoldenReference()
    {
        var provider = new MarketDataProvider<double>();
        provider.Add(new MarketDataPoint<double>(DateTime.Now, 10.0, 12.0, 9.0, 11.0, 500.0));
        provider.Add(new MarketDataPoint<double>(DateTime.Now.AddMinutes(1), 11.0, 13.0, 10.0, 12.0, 600.0));

        var tensor = provider.ToTensor(includeVolume: true);

        Assert.Equal(2, tensor.Shape[0]); // 2 points
        Assert.Equal(5, tensor.Shape[1]); // OHLCV

        // Row 0: O=10, H=12, L=9, C=11, V=500
        Assert.Equal(10.0, tensor[0, 0], Tolerance);
        Assert.Equal(12.0, tensor[0, 1], Tolerance);
        Assert.Equal(9.0, tensor[0, 2], Tolerance);
        Assert.Equal(11.0, tensor[0, 3], Tolerance);
        Assert.Equal(500.0, tensor[0, 4], Tolerance);

        // Row 1: O=11, H=13, L=10, C=12, V=600
        Assert.Equal(11.0, tensor[1, 0], Tolerance);
        Assert.Equal(600.0, tensor[1, 4], Tolerance);
    }

    [Fact]
    public void MarketDataProvider_ToTensor_WithoutVolume_HasFourColumns()
    {
        var provider = new MarketDataProvider<double>();
        provider.Add(new MarketDataPoint<double>(DateTime.Now, 10.0, 12.0, 9.0, 11.0, 500.0));

        var tensor = provider.ToTensor(includeVolume: false);

        Assert.Equal(1, tensor.Shape[0]);
        Assert.Equal(4, tensor.Shape[1]); // OHLC only
        Assert.Equal(10.0, tensor[0, 0], Tolerance); // Open
        Assert.Equal(11.0, tensor[0, 3], Tolerance); // Close
    }

    [Fact]
    public void MarketDataProvider_AddNull_Throws()
    {
        var provider = new MarketDataProvider<double>();

        Assert.Throws<ArgumentNullException>(() => provider.Add(null!));
    }

    [Fact]
    public void MarketDataProvider_AddRangeNull_Throws()
    {
        var provider = new MarketDataProvider<double>();

        Assert.Throws<ArgumentNullException>(() => provider.AddRange(null!));
    }

    #endregion

    #region FinancialPreprocessor Feature Tensor Tests

    [Fact]
    public void Preprocessor_GetFeatureCount_OHLCV()
    {
        var preprocessor = new FinancialPreprocessor<double>();

        Assert.Equal(4, preprocessor.GetFeatureCount(includeVolume: false, includeReturns: false));
        Assert.Equal(5, preprocessor.GetFeatureCount(includeVolume: true, includeReturns: false));
        Assert.Equal(5, preprocessor.GetFeatureCount(includeVolume: false, includeReturns: true));
        Assert.Equal(6, preprocessor.GetFeatureCount(includeVolume: true, includeReturns: true));
    }

    [Fact]
    public void Preprocessor_CreateFeatureTensor_GoldenReference_OHLCV()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = new List<MarketDataPoint<double>>
        {
            new(new DateTime(2024, 1, 1), 100.0, 110.0, 90.0, 105.0, 5000.0),
            new(new DateTime(2024, 1, 2), 105.0, 115.0, 95.0, 110.0, 6000.0),
        };

        var tensor = preprocessor.CreateFeatureTensor(points, includeVolume: true, includeReturns: false);

        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(5, tensor.Shape[1]);

        // Row 0: O=100, H=110, L=90, C=105, V=5000
        Assert.Equal(100.0, tensor[0 * 5 + 0], Tolerance);
        Assert.Equal(110.0, tensor[0 * 5 + 1], Tolerance);
        Assert.Equal(90.0, tensor[0 * 5 + 2], Tolerance);
        Assert.Equal(105.0, tensor[0 * 5 + 3], Tolerance);
        Assert.Equal(5000.0, tensor[0 * 5 + 4], Tolerance);
    }

    [Fact]
    public void Preprocessor_CreateFeatureTensor_WithReturns_GoldenReference()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = new List<MarketDataPoint<double>>
        {
            new(new DateTime(2024, 1, 1), 100.0, 110.0, 90.0, 100.0, 1000.0),
            new(new DateTime(2024, 1, 2), 105.0, 115.0, 95.0, 110.0, 2000.0),
            new(new DateTime(2024, 1, 3), 110.0, 120.0, 100.0, 105.0, 1500.0),
        };

        var tensor = preprocessor.CreateFeatureTensor(points, includeVolume: true, includeReturns: true);

        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(6, tensor.Shape[1]); // OHLCV + returns

        // First row returns = 0 (no previous)
        Assert.Equal(0.0, tensor[0 * 6 + 5], Tolerance);

        // Second row: return = (110 - 100) / 100 = 0.1
        Assert.Equal(0.1, tensor[1 * 6 + 5], Tolerance);

        // Third row: return = (105 - 110) / 110 = -0.04545...
        Assert.Equal(-5.0 / 110.0, tensor[2 * 6 + 5], Tolerance);
    }

    [Fact]
    public void Preprocessor_CreateFeatureTensor_WithoutVolume_OHLC()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = new List<MarketDataPoint<double>>
        {
            new(new DateTime(2024, 1, 1), 50.0, 55.0, 45.0, 52.0, 1000.0),
        };

        var tensor = preprocessor.CreateFeatureTensor(points, includeVolume: false, includeReturns: false);

        Assert.Equal(1, tensor.Shape[0]);
        Assert.Equal(4, tensor.Shape[1]);
        Assert.Equal(50.0, tensor[0], Tolerance); // Open
        Assert.Equal(52.0, tensor[3], Tolerance); // Close
    }

    [Fact]
    public void Preprocessor_CreateFeatureTensor_EmptySeries_Throws()
    {
        var preprocessor = new FinancialPreprocessor<double>();

        Assert.Throws<ArgumentException>(() =>
            preprocessor.CreateFeatureTensor(new List<MarketDataPoint<double>>()));
    }

    [Fact]
    public void Preprocessor_CreateFeatureTensor_NullSeries_Throws()
    {
        var preprocessor = new FinancialPreprocessor<double>();

        Assert.Throws<ArgumentNullException>(() =>
            preprocessor.CreateFeatureTensor(null!));
    }

    #endregion

    #region Supervised Learning Tensor Tests

    [Fact]
    public void Preprocessor_CreateSupervisedTensors_GoldenReference()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        // 10 points, sequenceLength=3, predictionHorizon=1 => 7 samples
        var points = CreateSeriesWithIncreasingPrices(10, 100.0, 1.0);

        var (features, targets) = preprocessor.CreateSupervisedLearningTensors(
            points, sequenceLength: 3, predictionHorizon: 1,
            includeVolume: false, includeReturns: false, predictReturns: false);

        // sampleCount = 10 - 3 - 1 + 1 = 7
        Assert.Equal(7, features.Shape[0]);
        Assert.Equal(3, features.Shape[1]); // sequenceLength
        Assert.Equal(4, features.Shape[2]); // features (OHLC)

        Assert.Equal(7, targets.Shape[0]);
        Assert.Equal(1, targets.Shape[1]); // predictionHorizon

        // Target for sample 0 = close price at index 3 = 103.0
        Assert.Equal(103.0, targets[0], Tolerance);

        // Target for sample 6 = close price at index 9 = 109.0
        Assert.Equal(109.0, targets[6], Tolerance);
    }

    [Fact]
    public void Preprocessor_CreateSupervisedTensors_MultiHorizon()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = CreateSeriesWithIncreasingPrices(10, 100.0, 1.0);

        var (features, targets) = preprocessor.CreateSupervisedLearningTensors(
            points, sequenceLength: 2, predictionHorizon: 3,
            includeVolume: false, includeReturns: false, predictReturns: false);

        // sampleCount = 10 - 2 - 3 + 1 = 6
        Assert.Equal(6, features.Shape[0]);
        Assert.Equal(2, features.Shape[1]);

        Assert.Equal(6, targets.Shape[0]);
        Assert.Equal(3, targets.Shape[1]); // predictionHorizon

        // Sample 0: targets at indices 2, 3, 4 => closes 102, 103, 104
        Assert.Equal(102.0, targets[0 * 3 + 0], Tolerance);
        Assert.Equal(103.0, targets[0 * 3 + 1], Tolerance);
        Assert.Equal(104.0, targets[0 * 3 + 2], Tolerance);
    }

    [Fact]
    public void Preprocessor_CreateSupervisedTensors_TooFewPoints_EmptyResult()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = CreateSeriesWithIncreasingPrices(3, 100.0, 1.0);

        // sequenceLength=5, predictionHorizon=1, but only 3 points => 0 samples
        var (features, targets) = preprocessor.CreateSupervisedLearningTensors(
            points, sequenceLength: 5, predictionHorizon: 1,
            includeVolume: false, includeReturns: false);

        Assert.Equal(0, features.Shape[0]);
        Assert.Equal(0, targets.Shape[0]);
    }

    [Fact]
    public void Preprocessor_CreateSupervisedTensors_InvalidWindow_Throws()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = CreateSeriesWithIncreasingPrices(10, 100.0, 1.0);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            preprocessor.CreateSupervisedLearningTensors(points, 0, 1));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            preprocessor.CreateSupervisedLearningTensors(points, 3, 0));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            preprocessor.CreateSupervisedLearningTensors(points, -1, 1));
    }

    [Fact]
    public void Preprocessor_CreateSupervisedTensors_PredictReturns_GoldenReference()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = CreateSeriesWithIncreasingPrices(6, 100.0, 10.0);

        var (_, targets) = preprocessor.CreateSupervisedLearningTensors(
            points, sequenceLength: 2, predictionHorizon: 1,
            includeVolume: false, includeReturns: false, predictReturns: true);

        // sampleCount = 6 - 2 - 1 + 1 = 4
        Assert.Equal(4, targets.Shape[0]);

        // Target for sample 0 = return at index 2 = (120-110)/110
        double expectedReturn = 10.0 / 110.0;
        Assert.Equal(expectedReturn, targets[0], Tolerance);
    }

    #endregion

    #region NormalizeMinMax Tests

    [Fact]
    public void Preprocessor_NormalizeMinMax_GoldenReference()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        // 3 samples, 2 features: [[0, 10], [5, 20], [10, 30]]
        var data = new double[] { 0, 10, 5, 20, 10, 30 };
        var input = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(data));

        var normalized = preprocessor.NormalizeMinMax(input, out var stats);

        // Feature 0: min=0, max=10 => [0, 0.5, 1.0]
        Assert.Equal(0.0, normalized[0 * 2 + 0], Tolerance);
        Assert.Equal(0.5, normalized[1 * 2 + 0], Tolerance);
        Assert.Equal(1.0, normalized[2 * 2 + 0], Tolerance);

        // Feature 1: min=10, max=30 => [0, 0.5, 1.0]
        Assert.Equal(0.0, normalized[0 * 2 + 1], Tolerance);
        Assert.Equal(0.5, normalized[1 * 2 + 1], Tolerance);
        Assert.Equal(1.0, normalized[2 * 2 + 1], Tolerance);

        // Stats
        Assert.Equal(0.0, stats.Min[0], Tolerance);
        Assert.Equal(10.0, stats.Max[0], Tolerance);
        Assert.Equal(10.0, stats.Min[1], Tolerance);
        Assert.Equal(30.0, stats.Max[1], Tolerance);
    }

    [Fact]
    public void Preprocessor_NormalizeMinMax_ConstantFeature_SafeRange()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        // All values same for feature 0
        var data = new double[] { 5, 10, 5, 20, 5, 30 };
        var input = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(data));

        var normalized = preprocessor.NormalizeMinMax(input, out _);

        // Constant feature: range=0, safeRange=1 => (5-5)/1 = 0
        Assert.Equal(0.0, normalized[0 * 2 + 0], Tolerance);
        Assert.Equal(0.0, normalized[1 * 2 + 0], Tolerance);
        Assert.Equal(0.0, normalized[2 * 2 + 0], Tolerance);
    }

    [Fact]
    public void Preprocessor_NormalizeMinMax_AllInRange01()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = CreateSeriesWithIncreasingPrices(20, 50.0, 5.0);
        var tensor = preprocessor.CreateFeatureTensor(points, includeVolume: true, includeReturns: false);

        var normalized = preprocessor.NormalizeMinMax(tensor, out _);

        for (int i = 0; i < normalized.Length; i++)
        {
            Assert.True(normalized[i] >= -Tolerance && normalized[i] <= 1.0 + Tolerance,
                $"Index {i}: value {normalized[i]} outside [0, 1]");
        }
    }

    [Fact]
    public void Preprocessor_NormalizeMinMax_EmptyTensor_ReturnsEmpty()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var input = new Tensor<double>(new[] { 0, 4 }, new Vector<double>(Array.Empty<double>()));

        var normalized = preprocessor.NormalizeMinMax(input, out var stats);

        Assert.Equal(0, normalized.Length);
        Assert.Equal(0, stats.Min.Length);
        Assert.Equal(0, stats.Max.Length);
    }

    [Fact]
    public void Preprocessor_NormalizeMinMax_NullInput_Throws()
    {
        var preprocessor = new FinancialPreprocessor<double>();

        Assert.Throws<ArgumentNullException>(() =>
            preprocessor.NormalizeMinMax(null!, out _));
    }

    #endregion

    #region NormalizeZScore Tests

    [Fact]
    public void Preprocessor_NormalizeZScore_GoldenReference()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        // 3 samples, 1 feature: [2, 4, 6]
        // Mean = 4, Std = sqrt(((2-4)^2 + (4-4)^2 + (6-4)^2) / 2) = sqrt(8/2) = 2
        var data = new double[] { 2, 4, 6 };
        var input = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(data));

        var normalized = preprocessor.NormalizeZScore(input, out var stats);

        Assert.Equal(4.0, stats.Mean[0], Tolerance);
        Assert.Equal(2.0, stats.StdDev[0], Tolerance);

        Assert.Equal(-1.0, normalized[0], Tolerance); // (2 - 4) / 2
        Assert.Equal(0.0, normalized[1], Tolerance);   // (4 - 4) / 2
        Assert.Equal(1.0, normalized[2], Tolerance);   // (6 - 4) / 2
    }

    [Fact]
    public void Preprocessor_NormalizeZScore_ConstantFeature_SafeStd()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        // All values same => std=0 => safeStd=1
        var data = new double[] { 7, 7, 7 };
        var input = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(data));

        var normalized = preprocessor.NormalizeZScore(input, out var stats);

        Assert.Equal(7.0, stats.Mean[0], Tolerance);
        Assert.Equal(1.0, stats.StdDev[0], Tolerance); // forced to 1

        // (7 - 7) / 1 = 0
        Assert.Equal(0.0, normalized[0], Tolerance);
        Assert.Equal(0.0, normalized[1], Tolerance);
        Assert.Equal(0.0, normalized[2], Tolerance);
    }

    [Fact]
    public void Preprocessor_NormalizeZScore_MeanApproximatelyZero()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = CreateSeriesWithIncreasingPrices(30, 50.0, 1.0);
        var tensor = preprocessor.CreateFeatureTensor(points, includeVolume: false, includeReturns: false);

        var normalized = preprocessor.NormalizeZScore(tensor, out _);

        // Each feature column should have approximately zero mean
        int features = tensor.Shape[^1];
        int samples = tensor.Shape[0];

        for (int f = 0; f < features; f++)
        {
            double sum = 0;
            for (int s = 0; s < samples; s++)
            {
                sum += normalized[s * features + f];
            }

            double mean = sum / samples;
            Assert.True(Math.Abs(mean) < 1e-4,
                $"Feature {f}: z-score mean {mean} is not approximately zero");
        }
    }

    [Fact]
    public void Preprocessor_NormalizeZScore_EmptyTensor_ReturnsEmpty()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var input = new Tensor<double>(new[] { 0, 4 }, new Vector<double>(Array.Empty<double>()));

        var normalized = preprocessor.NormalizeZScore(input, out var stats);

        Assert.Equal(0, normalized.Length);
        Assert.Equal(0, stats.Mean.Length);
    }

    [Fact]
    public void Preprocessor_NormalizeZScore_NullInput_Throws()
    {
        var preprocessor = new FinancialPreprocessor<double>();

        Assert.Throws<ArgumentNullException>(() =>
            preprocessor.NormalizeZScore(null!, out _));
    }

    #endregion

    #region Returns Computation Edge Cases

    [Fact]
    public void Preprocessor_Returns_NearZeroClose_DoesNotDivideByZero()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = new List<MarketDataPoint<double>>
        {
            new(new DateTime(2024, 1, 1), 0.0, 0.0, 0.0, 0.0, 100.0),     // close=0
            new(new DateTime(2024, 1, 2), 1.0, 1.0, 1.0, 1.0, 100.0),     // return = (1-0)/epsilon
        };

        var tensor = preprocessor.CreateFeatureTensor(points, includeVolume: false, includeReturns: true);

        // Should not throw - uses safe denominator
        Assert.Equal(5, tensor.Shape[1]); // 4 OHLC + 1 return
        Assert.True(double.IsFinite(tensor[1 * 5 + 4]),
            "Return should be finite even with zero-close previous bar");
    }

    [Fact]
    public void Preprocessor_Returns_FirstBarAlwaysZero()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = CreateSeriesWithIncreasingPrices(5, 100.0, 10.0);

        var tensor = preprocessor.CreateFeatureTensor(points, includeVolume: true, includeReturns: true);

        // First bar return is always 0
        Assert.Equal(0.0, tensor[0 * 6 + 5], Tolerance);
    }

    #endregion

    #region End-to-End Pipeline Tests

    [Fact]
    public void Pipeline_MarketData_ToTensor_ToNormalized_RoundTrip()
    {
        var provider = new MarketDataProvider<double>();
        var points = CreateSeriesWithIncreasingPrices(20, 100.0, 2.0);
        provider.AddRange(points);

        var rawTensor = provider.ToTensor(includeVolume: true);

        var preprocessor = new FinancialPreprocessor<double>();
        var normalizedMinMax = preprocessor.NormalizeMinMax(rawTensor, out var mmStats);
        var normalizedZScore = preprocessor.NormalizeZScore(rawTensor, out var zsStats);

        // Both normalized outputs should have same shape
        Assert.Equal(rawTensor.Shape[0], normalizedMinMax.Shape[0]);
        Assert.Equal(rawTensor.Shape[1], normalizedMinMax.Shape[1]);
        Assert.Equal(rawTensor.Shape[0], normalizedZScore.Shape[0]);
        Assert.Equal(rawTensor.Shape[1], normalizedZScore.Shape[1]);

        // MinMax should have values in [0, 1]
        for (int i = 0; i < normalizedMinMax.Length; i++)
        {
            Assert.True(normalizedMinMax[i] >= -Tolerance && normalizedMinMax[i] <= 1.0 + Tolerance);
        }
    }

    [Fact]
    public void Pipeline_PreprocessorToSupervisedLearning_ShapesConsistent()
    {
        var preprocessor = new FinancialPreprocessor<double>();
        var points = CreateSeriesWithIncreasingPrices(50, 100.0, 0.5);
        int seqLen = 10;
        int horizon = 3;

        var (features, targets) = preprocessor.CreateSupervisedLearningTensors(
            points, seqLen, horizon,
            includeVolume: true, includeReturns: true, predictReturns: false);

        int expectedSamples = 50 - seqLen - horizon + 1; // 38
        Assert.Equal(expectedSamples, features.Shape[0]);
        Assert.Equal(seqLen, features.Shape[1]);
        Assert.Equal(6, features.Shape[2]); // OHLCV + returns

        Assert.Equal(expectedSamples, targets.Shape[0]);
        Assert.Equal(horizon, targets.Shape[1]);
    }

    [Fact]
    public void Preprocessor_CreateFeatureTensor_Float_Works()
    {
        var preprocessor = new FinancialPreprocessor<float>();
        var points = new List<MarketDataPoint<float>>
        {
            new(new DateTime(2024, 1, 1), 100f, 110f, 90f, 105f, 5000f),
            new(new DateTime(2024, 1, 2), 105f, 115f, 95f, 110f, 6000f),
        };

        var tensor = preprocessor.CreateFeatureTensor(points, includeVolume: true);

        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(5, tensor.Shape[1]);
        Assert.Equal(100f, tensor[0], 1e-4f);
    }

    #endregion

    #region Helpers

    private static MarketDataPoint<double> CreatePoint(int minuteOffset, double price)
    {
        var ts = new DateTime(2024, 1, 1).AddMinutes(minuteOffset);
        return new MarketDataPoint<double>(ts, price, price + 5, price - 5, price, 1000.0);
    }

    private static List<MarketDataPoint<double>> CreateSeries(int count, double startPrice, double step)
    {
        var series = new List<MarketDataPoint<double>>(count);
        var baseTime = new DateTime(2024, 1, 1);

        for (int i = 0; i < count; i++)
        {
            double price = startPrice + i * step;
            series.Add(new MarketDataPoint<double>(
                baseTime.AddMinutes(i), price, price + 5, price - 5, price, 1000.0));
        }

        return series;
    }

    private static List<MarketDataPoint<double>> CreateSeriesWithIncreasingPrices(
        int count, double startClose, double increment)
    {
        var series = new List<MarketDataPoint<double>>(count);
        var baseTime = new DateTime(2024, 1, 1);

        for (int i = 0; i < count; i++)
        {
            double close = startClose + i * increment;
            double open = close - increment / 2;
            double high = close + 5;
            double low = close - 5;
            double volume = 1000.0 + i * 100;
            series.Add(new MarketDataPoint<double>(baseTime.AddMinutes(i), open, high, low, close, volume));
        }

        return series;
    }

    #endregion
}
