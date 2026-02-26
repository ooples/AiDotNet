using System;
using System.Linq;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.OutlierHandling;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep mathematical correctness tests for Preprocessing scalers, imputers, and outlier handlers.
/// Each test verifies exact hand-calculated values against industry-standard formulas
/// (scikit-learn, etc.) to catch math bugs in the production code.
/// </summary>
public class PreprocessingDeepMathIntegrationTests
{
    #region Helpers

    private static Matrix<double> M(double[,] data) => new(data);

    private static void AssertCell(Matrix<double> m, int row, int col, double expected, double tol = 1e-10)
    {
        Assert.True(
            Math.Abs(m[row, col] - expected) < tol,
            $"[{row},{col}]: expected {expected}, got {m[row, col]} (diff={Math.Abs(m[row, col] - expected)})");
    }

    #endregion

    #region StandardScaler - Exact Math

    [Fact]
    public void StandardScaler_HandCalculated_ExactTransformValues()
    {
        // data col0: [1, 2, 3, 4, 5]
        // mean = 3, variance (sample, n-1) = sum([4,1,0,1,4])/4 = 10/4 = 2.5
        // std = sqrt(2.5) = 1.58113883...
        // z-scores = [(1-3)/std, (2-3)/std, (3-3)/std, (4-3)/std, (5-3)/std]
        //          = [-1.26491, -0.63246, 0, 0.63246, 1.26491]
        var scaler = new StandardScaler<double>();
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        var result = scaler.FitTransform(data);

        double std = Math.Sqrt(2.5);
        AssertCell(result, 0, 0, (1.0 - 3.0) / std);
        AssertCell(result, 1, 0, (2.0 - 3.0) / std);
        AssertCell(result, 2, 0, 0.0);
        AssertCell(result, 3, 0, (4.0 - 3.0) / std);
        AssertCell(result, 4, 0, (5.0 - 3.0) / std);
    }

    [Fact]
    public void StandardScaler_WithoutMean_OnlyDividesByStd()
    {
        // withMean=false: only divide by std, don't subtract mean
        // data: [2, 4, 6], mean=4, std=sqrt(var)=sqrt(4)=2 (sample var = [(2-4)^2+(4-4)^2+(6-4)^2]/2 = 8/2 = 4)
        // result should be: [2/2, 4/2, 6/2] = [1, 2, 3]
        var scaler = new StandardScaler<double>(withMean: false, withStd: true);
        var data = M(new double[,] { { 2 }, { 4 }, { 6 } });

        var result = scaler.FitTransform(data);

        double std = 2.0; // sqrt(4.0)
        AssertCell(result, 0, 0, 2.0 / std);
        AssertCell(result, 1, 0, 4.0 / std);
        AssertCell(result, 2, 0, 6.0 / std);
    }

    [Fact]
    public void StandardScaler_WithoutStd_OnlySubtractsMean()
    {
        // withStd=false: only subtract mean, don't divide by std
        // data: [10, 20, 30], mean=20
        // result should be: [-10, 0, 10]
        var scaler = new StandardScaler<double>(withMean: true, withStd: false);
        var data = M(new double[,] { { 10 }, { 20 }, { 30 } });

        var result = scaler.FitTransform(data);

        AssertCell(result, 0, 0, -10.0);
        AssertCell(result, 1, 0, 0.0);
        AssertCell(result, 2, 0, 10.0);
    }

    [Fact]
    public void StandardScaler_InverseTransform_RecoversOriginalData()
    {
        var scaler = new StandardScaler<double>();
        var data = M(new double[,] { { 1, 100 }, { 2, 200 }, { 3, 300 }, { 4, 400 } });

        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                AssertCell(recovered, i, j, data[i, j], 1e-8);
            }
        }
    }

    [Fact]
    public void StandardScaler_ConstantColumn_HandlesGracefully()
    {
        // If all values are same, std=0 → should use std=1 (no scaling)
        // data: [5, 5, 5], mean=5, std=0→1
        // result: (5-5)/1 = 0, 0, 0
        var scaler = new StandardScaler<double>();
        var data = M(new double[,] { { 5 }, { 5 }, { 5 } });

        var result = scaler.FitTransform(data);

        AssertCell(result, 0, 0, 0.0);
        AssertCell(result, 1, 0, 0.0);
        AssertCell(result, 2, 0, 0.0);
    }

    [Fact]
    public void StandardScaler_TransformedData_HasZeroMeanUnitVariance()
    {
        var scaler = new StandardScaler<double>();
        var data = M(new double[,] { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } });

        var result = scaler.FitTransform(data);

        // Compute mean of transformed data
        double mean = 0;
        for (int i = 0; i < result.Rows; i++) mean += result[i, 0];
        mean /= result.Rows;
        Assert.True(Math.Abs(mean) < 1e-10, $"Mean should be 0, got {mean}");

        // Compute sample variance of transformed data
        double variance = 0;
        for (int i = 0; i < result.Rows; i++)
        {
            double diff = result[i, 0] - mean;
            variance += diff * diff;
        }
        variance /= (result.Rows - 1);
        Assert.True(Math.Abs(variance - 1.0) < 1e-10, $"Sample variance should be 1, got {variance}");
    }

    [Fact]
    public void StandardScaler_MultiColumn_IndependentScaling()
    {
        // Each column should be scaled independently
        // col0: [0, 10], mean=5, std=sqrt(50)=7.071...
        // col1: [100, 200], mean=150, std=sqrt(5000)=70.71...
        var scaler = new StandardScaler<double>();
        var data = M(new double[,] { { 0, 100 }, { 10, 200 } });

        var result = scaler.FitTransform(data);

        // Both columns should have z-scores of [-1/sqrt(2), 1/sqrt(2)] since n=2
        // sample variance = (x-mean)^2 / (n-1) = 50 / 1 = 50, std = sqrt(50)
        double std0 = Math.Sqrt(50.0);
        double std1 = Math.Sqrt(5000.0);
        AssertCell(result, 0, 0, (0 - 5.0) / std0);
        AssertCell(result, 1, 0, (10 - 5.0) / std0);
        AssertCell(result, 0, 1, (100 - 150.0) / std1);
        AssertCell(result, 1, 1, (200 - 150.0) / std1);
    }

    [Fact]
    public void StandardScaler_SelectiveColumns_OnlyScalesSpecified()
    {
        // Only scale column 0, leave column 1 unchanged
        var scaler = new StandardScaler<double>(columnIndices: new[] { 0 });
        var data = M(new double[,] { { 1, 100 }, { 3, 200 }, { 5, 300 } });

        var result = scaler.FitTransform(data);

        // Column 0 should be scaled, column 1 should be unchanged
        Assert.True(Math.Abs(result[0, 0] - 1.0) > 0.1, "Column 0 should be changed");
        AssertCell(result, 0, 1, 100.0);
        AssertCell(result, 1, 1, 200.0);
        AssertCell(result, 2, 1, 300.0);
    }

    #endregion

    #region MinMaxScaler - Exact Math

    [Fact]
    public void MinMaxScaler_Default01_HandCalculated()
    {
        // data: [2, 4, 6, 8, 10], min=2, max=10
        // scaled = (x - 2) / (10 - 2) = (x - 2) / 8
        // [0/8, 2/8, 4/8, 6/8, 8/8] = [0, 0.25, 0.5, 0.75, 1.0]
        var scaler = new MinMaxScaler<double>();
        var data = M(new double[,] { { 2 }, { 4 }, { 6 }, { 8 }, { 10 } });

        var result = scaler.FitTransform(data);

        AssertCell(result, 0, 0, 0.0);
        AssertCell(result, 1, 0, 0.25);
        AssertCell(result, 2, 0, 0.5);
        AssertCell(result, 3, 0, 0.75);
        AssertCell(result, 4, 0, 1.0);
    }

    [Fact]
    public void MinMaxScaler_CustomRange_HandCalculated()
    {
        // data: [0, 5, 10], min=0, max=10
        // feature range: [-1, 1]
        // scaled = (x - 0) / (10 - 0) * (1 - (-1)) + (-1) = x/10 * 2 - 1
        // [0*2-1, 0.5*2-1, 1*2-1] = [-1, 0, 1]
        var scaler = new MinMaxScaler<double>(-1.0, 1.0);
        var data = M(new double[,] { { 0 }, { 5 }, { 10 } });

        var result = scaler.FitTransform(data);

        AssertCell(result, 0, 0, -1.0);
        AssertCell(result, 1, 0, 0.0);
        AssertCell(result, 2, 0, 1.0);
    }

    [Fact]
    public void MinMaxScaler_InverseTransform_RecoversOriginalData()
    {
        var scaler = new MinMaxScaler<double>(-1.0, 1.0);
        var data = M(new double[,] { { 3, 50 }, { 7, 80 }, { 11, 110 } });

        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                AssertCell(recovered, i, j, data[i, j], 1e-8);
            }
        }
    }

    [Fact]
    public void MinMaxScaler_ConstantColumn_MapsToMidpoint()
    {
        // If min==max (constant column), map to midpoint of feature range
        // feature range [0, 1], midpoint = 0.5
        var scaler = new MinMaxScaler<double>();
        var data = M(new double[,] { { 5 }, { 5 }, { 5 } });

        var result = scaler.FitTransform(data);

        AssertCell(result, 0, 0, 0.5);
        AssertCell(result, 1, 0, 0.5);
        AssertCell(result, 2, 0, 0.5);
    }

    [Fact]
    public void MinMaxScaler_OutOfRangeValues_CanExceedBounds()
    {
        // Fit on [0, 10], then transform [15] → (15-0)/(10-0) = 1.5
        var scaler = new MinMaxScaler<double>();
        var fitData = M(new double[,] { { 0 }, { 10 } });
        scaler.Fit(fitData);

        var testData = M(new double[,] { { 15 }, { -5 } });
        var result = scaler.Transform(testData);

        AssertCell(result, 0, 0, 1.5);
        AssertCell(result, 1, 0, -0.5);
    }

    [Fact]
    public void MinMaxScaler_NegativeValues_HandCalculated()
    {
        // data: [-10, -5, 0, 5, 10], min=-10, max=10, range=20
        // scaled = (x - (-10)) / 20 = (x+10)/20
        // [0/20, 5/20, 10/20, 15/20, 20/20] = [0, 0.25, 0.5, 0.75, 1.0]
        var scaler = new MinMaxScaler<double>();
        var data = M(new double[,] { { -10 }, { -5 }, { 0 }, { 5 }, { 10 } });

        var result = scaler.FitTransform(data);

        AssertCell(result, 0, 0, 0.0);
        AssertCell(result, 1, 0, 0.25);
        AssertCell(result, 2, 0, 0.5);
        AssertCell(result, 3, 0, 0.75);
        AssertCell(result, 4, 0, 1.0);
    }

    #endregion

    #region RobustScaler - Exact Quantile Math

    [Fact]
    public void RobustScaler_HandCalculated_ExactQuantiles()
    {
        // data sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9]  (9 elements)
        // Q1 (25th percentile): index = 0.25 * (9-1) = 2.0 → sorted[2] = 3
        // Q3 (75th percentile): index = 0.75 * (9-1) = 6.0 → sorted[6] = 7
        // median (50th): index = 0.5 * 8 = 4.0 → sorted[4] = 5
        // IQR = Q3 - Q1 = 7 - 3 = 4
        // scaled = (x - median) / IQR = (x - 5) / 4
        var scaler = new RobustScaler<double>();
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 } });

        var result = scaler.FitTransform(data);

        // (1-5)/4 = -1.0, (2-5)/4 = -0.75, ..., (9-5)/4 = 1.0
        AssertCell(result, 0, 0, (1.0 - 5.0) / 4.0);
        AssertCell(result, 1, 0, (2.0 - 5.0) / 4.0);
        AssertCell(result, 2, 0, (3.0 - 5.0) / 4.0);
        AssertCell(result, 3, 0, (4.0 - 5.0) / 4.0);
        AssertCell(result, 4, 0, 0.0);  // median → 0
        AssertCell(result, 5, 0, (6.0 - 5.0) / 4.0);
        AssertCell(result, 6, 0, (7.0 - 5.0) / 4.0);
        AssertCell(result, 7, 0, (8.0 - 5.0) / 4.0);
        AssertCell(result, 8, 0, (9.0 - 5.0) / 4.0);
    }

    [Fact]
    public void RobustScaler_InterpolatedQuantiles_HandCalculated()
    {
        // data sorted: [10, 20, 30, 40]  (4 elements)
        // Q1 (25th percentile): index = 0.25 * (4-1) = 0.75
        //   interpolate between sorted[0]=10 and sorted[1]=20: 10 + 0.75*(20-10) = 17.5
        // Q3 (75th percentile): index = 0.75 * 3 = 2.25
        //   interpolate between sorted[2]=30 and sorted[3]=40: 30 + 0.25*(40-30) = 32.5
        // median (50th): index = 0.5 * 3 = 1.5 → interp between sorted[1]=20 and sorted[2]=30: 25
        // IQR = 32.5 - 17.5 = 15.0
        var scaler = new RobustScaler<double>();
        var data = M(new double[,] { { 10 }, { 20 }, { 30 }, { 40 } });

        var result = scaler.FitTransform(data);

        double median = 25.0;
        double iqr = 15.0;
        AssertCell(result, 0, 0, (10.0 - median) / iqr);
        AssertCell(result, 1, 0, (20.0 - median) / iqr);
        AssertCell(result, 2, 0, (30.0 - median) / iqr);
        AssertCell(result, 3, 0, (40.0 - median) / iqr);
    }

    [Fact]
    public void RobustScaler_InverseTransform_RecoversOriginalData()
    {
        var scaler = new RobustScaler<double>();
        var data = M(new double[,] { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } });

        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            AssertCell(recovered, i, 0, data[i, 0], 1e-8);
        }
    }

    [Fact]
    public void RobustScaler_OutliersDoNotAffectIQR()
    {
        // Data with outliers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]
        // Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000] (10 elements)
        // Q1: index = 0.25*9 = 2.25 → sorted[2] + 0.25*(sorted[3]-sorted[2]) = 3 + 0.25 = 3.25
        // Q3: index = 0.75*9 = 6.75 → sorted[6] + 0.75*(sorted[7]-sorted[6]) = 7 + 0.75 = 7.75
        // IQR = 7.75 - 3.25 = 4.5
        // Median: index = 0.5*9 = 4.5 → 5 + 0.5*(6-5) = 5.5
        var scaler = new RobustScaler<double>();
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 1000 } });

        var result = scaler.FitTransform(data);

        double median = 5.5;
        double iqr = 4.5;
        // The value 1000 should be scaled normally: (1000-5.5)/4.5 ≈ 221.0
        // This is a large value because IQR doesn't absorb outliers
        AssertCell(result, 9, 0, (1000.0 - median) / iqr, 1e-8);
        // Middle values should be small
        AssertCell(result, 4, 0, (5.0 - median) / iqr, 1e-8);
    }

    [Fact]
    public void RobustScaler_CustomQuantileRange_HandCalculated()
    {
        // Use 10th and 90th percentile instead of 25th and 75th
        // data sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] (11 elements)
        // Q10: index = 0.10 * 10 = 1.0 → sorted[1] = 2
        // Q90: index = 0.90 * 10 = 9.0 → sorted[9] = 10
        // IQR = 10 - 2 = 8
        // Median: index = 0.5*10 = 5.0 → sorted[5] = 6
        var scaler = new RobustScaler<double>(10.0, 90.0);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 }, { 11 } });

        var result = scaler.FitTransform(data);

        double median = 6.0;
        double iqr = 8.0;
        AssertCell(result, 0, 0, (1.0 - median) / iqr);
        AssertCell(result, 5, 0, 0.0);  // median → 0
        AssertCell(result, 10, 0, (11.0 - median) / iqr);
    }

    #endregion

    #region ZScoreClipper - Exact Bound Calculations

    [Fact]
    public void ZScoreClipper_HandCalculated_ExactBounds()
    {
        // data: [10, 20, 30, 40, 50]
        // mean = 30, variance (population, /n) = sum([400,100,0,100,400])/5 = 200
        // std = sqrt(200) = 14.14213562...
        // threshold = 2.0
        // lower = 30 - 2*14.142 = 30 - 28.284 = 1.716
        // upper = 30 + 2*14.142 = 30 + 28.284 = 58.284
        var clipper = new ZScoreClipper<double>(threshold: 2.0);
        var data = M(new double[,] { { 10 }, { 20 }, { 30 }, { 40 }, { 50 } });

        clipper.Fit(data);

        double mean = 30.0;
        double std = Math.Sqrt(200.0);
        Assert.True(Math.Abs(clipper.Means[0] - mean) < 1e-10, $"Mean: expected {mean}, got {clipper.Means[0]}");
        Assert.True(Math.Abs(clipper.StandardDeviations[0] - std) < 1e-10, $"Std: expected {std}, got {clipper.StandardDeviations[0]}");
        Assert.True(Math.Abs(clipper.LowerBounds[0] - (mean - 2.0 * std)) < 1e-10);
        Assert.True(Math.Abs(clipper.UpperBounds[0] - (mean + 2.0 * std)) < 1e-10);
    }

    [Fact]
    public void ZScoreClipper_ClipsOutliers_HandCalculated()
    {
        // data: [10, 20, 30, 40, 50], mean=30, std=sqrt(200)=14.142
        // threshold=1.0: bounds = [30-14.142, 30+14.142] = [15.858, 44.142]
        // 10 < 15.858 → clipped to 15.858
        // 50 > 44.142 → clipped to 44.142
        // 20, 30, 40 within bounds → unchanged
        var clipper = new ZScoreClipper<double>(threshold: 1.0);
        var data = M(new double[,] { { 10 }, { 20 }, { 30 }, { 40 }, { 50 } });

        var result = clipper.FitTransform(data);

        double std = Math.Sqrt(200.0);
        double lower = 30.0 - 1.0 * std;
        double upper = 30.0 + 1.0 * std;
        AssertCell(result, 0, 0, lower, 1e-8);  // 10 clipped up to lower
        AssertCell(result, 1, 0, 20.0, 1e-8);   // unchanged
        AssertCell(result, 2, 0, 30.0, 1e-8);   // unchanged
        AssertCell(result, 3, 0, 40.0, 1e-8);   // unchanged
        AssertCell(result, 4, 0, upper, 1e-8);   // 50 clipped down to upper
    }

    [Fact]
    public void ZScoreClipper_OutlierMask_CorrectlyIdentifiesOutliers()
    {
        var clipper = new ZScoreClipper<double>(threshold: 1.0);
        var data = M(new double[,] { { 10 }, { 20 }, { 30 }, { 40 }, { 50 } });

        clipper.Fit(data);
        var mask = clipper.GetOutlierMask(data);

        // With threshold=1, 10 and 50 should be outliers
        Assert.True(mask[0, 0], "10 should be an outlier");
        Assert.False(mask[1, 0], "20 should not be an outlier");
        Assert.False(mask[2, 0], "30 should not be an outlier");
        Assert.False(mask[3, 0], "40 should not be an outlier");
        Assert.True(mask[4, 0], "50 should be an outlier");
    }

    [Fact]
    public void ZScoreClipper_GetZScores_HandCalculated()
    {
        // data: [10, 20, 30, 40, 50], mean=30, std=sqrt(200)
        // z-scores: (x-30)/sqrt(200)
        var clipper = new ZScoreClipper<double>();
        var data = M(new double[,] { { 10 }, { 20 }, { 30 }, { 40 }, { 50 } });

        clipper.Fit(data);
        var zScores = clipper.GetZScores(data);

        double std = Math.Sqrt(200.0);
        AssertCell(zScores, 0, 0, (10.0 - 30.0) / std, 1e-8);
        AssertCell(zScores, 2, 0, 0.0, 1e-8);  // mean → z=0
        AssertCell(zScores, 4, 0, (50.0 - 30.0) / std, 1e-8);
    }

    [Fact]
    public void ZScoreClipper_ConstantColumn_NoClipping()
    {
        // If std=0, all values are same → no clipping needed (bounds set to min/max)
        var clipper = new ZScoreClipper<double>(threshold: 2.0);
        var data = M(new double[,] { { 5 }, { 5 }, { 5 } });

        var result = clipper.FitTransform(data);

        AssertCell(result, 0, 0, 5.0);
        AssertCell(result, 1, 0, 5.0);
        AssertCell(result, 2, 0, 5.0);
    }

    [Fact]
    public void ZScoreClipper_UsesPopulationVariance_NotSample()
    {
        // ZScoreClipper divides by N, not N-1
        // data: [0, 10], mean=5
        // Population variance: (25+25)/2 = 25, std = 5
        // Sample variance: (25+25)/1 = 50, std = 7.071
        // With threshold=1:
        //   Population: bounds = [5-5, 5+5] = [0, 10] → no clipping
        //   Sample: bounds = [5-7.071, 5+7.071] = [-2.071, 12.071] → no clipping (wider)
        var clipper = new ZScoreClipper<double>(threshold: 1.0);
        var data = M(new double[,] { { 0 }, { 10 } });

        clipper.Fit(data);

        double populationStd = 5.0;
        Assert.True(Math.Abs(clipper.StandardDeviations[0] - populationStd) < 1e-10,
            $"ZScoreClipper should use population std={populationStd}, got {clipper.StandardDeviations[0]}");
    }

    #endregion

    #region IQRClipper - Exact Bound Calculations

    [Fact]
    public void IQRClipper_HandCalculated_ExactBounds()
    {
        // data sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9] (9 elements)
        // Q1: index = 0.25*8 = 2.0 → sorted[2] = 3
        // Q3: index = 0.75*8 = 6.0 → sorted[6] = 7
        // IQR = 7 - 3 = 4
        // multiplier = 1.5
        // lower = 3 - 1.5*4 = 3 - 6 = -3
        // upper = 7 + 1.5*4 = 7 + 6 = 13
        var clipper = new IQRClipper<double>(multiplier: 1.5);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 } });

        clipper.Fit(data);

        Assert.True(Math.Abs(clipper.Q1Values[0] - 3.0) < 1e-10, $"Q1: expected 3, got {clipper.Q1Values[0]}");
        Assert.True(Math.Abs(clipper.Q3Values[0] - 7.0) < 1e-10, $"Q3: expected 7, got {clipper.Q3Values[0]}");
        Assert.True(Math.Abs(clipper.IQRValues[0] - 4.0) < 1e-10, $"IQR: expected 4, got {clipper.IQRValues[0]}");
        Assert.True(Math.Abs(clipper.LowerBounds[0] - (-3.0)) < 1e-10, $"Lower: expected -3, got {clipper.LowerBounds[0]}");
        Assert.True(Math.Abs(clipper.UpperBounds[0] - 13.0) < 1e-10, $"Upper: expected 13, got {clipper.UpperBounds[0]}");
    }

    [Fact]
    public void IQRClipper_ClipsOutliers_HandCalculated()
    {
        // data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100] (10 elements, 100 is outlier)
        // sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        // Q1: index = 0.25*9 = 2.25 → 3 + 0.25*(4-3) = 3.25
        // Q3: index = 0.75*9 = 6.75 → 7 + 0.75*(8-7) = 7.75
        // IQR = 7.75 - 3.25 = 4.5
        // multiplier = 1.5
        // lower = 3.25 - 1.5*4.5 = 3.25 - 6.75 = -3.5
        // upper = 7.75 + 1.5*4.5 = 7.75 + 6.75 = 14.5
        // 100 > 14.5 → clipped to 14.5
        var clipper = new IQRClipper<double>(multiplier: 1.5);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 100 } });

        var result = clipper.FitTransform(data);

        AssertCell(result, 0, 0, 1.0, 1e-8);    // within bounds
        AssertCell(result, 9, 0, 14.5, 1e-8);   // clipped to upper bound
    }

    [Fact]
    public void IQRClipper_CountOutliers_HandCalculated()
    {
        var clipper = new IQRClipper<double>(multiplier: 1.5);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 100 } });

        clipper.Fit(data);
        var counts = clipper.CountOutliersPerFeature(data);

        // Only 100 should be an outlier (lower bound = -3.5, so 1 is within bounds)
        Assert.Equal(1, counts[0]);
    }

    [Fact]
    public void IQRClipper_EvenElements_InterpolatedQuantiles()
    {
        // data sorted: [10, 20, 30, 40] (4 elements)
        // Q1: index = 0.25*3 = 0.75 → 10 + 0.75*(20-10) = 17.5
        // Q3: index = 0.75*3 = 2.25 → 30 + 0.25*(40-30) = 32.5
        // IQR = 32.5 - 17.5 = 15.0
        var clipper = new IQRClipper<double>();
        var data = M(new double[,] { { 10 }, { 20 }, { 30 }, { 40 } });

        clipper.Fit(data);

        Assert.True(Math.Abs(clipper.Q1Values[0] - 17.5) < 1e-10, $"Q1: expected 17.5, got {clipper.Q1Values[0]}");
        Assert.True(Math.Abs(clipper.Q3Values[0] - 32.5) < 1e-10, $"Q3: expected 32.5, got {clipper.Q3Values[0]}");
        Assert.True(Math.Abs(clipper.IQRValues[0] - 15.0) < 1e-10, $"IQR: expected 15, got {clipper.IQRValues[0]}");
    }

    #endregion

    #region SimpleImputer - Strategy Correctness

    [Fact]
    public void SimpleImputer_MeanStrategy_HandCalculated()
    {
        // col0: [1, NaN, 3, NaN, 5], valid = [1, 3, 5], mean = 3.0
        // After imputation: [1, 3, 3, 3, 5]
        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        var data = M(new double[,] { { 1 }, { double.NaN }, { 3 }, { double.NaN }, { 5 } });

        var result = imputer.FitTransform(data);

        AssertCell(result, 0, 0, 1.0);
        AssertCell(result, 1, 0, 3.0);
        AssertCell(result, 2, 0, 3.0);
        AssertCell(result, 3, 0, 3.0);
        AssertCell(result, 4, 0, 5.0);
    }

    [Fact]
    public void SimpleImputer_MedianStrategy_OddCount_HandCalculated()
    {
        // col0: [1, NaN, 3, 7, 5], valid sorted = [1, 3, 5, 7], length=4
        // median = (sorted[1] + sorted[2]) / 2 = (3+5)/2 = 4.0
        var imputer = new SimpleImputer<double>(ImputationStrategy.Median);
        var data = M(new double[,] { { 1 }, { double.NaN }, { 3 }, { 7 }, { 5 } });

        var result = imputer.FitTransform(data);

        AssertCell(result, 0, 0, 1.0);
        AssertCell(result, 1, 0, 4.0);  // NaN replaced with median=4
        AssertCell(result, 2, 0, 3.0);
        AssertCell(result, 3, 0, 7.0);
        AssertCell(result, 4, 0, 5.0);
    }

    [Fact]
    public void SimpleImputer_MedianStrategy_EvenCount_HandCalculated()
    {
        // col0: [2, NaN, 8, 4, NaN, 6], valid sorted = [2, 4, 6, 8], length=4
        // median = (sorted[1] + sorted[2]) / 2 = (4 + 6) / 2 = 5.0
        var imputer = new SimpleImputer<double>(ImputationStrategy.Median);
        var data = M(new double[,] { { 2 }, { double.NaN }, { 8 }, { 4 }, { double.NaN }, { 6 } });

        var result = imputer.FitTransform(data);

        AssertCell(result, 1, 0, 5.0);
        AssertCell(result, 4, 0, 5.0);
    }

    [Fact]
    public void SimpleImputer_MostFrequentStrategy_HandCalculated()
    {
        // col0: [1, 2, 2, NaN, 3, 2], valid = [1, 2, 2, 3, 2], most frequent = 2 (appears 3x)
        var imputer = new SimpleImputer<double>(ImputationStrategy.MostFrequent);
        var data = M(new double[,] { { 1 }, { 2 }, { 2 }, { double.NaN }, { 3 }, { 2 } });

        var result = imputer.FitTransform(data);

        AssertCell(result, 3, 0, 2.0);  // NaN replaced with most frequent value
    }

    [Fact]
    public void SimpleImputer_ConstantStrategy_UsesGivenValue()
    {
        var imputer = new SimpleImputer<double>(ImputationStrategy.Constant, fillValue: -999.0);
        var data = M(new double[,] { { 1 }, { double.NaN }, { 3 } });

        var result = imputer.FitTransform(data);

        AssertCell(result, 0, 0, 1.0);
        AssertCell(result, 1, 0, -999.0);
        AssertCell(result, 2, 0, 3.0);
    }

    [Fact]
    public void SimpleImputer_NoMissingValues_DataUnchanged()
    {
        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 } });

        var result = imputer.FitTransform(data);

        AssertCell(result, 0, 0, 1.0);
        AssertCell(result, 1, 0, 2.0);
        AssertCell(result, 2, 0, 3.0);
    }

    [Fact]
    public void SimpleImputer_MeanStrategy_MultiColumn_Independent()
    {
        // col0: [1, NaN, 5], mean = (1+5)/2 = 3
        // col1: [10, 20, NaN], mean = (10+20)/2 = 15
        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        var data = M(new double[,] { { 1, 10 }, { double.NaN, 20 }, { 5, double.NaN } });

        var result = imputer.FitTransform(data);

        AssertCell(result, 1, 0, 3.0);   // col0 NaN → mean=3
        AssertCell(result, 2, 1, 15.0);  // col1 NaN → mean=15
    }

    [Fact]
    public void SimpleImputer_AllMissing_ReturnsZero()
    {
        // All values are NaN → should return 0 for Mean strategy
        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        var data = M(new double[,] { { double.NaN }, { double.NaN }, { double.NaN } });

        var result = imputer.FitTransform(data);

        AssertCell(result, 0, 0, 0.0);
        AssertCell(result, 1, 0, 0.0);
        AssertCell(result, 2, 0, 0.0);
    }

    #endregion

    #region Scaler Properties and Invariants

    [Fact]
    public void MinMaxScaler_OutputRange_IsExactly01()
    {
        var scaler = new MinMaxScaler<double>();
        var data = M(new double[,] { { -5 }, { 0 }, { 3 }, { 7 }, { 15 } });

        var result = scaler.FitTransform(data);

        // Find min and max of output
        double minVal = double.MaxValue, maxVal = double.MinValue;
        for (int i = 0; i < result.Rows; i++)
        {
            if (result[i, 0] < minVal) minVal = result[i, 0];
            if (result[i, 0] > maxVal) maxVal = result[i, 0];
        }

        Assert.Equal(0.0, minVal, 10);
        Assert.Equal(1.0, maxVal, 10);
    }

    [Fact]
    public void RobustScaler_MedianValue_MapsToZero()
    {
        // After robust scaling, the median should map to 0
        var scaler = new RobustScaler<double>();
        var data = M(new double[,] { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } });

        var result = scaler.FitTransform(data);

        // Median of [1,3,5,7,9] is 5, which should map to 0
        AssertCell(result, 2, 0, 0.0);
    }

    [Fact]
    public void StandardScaler_FitExposesCorrectParameters()
    {
        // data: [2, 4, 6], mean=4, sample_var=4, std=2
        var scaler = new StandardScaler<double>();
        var data = M(new double[,] { { 2 }, { 4 }, { 6 } });

        scaler.Fit(data);

        Assert.True(Math.Abs(scaler.Mean[0] - 4.0) < 1e-10, $"Mean: expected 4, got {scaler.Mean[0]}");
        Assert.True(Math.Abs(scaler.StandardDeviation[0] - 2.0) < 1e-10, $"Std: expected 2, got {scaler.StandardDeviation[0]}");
    }

    [Fact]
    public void MinMaxScaler_FitExposesCorrectMinMax()
    {
        var scaler = new MinMaxScaler<double>();
        var data = M(new double[,] { { 3 }, { 7 }, { 11 } });

        scaler.Fit(data);

        Assert.True(Math.Abs(scaler.DataMin[0] - 3.0) < 1e-10, $"DataMin: expected 3, got {scaler.DataMin[0]}");
        Assert.True(Math.Abs(scaler.DataMax[0] - 11.0) < 1e-10, $"DataMax: expected 11, got {scaler.DataMax[0]}");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void StandardScaler_SingleRow_StdIsZero_FallbackToOne()
    {
        // With only one row, sample variance = 0 (n-1 = 0 would cause div-by-zero)
        // StatisticsHelper.CalculateVariance uses n-1, so with n=1 it returns 0
        // Scaler should handle this gracefully (fallback std=1)
        var scaler = new StandardScaler<double>();
        var data = M(new double[,] { { 42 } });

        var result = scaler.FitTransform(data);

        // With std=1 fallback: (42 - 42) / 1 = 0
        AssertCell(result, 0, 0, 0.0);
    }

    [Fact]
    public void MinMaxScaler_TwoRows_HandCalculated()
    {
        var scaler = new MinMaxScaler<double>();
        var data = M(new double[,] { { 5 }, { 15 } });

        var result = scaler.FitTransform(data);

        // min=5, max=15, range=10
        // (5-5)/10 = 0, (15-5)/10 = 1
        AssertCell(result, 0, 0, 0.0);
        AssertCell(result, 1, 0, 1.0);
    }

    [Fact]
    public void AllScalers_TransformBeforeFit_ThrowsInvalidOperation()
    {
        var standardScaler = new StandardScaler<double>();
        var minMaxScaler = new MinMaxScaler<double>();
        var robustScaler = new RobustScaler<double>();
        var data = M(new double[,] { { 1 }, { 2 } });

        Assert.Throws<InvalidOperationException>(() => standardScaler.Transform(data));
        Assert.Throws<InvalidOperationException>(() => minMaxScaler.Transform(data));
        Assert.Throws<InvalidOperationException>(() => robustScaler.Transform(data));
    }

    [Fact]
    public void ZScoreClipper_ThresholdMustBePositive()
    {
        Assert.Throws<ArgumentException>(() => new ZScoreClipper<double>(threshold: 0));
        Assert.Throws<ArgumentException>(() => new ZScoreClipper<double>(threshold: -1.0));
    }

    [Fact]
    public void IQRClipper_MultiplierMustBePositive()
    {
        Assert.Throws<ArgumentException>(() => new IQRClipper<double>(multiplier: 0));
        Assert.Throws<ArgumentException>(() => new IQRClipper<double>(multiplier: -1.0));
    }

    [Fact]
    public void MinMaxScaler_InvalidRange_Throws()
    {
        Assert.Throws<ArgumentException>(() => new MinMaxScaler<double>(1.0, 1.0));
        Assert.Throws<ArgumentException>(() => new MinMaxScaler<double>(1.0, 0.0));
    }

    [Fact]
    public void RobustScaler_InvalidQuantileRange_Throws()
    {
        Assert.Throws<ArgumentException>(() => new RobustScaler<double>(75.0, 25.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RobustScaler<double>(-1.0, 75.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RobustScaler<double>(25.0, 101.0));
    }

    [Fact]
    public void SimpleImputer_TransformBeforeFit_ThrowsInvalidOperation()
    {
        var imputer = new SimpleImputer<double>();
        var data = M(new double[,] { { 1 }, { 2 } });

        Assert.Throws<InvalidOperationException>(() => imputer.Transform(data));
    }

    [Fact]
    public void ZScoreClipper_TransformBeforeFit_ThrowsInvalidOperation()
    {
        var clipper = new ZScoreClipper<double>();
        var data = M(new double[,] { { 1 }, { 2 } });

        Assert.Throws<InvalidOperationException>(() => clipper.Transform(data));
    }

    #endregion

    #region Cross-Scaler Consistency

    [Fact]
    public void StandardAndZScore_UseConsistentVarianceFormula()
    {
        // StandardScaler uses StatisticsHelper.CalculateVariance (sample variance, n-1)
        // ZScoreClipper uses population variance (n)
        // This test documents the discrepancy:
        // data: [1, 2, 3, 4, 5], mean=3
        // Population var = 10/5 = 2.0, pop std = sqrt(2) = 1.414
        // Sample var = 10/4 = 2.5, sample std = sqrt(2.5) = 1.581
        var standardScaler = new StandardScaler<double>();
        var zScoreClipper = new ZScoreClipper<double>(threshold: 3.0);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        standardScaler.Fit(data);
        zScoreClipper.Fit(data);

        double stdScalerStd = standardScaler.StandardDeviation[0];
        double zScoreClipperStd = zScoreClipper.StandardDeviations[0];

        // StandardScaler uses sample std (n-1)
        Assert.True(Math.Abs(stdScalerStd - Math.Sqrt(2.5)) < 1e-10,
            $"StandardScaler std should be sqrt(2.5)={Math.Sqrt(2.5)}, got {stdScalerStd}");

        // ZScoreClipper uses population std (n)
        Assert.True(Math.Abs(zScoreClipperStd - Math.Sqrt(2.0)) < 1e-10,
            $"ZScoreClipper std should be sqrt(2.0)={Math.Sqrt(2.0)}, got {zScoreClipperStd}");
    }

    [Fact]
    public void MinMax_ThenInverse_IdentityForAllValues()
    {
        // Round-trip test with various data ranges
        var scaler = new MinMaxScaler<double>(-5.0, 5.0);
        var data = M(new double[,] {
            { -100, 0.001 },
            { 0, 1.0 },
            { 100, 999.999 }
        });

        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                AssertCell(recovered, i, j, data[i, j], 1e-6);
            }
        }
    }

    [Fact]
    public void Robust_ThenInverse_IdentityForAllValues()
    {
        var scaler = new RobustScaler<double>();
        var data = M(new double[,] {
            { -100 },
            { 0 },
            { 50 },
            { 100 },
            { 1000 }
        });

        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            AssertCell(recovered, i, 0, data[i, 0], 1e-6);
        }
    }

    #endregion
}
