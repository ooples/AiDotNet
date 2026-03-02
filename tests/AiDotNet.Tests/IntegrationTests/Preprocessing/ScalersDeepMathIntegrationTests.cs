using AiDotNet.Preprocessing.Scalers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep math-correctness integration tests for preprocessing scalers.
/// Each test verifies a specific mathematical property or hand-calculated result.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class ScalersDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double MediumTolerance = 1e-6;

    // ========================================================================
    // STANDARD SCALER: z = (x - mean) / std
    // ========================================================================

    #region StandardScaler

    [Fact]
    public void StandardScaler_HandCalculated_SingleColumn()
    {
        // x = {2, 4, 4, 4, 5, 5, 7, 9}
        // mean = 40/8 = 5
        // variance = sum((x-5)^2) / (8-1) = (9+1+1+1+0+0+4+16)/7 = 32/7
        // std = sqrt(32/7) ≈ 2.13809
        // z-scores: (x-5)/std
        var scaler = new StandardScaler<double>();
        var data = CreateMatrix(new double[,] { { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 } });
        var transformed = scaler.FitTransform(data);

        double mean = 5.0;
        double std = Math.Sqrt(32.0 / 7.0);

        Assert.Equal((2.0 - mean) / std, transformed[0, 0], MediumTolerance);
        Assert.Equal((4.0 - mean) / std, transformed[1, 0], MediumTolerance);
        Assert.Equal((9.0 - mean) / std, transformed[7, 0], MediumTolerance);
    }

    [Fact]
    public void StandardScaler_TransformedData_MeanIsZero()
    {
        // After standardization, mean of transformed data should be ~0
        var scaler = new StandardScaler<double>();
        var data = CreateMatrix(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 }, { 4, 40 }, { 5, 50 } });
        var transformed = scaler.FitTransform(data);

        for (int col = 0; col < 2; col++)
        {
            double sum = 0;
            for (int row = 0; row < 5; row++)
                sum += transformed[row, col];
            Assert.Equal(0.0, sum / 5, MediumTolerance);
        }
    }

    [Fact]
    public void StandardScaler_TransformedData_StdIsOne()
    {
        // After standardization, std of transformed data should be ~1
        var scaler = new StandardScaler<double>();
        var data = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 } });
        var transformed = scaler.FitTransform(data);

        // Calculate sample std of transformed data
        double mean = 0;
        for (int i = 0; i < 8; i++) mean += transformed[i, 0];
        mean /= 8;
        double sumSq = 0;
        for (int i = 0; i < 8; i++)
            sumSq += (transformed[i, 0] - mean) * (transformed[i, 0] - mean);
        double std = Math.Sqrt(sumSq / 7);

        Assert.Equal(1.0, std, MediumTolerance);
    }

    [Fact]
    public void StandardScaler_InverseTransform_RecoverOriginal()
    {
        // FitTransform then InverseTransform should give back original data
        var scaler = new StandardScaler<double>();
        double[,] original = { { 1, 100 }, { 2, 200 }, { 3, 300 }, { 4, 400 } };
        var data = CreateMatrix(original);
        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(original[i, j], recovered[i, j], MediumTolerance);
    }

    [Fact]
    public void StandardScaler_WithMeanFalse_OnlyScales()
    {
        // Without centering, should only divide by std
        var scaler = new StandardScaler<double>(withMean: false, withStd: true);
        var data = CreateMatrix(new double[,] { { 2 }, { 4 }, { 6 } });
        scaler.Fit(data);
        var transformed = scaler.Transform(data);

        // mean=4, std=sqrt(((2-4)^2+(4-4)^2+(6-4)^2)/2) = sqrt(4) = 2
        // Without centering: values/std = {1, 2, 3}
        double std = 2.0;
        Assert.Equal(2.0 / std, transformed[0, 0], MediumTolerance);
        Assert.Equal(4.0 / std, transformed[1, 0], MediumTolerance);
        Assert.Equal(6.0 / std, transformed[2, 0], MediumTolerance);
    }

    [Fact]
    public void StandardScaler_WithStdFalse_OnlyCenters()
    {
        // Without scaling, should only subtract mean
        var scaler = new StandardScaler<double>(withMean: true, withStd: false);
        var data = CreateMatrix(new double[,] { { 2 }, { 4 }, { 6 } });
        scaler.Fit(data);
        var transformed = scaler.Transform(data);

        // mean = 4
        Assert.Equal(-2.0, transformed[0, 0], Tolerance);
        Assert.Equal(0.0, transformed[1, 0], Tolerance);
        Assert.Equal(2.0, transformed[2, 0], Tolerance);
    }

    [Fact]
    public void StandardScaler_ConstantColumn_NoScaling()
    {
        // When all values are the same, std=0. Scaler should use 1 instead (no scaling)
        var scaler = new StandardScaler<double>();
        var data = CreateMatrix(new double[,] { { 5 }, { 5 }, { 5 } });
        var transformed = scaler.FitTransform(data);

        // mean=5, std=0→1, so (5-5)/1 = 0 for all rows
        for (int i = 0; i < 3; i++)
            Assert.Equal(0.0, transformed[i, 0], Tolerance);
    }

    [Fact]
    public void StandardScaler_FitAndTransform_SeparateData()
    {
        // Fit on training data, transform test data using training parameters
        var scaler = new StandardScaler<double>();
        var train = CreateMatrix(new double[,] { { 0 }, { 10 } });
        scaler.Fit(train);

        // mean=5, std=sqrt((25+25)/1)=sqrt(50)=5*sqrt(2)
        var test = CreateMatrix(new double[,] { { 5 }, { 15 }, { -5 } });
        var transformed = scaler.Transform(test);

        double mean = 5.0;
        double std = Math.Sqrt(50.0);
        Assert.Equal((5.0 - mean) / std, transformed[0, 0], MediumTolerance);
        Assert.Equal((15.0 - mean) / std, transformed[1, 0], MediumTolerance);
        Assert.Equal((-5.0 - mean) / std, transformed[2, 0], MediumTolerance);
    }

    [Fact]
    public void StandardScaler_ColumnIndices_OnlyScalesSpecifiedColumns()
    {
        // Only scale column 0, leave column 1 untouched
        var scaler = new StandardScaler<double>(columnIndices: new[] { 0 });
        var data = CreateMatrix(new double[,] { { 1, 100 }, { 2, 200 }, { 3, 300 } });
        var transformed = scaler.FitTransform(data);

        // Column 0 should be standardized
        double mean0 = 2.0;
        double std0 = 1.0; // sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/2) = sqrt(1) = 1
        Assert.Equal((1.0 - mean0) / std0, transformed[0, 0], MediumTolerance);

        // Column 1 should be untouched
        Assert.Equal(100.0, transformed[0, 1], Tolerance);
        Assert.Equal(200.0, transformed[1, 1], Tolerance);
        Assert.Equal(300.0, transformed[2, 1], Tolerance);
    }

    #endregion

    // ========================================================================
    // MIN-MAX SCALER: x_scaled = (x - min) / (max - min) * range + range_min
    // ========================================================================

    #region MinMaxScaler

    [Fact]
    public void MinMaxScaler_Default_ScalesTo01()
    {
        // Default range is [0, 1]
        var scaler = new MinMaxScaler<double>();
        var data = CreateMatrix(new double[,] { { 10 }, { 20 }, { 30 }, { 40 }, { 50 } });
        var transformed = scaler.FitTransform(data);

        Assert.Equal(0.0, transformed[0, 0], Tolerance);  // min → 0
        Assert.Equal(0.25, transformed[1, 0], Tolerance);  // (20-10)/(50-10) = 0.25
        Assert.Equal(0.5, transformed[2, 0], Tolerance);   // (30-10)/(50-10) = 0.5
        Assert.Equal(0.75, transformed[3, 0], Tolerance);   // (40-10)/(50-10) = 0.75
        Assert.Equal(1.0, transformed[4, 0], Tolerance);   // max → 1
    }

    [Fact]
    public void MinMaxScaler_CustomRange_Minus1To1()
    {
        // Scale to [-1, 1]
        var scaler = new MinMaxScaler<double>(-1.0, 1.0);
        var data = CreateMatrix(new double[,] { { 0 }, { 50 }, { 100 } });
        var transformed = scaler.FitTransform(data);

        Assert.Equal(-1.0, transformed[0, 0], Tolerance); // min → -1
        Assert.Equal(0.0, transformed[1, 0], Tolerance);  // mid → 0
        Assert.Equal(1.0, transformed[2, 0], Tolerance);  // max → 1
    }

    [Fact]
    public void MinMaxScaler_InverseTransform_RecoverOriginal()
    {
        var scaler = new MinMaxScaler<double>();
        double[,] original = { { 1, 10 }, { 5, 50 }, { 10, 100 } };
        var data = CreateMatrix(original);
        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(original[i, j], recovered[i, j], MediumTolerance);
    }

    [Fact]
    public void MinMaxScaler_ConstantColumn_ScalesToMidpoint()
    {
        // When min == max, should map to midpoint of feature range
        var scaler = new MinMaxScaler<double>();
        var data = CreateMatrix(new double[,] { { 7 }, { 7 }, { 7 } });
        var transformed = scaler.FitTransform(data);

        // Default range [0,1], midpoint = 0.5
        for (int i = 0; i < 3; i++)
            Assert.Equal(0.5, transformed[i, 0], Tolerance);
    }

    [Fact]
    public void MinMaxScaler_NegativeValues_HandlesCorrectly()
    {
        // Data with negative values
        var scaler = new MinMaxScaler<double>();
        var data = CreateMatrix(new double[,] { { -10 }, { 0 }, { 10 } });
        var transformed = scaler.FitTransform(data);

        Assert.Equal(0.0, transformed[0, 0], Tolerance);  // -10 → 0
        Assert.Equal(0.5, transformed[1, 0], Tolerance);   // 0 → 0.5
        Assert.Equal(1.0, transformed[2, 0], Tolerance);   // 10 → 1
    }

    [Fact]
    public void MinMaxScaler_MultipleColumns_IndependentScaling()
    {
        // Each column should be scaled independently
        var scaler = new MinMaxScaler<double>();
        var data = CreateMatrix(new double[,] { { 0, 100 }, { 100, 200 } });
        var transformed = scaler.FitTransform(data);

        // Column 0: min=0, max=100
        Assert.Equal(0.0, transformed[0, 0], Tolerance);
        Assert.Equal(1.0, transformed[1, 0], Tolerance);
        // Column 1: min=100, max=200
        Assert.Equal(0.0, transformed[0, 1], Tolerance);
        Assert.Equal(1.0, transformed[1, 1], Tolerance);
    }

    [Fact]
    public void MinMaxScaler_TransformNewData_UsesTrainingMinMax()
    {
        // Transform new data using training data's min/max
        var scaler = new MinMaxScaler<double>();
        var train = CreateMatrix(new double[,] { { 0 }, { 100 } });
        scaler.Fit(train);

        var test = CreateMatrix(new double[,] { { 50 }, { 150 }, { -50 } });
        var transformed = scaler.Transform(test);

        Assert.Equal(0.5, transformed[0, 0], Tolerance);  // 50/100
        Assert.Equal(1.5, transformed[1, 0], Tolerance);   // 150/100 (out of range is fine)
        Assert.Equal(-0.5, transformed[2, 0], Tolerance);  // -50/100
    }

    [Fact]
    public void MinMaxScaler_CustomRange_InverseRecovery()
    {
        var scaler = new MinMaxScaler<double>(-1.0, 1.0);
        double[,] original = { { -100 }, { 0 }, { 100 } };
        var data = CreateMatrix(original);
        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < 3; i++)
            Assert.Equal(original[i, 0], recovered[i, 0], MediumTolerance);
    }

    #endregion

    // ========================================================================
    // MAX ABS SCALER: x_scaled = x / max(|x|)
    // ========================================================================

    #region MaxAbsScaler

    [Fact]
    public void MaxAbsScaler_HandCalculated()
    {
        // x = {-10, 5, 8}, max_abs = 10
        // scaled: {-1, 0.5, 0.8}
        var scaler = new MaxAbsScaler<double>();
        var data = CreateMatrix(new double[,] { { -10 }, { 5 }, { 8 } });
        var transformed = scaler.FitTransform(data);

        Assert.Equal(-1.0, transformed[0, 0], Tolerance);
        Assert.Equal(0.5, transformed[1, 0], Tolerance);
        Assert.Equal(0.8, transformed[2, 0], Tolerance);
    }

    [Fact]
    public void MaxAbsScaler_OutputInMinus1To1()
    {
        var scaler = new MaxAbsScaler<double>();
        var data = CreateMatrix(new double[,] { { -100 }, { 50 }, { -75 }, { 30 }, { 100 } });
        var transformed = scaler.FitTransform(data);

        for (int i = 0; i < 5; i++)
        {
            Assert.True(transformed[i, 0] >= -1.0 && transformed[i, 0] <= 1.0,
                $"Scaled value {transformed[i, 0]} should be in [-1, 1]");
        }
    }

    [Fact]
    public void MaxAbsScaler_PreservesZeros()
    {
        // Zero values should remain zero
        var scaler = new MaxAbsScaler<double>();
        var data = CreateMatrix(new double[,] { { 0 }, { 10 }, { 0 }, { -5 } });
        var transformed = scaler.FitTransform(data);

        Assert.Equal(0.0, transformed[0, 0], Tolerance);
        Assert.Equal(0.0, transformed[2, 0], Tolerance);
    }

    [Fact]
    public void MaxAbsScaler_InverseTransform_RecoverOriginal()
    {
        var scaler = new MaxAbsScaler<double>();
        double[,] original = { { -3 }, { 0 }, { 5 }, { -8 } };
        var data = CreateMatrix(original);
        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < 4; i++)
            Assert.Equal(original[i, 0], recovered[i, 0], MediumTolerance);
    }

    [Fact]
    public void MaxAbsScaler_AllZeros_NoScaling()
    {
        // When all values are zero, max_abs=0→1, so output should be 0
        var scaler = new MaxAbsScaler<double>();
        var data = CreateMatrix(new double[,] { { 0 }, { 0 }, { 0 } });
        var transformed = scaler.FitTransform(data);

        for (int i = 0; i < 3; i++)
            Assert.Equal(0.0, transformed[i, 0], Tolerance);
    }

    #endregion

    // ========================================================================
    // ROBUST SCALER: x_scaled = (x - median) / IQR
    // ========================================================================

    #region RobustScaler

    [Fact]
    public void RobustScaler_HandCalculated_OddSamples()
    {
        // Sorted: {1, 2, 3, 4, 5, 6, 7, 8, 9}
        // n=9, median = 5 (index 4)
        // Q1 at 25th percentile: index = 0.25*8 = 2.0, so Q1 = sorted[2] = 3
        // Q3 at 75th percentile: index = 0.75*8 = 6.0, so Q3 = sorted[6] = 7
        // IQR = 7 - 3 = 4
        var scaler = new RobustScaler<double>();
        var data = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 } });
        var transformed = scaler.FitTransform(data);

        double median = 5.0;
        double iqr = 4.0;
        Assert.Equal((1.0 - median) / iqr, transformed[0, 0], MediumTolerance);
        Assert.Equal((5.0 - median) / iqr, transformed[4, 0], MediumTolerance); // 0.0
        Assert.Equal((9.0 - median) / iqr, transformed[8, 0], MediumTolerance);
    }

    [Fact]
    public void RobustScaler_MedianValueMapsToZero()
    {
        // The median value should map to 0 after centering
        var scaler = new RobustScaler<double>();
        var data = CreateMatrix(new double[,] { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } });
        var transformed = scaler.FitTransform(data);

        // Median of {1,3,5,7,9} is 5 (index 2)
        Assert.Equal(0.0, transformed[2, 0], MediumTolerance);
    }

    [Fact]
    public void RobustScaler_InverseTransform_RecoverOriginal()
    {
        var scaler = new RobustScaler<double>();
        double[,] original = { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } };
        var data = CreateMatrix(original);
        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < 5; i++)
            Assert.Equal(original[i, 0], recovered[i, 0], MediumTolerance);
    }

    [Fact]
    public void RobustScaler_WithCenteringFalse_OnlyScales()
    {
        var scaler = new RobustScaler<double>(withCentering: false, withScaling: true);
        var data = CreateMatrix(new double[,] { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } });
        scaler.Fit(data);
        var transformed = scaler.Transform(data);

        // No centering, so values are divided by IQR but not shifted
        // Q1=3, Q3=7, IQR=4
        double iqr = 4.0;
        Assert.Equal(1.0 / iqr, transformed[0, 0], MediumTolerance);
        Assert.Equal(9.0 / iqr, transformed[4, 0], MediumTolerance);
    }

    [Fact]
    public void RobustScaler_WithScalingFalse_OnlyCenters()
    {
        var scaler = new RobustScaler<double>(withCentering: true, withScaling: false);
        var data = CreateMatrix(new double[,] { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } });
        scaler.Fit(data);
        var transformed = scaler.Transform(data);

        // Only centering: value - median
        double median = 5.0;
        Assert.Equal(1.0 - median, transformed[0, 0], MediumTolerance);
        Assert.Equal(9.0 - median, transformed[4, 0], MediumTolerance);
    }

    [Fact]
    public void RobustScaler_OutliersDoNotAffectScaling()
    {
        // Add extreme outliers - they shouldn't affect median or IQR much
        var scaler1 = new RobustScaler<double>();
        var data1 = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        scaler1.Fit(data1);

        var scaler2 = new RobustScaler<double>();
        var data2 = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 1000 } });
        scaler2.Fit(data2);

        // Median should be 3 for both
        Assert.Equal(scaler1.Median![0], scaler2.Median![0], MediumTolerance);
    }

    #endregion

    // ========================================================================
    // NORMALIZER: row-wise normalization
    // ========================================================================

    #region Normalizer

    [Fact]
    public void Normalizer_L2_HandCalculated()
    {
        // Row [3, 4]: L2 norm = sqrt(9+16) = 5
        // Normalized: [3/5, 4/5] = [0.6, 0.8]
        var normalizer = new Normalizer<double>(NormType.L2);
        var data = CreateMatrix(new double[,] { { 3, 4 } });
        var transformed = normalizer.FitTransform(data);

        Assert.Equal(0.6, transformed[0, 0], Tolerance);
        Assert.Equal(0.8, transformed[0, 1], Tolerance);
    }

    [Fact]
    public void Normalizer_L2_ResultHasUnitNorm()
    {
        // After L2 normalization, each row should have L2 norm = 1
        var normalizer = new Normalizer<double>(NormType.L2);
        var data = CreateMatrix(new double[,] { { 3, 4 }, { 1, 2 }, { -5, 12 } });
        var transformed = normalizer.FitTransform(data);

        for (int i = 0; i < 3; i++)
        {
            double norm = 0;
            for (int j = 0; j < 2; j++)
                norm += transformed[i, j] * transformed[i, j];
            Assert.Equal(1.0, Math.Sqrt(norm), MediumTolerance);
        }
    }

    [Fact]
    public void Normalizer_L1_HandCalculated()
    {
        // Row [3, -4]: L1 norm = |3|+|-4| = 7
        // Normalized: [3/7, -4/7]
        var normalizer = new Normalizer<double>(NormType.L1);
        var data = CreateMatrix(new double[,] { { 3, -4 } });
        var transformed = normalizer.FitTransform(data);

        Assert.Equal(3.0 / 7.0, transformed[0, 0], Tolerance);
        Assert.Equal(-4.0 / 7.0, transformed[0, 1], Tolerance);
    }

    [Fact]
    public void Normalizer_L1_ResultSumAbsEqualsOne()
    {
        // After L1 normalization, sum of absolute values of each row = 1
        var normalizer = new Normalizer<double>(NormType.L1);
        var data = CreateMatrix(new double[,] { { 3, -4, 5 }, { 1, 2, -3 } });
        var transformed = normalizer.FitTransform(data);

        for (int i = 0; i < 2; i++)
        {
            double absSum = 0;
            for (int j = 0; j < 3; j++)
                absSum += Math.Abs(transformed[i, j]);
            Assert.Equal(1.0, absSum, MediumTolerance);
        }
    }

    [Fact]
    public void Normalizer_Max_HandCalculated()
    {
        // Row [3, -4, 2]: Max norm = max(3, 4, 2) = 4
        // Normalized: [3/4, -4/4, 2/4] = [0.75, -1, 0.5]
        var normalizer = new Normalizer<double>(NormType.Max);
        var data = CreateMatrix(new double[,] { { 3, -4, 2 } });
        var transformed = normalizer.FitTransform(data);

        Assert.Equal(0.75, transformed[0, 0], Tolerance);
        Assert.Equal(-1.0, transformed[0, 1], Tolerance);
        Assert.Equal(0.5, transformed[0, 2], Tolerance);
    }

    [Fact]
    public void Normalizer_Max_MaxAbsValueIsOne()
    {
        // After Max normalization, max absolute value of each row = 1
        var normalizer = new Normalizer<double>(NormType.Max);
        var data = CreateMatrix(new double[,] { { 3, -4, 2 }, { 10, -5, 7 } });
        var transformed = normalizer.FitTransform(data);

        for (int i = 0; i < 2; i++)
        {
            double maxAbs = 0;
            for (int j = 0; j < 3; j++)
                maxAbs = Math.Max(maxAbs, Math.Abs(transformed[i, j]));
            Assert.Equal(1.0, maxAbs, MediumTolerance);
        }
    }

    [Fact]
    public void Normalizer_L2_ZeroRow_HandleGracefully()
    {
        // A row of all zeros has norm 0. Normalizer should handle without NaN.
        var normalizer = new Normalizer<double>(NormType.L2);
        var data = CreateMatrix(new double[,] { { 0, 0, 0 }, { 3, 4, 0 } });
        var transformed = normalizer.FitTransform(data);

        // Zero row: norm is 0, falls back to 1 → still all zeros
        for (int j = 0; j < 3; j++)
        {
            Assert.False(double.IsNaN(transformed[0, j]),
                $"Zero row should not produce NaN at column {j}");
            Assert.Equal(0.0, transformed[0, j], Tolerance);
        }

        // Non-zero row should be normalized properly
        double norm = Math.Sqrt(9 + 16);
        Assert.Equal(3.0 / norm, transformed[1, 0], Tolerance);
    }

    [Fact]
    public void Normalizer_PreservesDirection()
    {
        // L2 normalization preserves the direction (ratio) between elements
        var normalizer = new Normalizer<double>(NormType.L2);
        var data = CreateMatrix(new double[,] { { 6, 8 } }); // ratio 6:8 = 3:4
        var transformed = normalizer.FitTransform(data);

        // After normalization, ratio should be preserved: 0.6:0.8 = 3:4
        double ratio = transformed[0, 0] / transformed[0, 1];
        Assert.Equal(6.0 / 8.0, ratio, Tolerance);
    }

    #endregion

    // ========================================================================
    // LOG SCALER
    // ========================================================================

    #region LogScaler

    [Fact]
    public void LogScaler_PositiveValues_ScalesToZeroOne()
    {
        // All positive values: shift=0, log([1,10,100]) then normalize to [0,1]
        var scaler = new LogScaler<double>();
        var data = CreateMatrix(new double[,] { { 1 }, { 10 }, { 100 } });
        var transformed = scaler.FitTransform(data);

        // min is 1, shift=0, logMin=ln(1)=0, logMax=ln(100)
        // Range: ln(100) - 0 = ln(100)
        Assert.Equal(0.0, transformed[0, 0], MediumTolerance);  // (ln(1)-0)/ln(100) = 0
        Assert.Equal(1.0, transformed[2, 0], MediumTolerance);  // (ln(100)-0)/ln(100) = 1

        // ln(10)/ln(100) = ln(10)/(2*ln(10)) = 0.5
        Assert.Equal(0.5, transformed[1, 0], MediumTolerance);
    }

    [Fact]
    public void LogScaler_NegativeValues_ShiftsToPositive()
    {
        // Data with negatives: min=-5, shift = -(-5) + 1 = 6
        // Shifted: {6-5, 6+0, 6+5} = {1, 6, 11}
        var scaler = new LogScaler<double>();
        var data = CreateMatrix(new double[,] { { -5 }, { 0 }, { 5 } });
        var transformed = scaler.FitTransform(data);

        // All values should be in [0, 1] for training data
        Assert.Equal(0.0, transformed[0, 0], MediumTolerance); // min → 0
        Assert.Equal(1.0, transformed[2, 0], MediumTolerance); // max → 1
        Assert.True(transformed[1, 0] > 0 && transformed[1, 0] < 1,
            $"Middle value should be in (0,1), got {transformed[1, 0]}");
    }

    [Fact]
    public void LogScaler_InverseTransform_RecoverOriginal()
    {
        var scaler = new LogScaler<double>();
        double[,] original = { { 1 }, { 10 }, { 100 }, { 1000 } };
        var data = CreateMatrix(original);
        var transformed = scaler.FitTransform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < 4; i++)
            Assert.Equal(original[i, 0], recovered[i, 0], MediumTolerance);
    }

    [Fact]
    public void LogScaler_EquallySpaced_LogValues_EquallySpaced_Output()
    {
        // Powers of 10: [1, 10, 100, 1000]
        // In log space these are equally spaced: [0, ln10, 2*ln10, 3*ln10]
        // So output should be equally spaced: [0, 1/3, 2/3, 1]
        var scaler = new LogScaler<double>();
        var data = CreateMatrix(new double[,] { { 1 }, { 10 }, { 100 }, { 1000 } });
        var transformed = scaler.FitTransform(data);

        double step = transformed[1, 0] - transformed[0, 0]; // should be 1/3
        Assert.Equal(step, transformed[2, 0] - transformed[1, 0], MediumTolerance);
        Assert.Equal(step, transformed[3, 0] - transformed[2, 0], MediumTolerance);
    }

    #endregion

    // ========================================================================
    // CROSS-SCALER CONSISTENCY
    // ========================================================================

    #region Cross-Scaler Consistency

    [Fact]
    public void StandardScaler_And_MinMaxScaler_BothPreserveMeanRelativeOrder()
    {
        // Both scalers should preserve the relative order of values
        var stdScaler = new StandardScaler<double>();
        var mmScaler = new MinMaxScaler<double>();
        var data = CreateMatrix(new double[,] { { 5 }, { 1 }, { 3 }, { 9 }, { 7 } });

        var stdTransformed = stdScaler.FitTransform(data);
        var mmTransformed = mmScaler.FitTransform(data);

        // Check order is preserved: data[1] < data[2] < data[0] < data[4] < data[3]
        Assert.True(stdTransformed[1, 0] < stdTransformed[2, 0]);
        Assert.True(stdTransformed[2, 0] < stdTransformed[0, 0]);
        Assert.True(stdTransformed[0, 0] < stdTransformed[4, 0]);
        Assert.True(stdTransformed[4, 0] < stdTransformed[3, 0]);

        Assert.True(mmTransformed[1, 0] < mmTransformed[2, 0]);
        Assert.True(mmTransformed[2, 0] < mmTransformed[0, 0]);
        Assert.True(mmTransformed[0, 0] < mmTransformed[4, 0]);
        Assert.True(mmTransformed[4, 0] < mmTransformed[3, 0]);
    }

    [Fact]
    public void AllScalers_InverseTransform_Idempotent()
    {
        // For all invertible scalers: InverseTransform(Transform(x)) ≈ x
        double[,] original = { { 1, 10 }, { 5, 50 }, { 10, 100 }, { 2, 20 }, { 8, 80 } };
        var data = CreateMatrix(original);

        // StandardScaler
        var stdScaler = new StandardScaler<double>();
        var stdResult = stdScaler.FitTransform(data);
        var stdRecovered = stdScaler.InverseTransform(stdResult);

        // MinMaxScaler
        var mmScaler = new MinMaxScaler<double>();
        var mmResult = mmScaler.FitTransform(data);
        var mmRecovered = mmScaler.InverseTransform(mmResult);

        // MaxAbsScaler
        var maScaler = new MaxAbsScaler<double>();
        var maResult = maScaler.FitTransform(data);
        var maRecovered = maScaler.InverseTransform(maResult);

        // RobustScaler
        var robScaler = new RobustScaler<double>();
        var robResult = robScaler.FitTransform(data);
        var robRecovered = robScaler.InverseTransform(robResult);

        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(original[i, j], stdRecovered[i, j], MediumTolerance);
                Assert.Equal(original[i, j], mmRecovered[i, j], MediumTolerance);
                Assert.Equal(original[i, j], maRecovered[i, j], MediumTolerance);
                Assert.Equal(original[i, j], robRecovered[i, j], MediumTolerance);
            }
        }
    }

    [Fact]
    public void MinMaxScaler_InvalidRange_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new MinMaxScaler<double>(1.0, 0.0));
        Assert.Throws<ArgumentException>(() => new MinMaxScaler<double>(5.0, 5.0));
    }

    [Fact]
    public void Scaler_NotFitted_ThrowsOnTransform()
    {
        var scaler = new StandardScaler<double>();
        var data = CreateMatrix(new double[,] { { 1 }, { 2 } });

        Assert.Throws<InvalidOperationException>(() => scaler.Transform(data));
    }

    [Fact]
    public void StandardScaler_TwoSamples_VarianceCalculation()
    {
        // With only 2 samples: mean = 1.5, variance = (0.25+0.25)/1 = 0.5, std = sqrt(0.5)
        var scaler = new StandardScaler<double>();
        var data = CreateMatrix(new double[,] { { 1 }, { 2 } });
        var transformed = scaler.FitTransform(data);

        double mean = 1.5;
        double std = Math.Sqrt(0.5);
        Assert.Equal((1.0 - mean) / std, transformed[0, 0], MediumTolerance);
        Assert.Equal((2.0 - mean) / std, transformed[1, 0], MediumTolerance);
    }

    #endregion

    // ========================================================================
    // HELPERS
    // ========================================================================

    #region Helpers

    private static Matrix<double> CreateMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = data[i, j];
        return matrix;
    }

    #endregion
}
