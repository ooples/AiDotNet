using AiDotNet.Preprocessing.OutlierHandling;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep math-correctness integration tests for outlier handling:
/// ZScoreClipper, IQRClipper, MADClipper, Winsorizer, ThresholdClipper.
/// Each test hand-computes expected values and verifies code matches.
/// </summary>
public class OutlierHandlingDeepMathIntegrationTests
{
    private const double Tol = 1e-8;

    private static Matrix<double> MakeMatrix(double[,] data) => new(data);

    // ========================================================================
    // ZScoreClipper - Mean, Std, and Bounds
    // ========================================================================

    [Fact]
    public void ZScore_FitComputes_MeanAndStd_SingleColumn()
    {
        // Data: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 40/8 = 5
        // Population std = sqrt((9+1+1+1+0+0+4+16)/8) = sqrt(32/8) = sqrt(4) = 2
        var data = MakeMatrix(new double[,] {
            { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 3.0);
        clipper.Fit(data);

        Assert.NotNull(clipper.Means);
        Assert.NotNull(clipper.StandardDeviations);
        Assert.Equal(5.0, clipper.Means[0], Tol);
        Assert.Equal(2.0, clipper.StandardDeviations[0], Tol);
    }

    [Fact]
    public void ZScore_Bounds_AreCorrect_Threshold3()
    {
        // Mean=5, Std=2, threshold=3
        // Lower = 5 - 3*2 = -1, Upper = 5 + 3*2 = 11
        var data = MakeMatrix(new double[,] {
            { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 3.0);
        clipper.Fit(data);

        Assert.NotNull(clipper.LowerBounds);
        Assert.NotNull(clipper.UpperBounds);
        Assert.Equal(-1.0, clipper.LowerBounds[0], Tol);
        Assert.Equal(11.0, clipper.UpperBounds[0], Tol);
    }

    [Fact]
    public void ZScore_Bounds_CustomThreshold2()
    {
        // Mean=5, Std=2, threshold=2
        // Lower = 5 - 2*2 = 1, Upper = 5 + 2*2 = 9
        var data = MakeMatrix(new double[,] {
            { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 2.0);
        clipper.Fit(data);

        Assert.NotNull(clipper.LowerBounds);
        Assert.NotNull(clipper.UpperBounds);
        Assert.Equal(1.0, clipper.LowerBounds[0], Tol);
        Assert.Equal(9.0, clipper.UpperBounds[0], Tol);
    }

    [Fact]
    public void ZScore_ClipsOutliers_Correctly()
    {
        // Fit on [2,4,4,4,5,5,7,9], Mean=5, Std=2, threshold=2
        // Bounds: [1, 9]
        // Transform [0, 5, 10] -> [1, 5, 9]
        var data = MakeMatrix(new double[,] {
            { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 2.0);
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { 0 }, { 5 }, { 10 } });
        var result = clipper.Transform(test);

        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(5.0, result[1, 0], Tol);
        Assert.Equal(9.0, result[2, 0], Tol);
    }

    [Fact]
    public void ZScore_InBoundValues_Unchanged()
    {
        var data = MakeMatrix(new double[,] {
            { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 3.0);
        clipper.Fit(data);

        // Bounds are [-1, 11], so values 2 and 9 should not change
        var test = MakeMatrix(new double[,] { { 2 }, { 9 } });
        var result = clipper.Transform(test);

        Assert.Equal(2.0, result[0, 0], Tol);
        Assert.Equal(9.0, result[1, 0], Tol);
    }

    [Fact]
    public void ZScore_GetZScores_HandComputed()
    {
        // Mean=5, Std=2
        // Z-score for 2: (2-5)/2 = -1.5
        // Z-score for 5: (5-5)/2 = 0
        // Z-score for 9: (9-5)/2 = 2
        var data = MakeMatrix(new double[,] {
            { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 3.0);
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { 2 }, { 5 }, { 9 } });
        var zScores = clipper.GetZScores(test);

        Assert.Equal(-1.5, zScores[0, 0], Tol);
        Assert.Equal(0.0, zScores[1, 0], Tol);
        Assert.Equal(2.0, zScores[2, 0], Tol);
    }

    [Fact]
    public void ZScore_OutlierMask_DetectsOutliers()
    {
        // Mean=5, Std=2, threshold=2 => Bounds [1, 9]
        var data = MakeMatrix(new double[,] {
            { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 2.0);
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { 0 }, { 5 }, { 10 } });
        var mask = clipper.GetOutlierMask(test);

        Assert.True(mask[0, 0]);   // 0 < 1 => outlier
        Assert.False(mask[1, 0]);  // 5 in [1,9] => not outlier
        Assert.True(mask[2, 0]);   // 10 > 9 => outlier
    }

    [Fact]
    public void ZScore_ConstantColumn_NoBounds()
    {
        // All same value => std=0 => bounds [MinValue, MaxValue]
        var data = MakeMatrix(new double[,] { { 5 }, { 5 }, { 5 }, { 5 } });
        var clipper = new ZScoreClipper<double>(threshold: 3.0);
        clipper.Fit(data);

        Assert.NotNull(clipper.StandardDeviations);
        Assert.Equal(0.0, clipper.StandardDeviations[0], Tol);

        // Transform should not clip anything
        var test = MakeMatrix(new double[,] { { 100 }, { -100 } });
        var result = clipper.Transform(test);

        Assert.Equal(100.0, result[0, 0], Tol);
        Assert.Equal(-100.0, result[1, 0], Tol);
    }

    [Fact]
    public void ZScore_InvalidThreshold_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ZScoreClipper<double>(threshold: 0));
        Assert.Throws<ArgumentException>(() => new ZScoreClipper<double>(threshold: -1));
    }

    [Fact]
    public void ZScore_MultipleColumns_IndependentStats()
    {
        // Col 0: [1, 3, 5, 7] => mean=4, std=sqrt((9+1+1+9)/4) = sqrt(5) ~ 2.236
        // Col 1: [10, 20, 30, 40] => mean=25, std=sqrt((225+25+25+225)/4) = sqrt(125) ~ 11.180
        var data = MakeMatrix(new double[,] {
            { 1, 10 }, { 3, 20 }, { 5, 30 }, { 7, 40 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 3.0);
        clipper.Fit(data);

        Assert.NotNull(clipper.Means);
        Assert.NotNull(clipper.StandardDeviations);
        Assert.Equal(4.0, clipper.Means[0], Tol);
        Assert.Equal(25.0, clipper.Means[1], Tol);
        Assert.Equal(Math.Sqrt(5.0), clipper.StandardDeviations[0], Tol);
        Assert.Equal(Math.Sqrt(125.0), clipper.StandardDeviations[1], Tol);
    }

    // ========================================================================
    // IQRClipper - Quartiles and IQR-Based Bounds
    // ========================================================================

    [Fact]
    public void IQR_FitComputes_Q1_Q3_IQR()
    {
        // Data (sorted): [1, 2, 3, 4, 5, 6, 7, 8, 9]
        // Using percentile = index * (n-1) / 100:
        // Q1 (25th): index = 0.25 * 8 = 2.0 => value = 3
        // Q3 (75th): index = 0.75 * 8 = 6.0 => value = 7
        // IQR = 7 - 3 = 4
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var clipper = new IQRClipper<double>(multiplier: 1.5);
        clipper.Fit(data);

        Assert.NotNull(clipper.Q1Values);
        Assert.NotNull(clipper.Q3Values);
        Assert.NotNull(clipper.IQRValues);
        Assert.Equal(3.0, clipper.Q1Values[0], Tol);
        Assert.Equal(7.0, clipper.Q3Values[0], Tol);
        Assert.Equal(4.0, clipper.IQRValues[0], Tol);
    }

    [Fact]
    public void IQR_Bounds_Standard_Multiplier1_5()
    {
        // Q1=3, Q3=7, IQR=4, multiplier=1.5
        // Lower = 3 - 1.5*4 = 3 - 6 = -3
        // Upper = 7 + 1.5*4 = 7 + 6 = 13
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var clipper = new IQRClipper<double>(multiplier: 1.5);
        clipper.Fit(data);

        Assert.NotNull(clipper.LowerBounds);
        Assert.NotNull(clipper.UpperBounds);
        Assert.Equal(-3.0, clipper.LowerBounds[0], Tol);
        Assert.Equal(13.0, clipper.UpperBounds[0], Tol);
    }

    [Fact]
    public void IQR_Bounds_Extreme_Multiplier3()
    {
        // Q1=3, Q3=7, IQR=4, multiplier=3.0
        // Lower = 3 - 3*4 = 3 - 12 = -9
        // Upper = 7 + 3*4 = 7 + 12 = 19
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var clipper = new IQRClipper<double>(multiplier: 3.0);
        clipper.Fit(data);

        Assert.NotNull(clipper.LowerBounds);
        Assert.NotNull(clipper.UpperBounds);
        Assert.Equal(-9.0, clipper.LowerBounds[0], Tol);
        Assert.Equal(19.0, clipper.UpperBounds[0], Tol);
    }

    [Fact]
    public void IQR_ClipsOutliers_Correctly()
    {
        // Bounds [-3, 13] with multiplier=1.5
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var clipper = new IQRClipper<double>(multiplier: 1.5);
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -10 }, { 5 }, { 20 } });
        var result = clipper.Transform(test);

        Assert.Equal(-3.0, result[0, 0], Tol);
        Assert.Equal(5.0, result[1, 0], Tol);
        Assert.Equal(13.0, result[2, 0], Tol);
    }

    [Fact]
    public void IQR_Percentile_Interpolation()
    {
        // 4 elements sorted: [1, 3, 5, 7]
        // Q1 (25th): index = 0.25 * 3 = 0.75 => 1*(1-0.75) + 3*0.75 = 0.25 + 2.25 = 2.5
        // Q3 (75th): index = 0.75 * 3 = 2.25 => 5*(1-0.25) + 7*0.25 = 3.75 + 1.75 = 5.5
        // IQR = 5.5 - 2.5 = 3.0
        var data = MakeMatrix(new double[,] { { 1 }, { 3 }, { 5 }, { 7 } });

        var clipper = new IQRClipper<double>(multiplier: 1.5);
        clipper.Fit(data);

        Assert.NotNull(clipper.Q1Values);
        Assert.NotNull(clipper.Q3Values);
        Assert.NotNull(clipper.IQRValues);
        Assert.Equal(2.5, clipper.Q1Values[0], Tol);
        Assert.Equal(5.5, clipper.Q3Values[0], Tol);
        Assert.Equal(3.0, clipper.IQRValues[0], Tol);
    }

    [Fact]
    public void IQR_OutlierMask_DetectsOutliers()
    {
        // Bounds [-3, 13]
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var clipper = new IQRClipper<double>(multiplier: 1.5);
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -5 }, { 5 }, { 14 } });
        var mask = clipper.GetOutlierMask(test);

        Assert.True(mask[0, 0]);   // -5 < -3
        Assert.False(mask[1, 0]);  // 5 in [-3, 13]
        Assert.True(mask[2, 0]);   // 14 > 13
    }

    [Fact]
    public void IQR_CountOutliers_PerFeature()
    {
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var clipper = new IQRClipper<double>(multiplier: 1.5);
        clipper.Fit(data);

        // Bounds [-3, 13] => only -5 is an outlier
        var test = MakeMatrix(new double[,] { { -5 }, { 5 }, { 10 } });
        var counts = clipper.CountOutliersPerFeature(test);

        Assert.Equal(1, counts[0]); // Only -5 is outside
    }

    [Fact]
    public void IQR_InvalidMultiplier_Throws()
    {
        Assert.Throws<ArgumentException>(() => new IQRClipper<double>(multiplier: 0));
        Assert.Throws<ArgumentException>(() => new IQRClipper<double>(multiplier: -1));
    }

    // ========================================================================
    // MADClipper - Median Absolute Deviation
    // ========================================================================

    [Fact]
    public void MAD_FitComputes_Median_SingleColumn()
    {
        // Data: [1, 2, 3, 4, 5] (sorted)
        // Median = 3 (middle element of odd count)
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(data);

        Assert.NotNull(clipper.Medians);
        Assert.Equal(3.0, clipper.Medians[0], Tol);
    }

    [Fact]
    public void MAD_FitComputes_MAD_Correctly()
    {
        // Data: [1, 2, 3, 4, 5], Median = 3
        // Absolute deviations: |1-3|=2, |2-3|=1, |3-3|=0, |4-3|=1, |5-3|=2
        // Sorted: [0, 1, 1, 2, 2]
        // MAD = median([0,1,1,2,2]) = 1
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(data);

        Assert.NotNull(clipper.MADs);
        Assert.Equal(1.0, clipper.MADs[0], Tol);
    }

    [Fact]
    public void MAD_Bounds_HandComputed()
    {
        // Median=3, MAD=1, MADScaleFactor=1.4826, threshold=3.5
        // scaledMad = 1 * 1.4826 = 1.4826
        // Lower = 3 - 3.5 * 1.4826 = 3 - 5.1891 = -2.1891
        // Upper = 3 + 3.5 * 1.4826 = 3 + 5.1891 = 8.1891
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(data);

        Assert.NotNull(clipper.LowerBounds);
        Assert.NotNull(clipper.UpperBounds);
        Assert.Equal(3.0 - 3.5 * 1.4826, clipper.LowerBounds[0], Tol);
        Assert.Equal(3.0 + 3.5 * 1.4826, clipper.UpperBounds[0], Tol);
    }

    [Fact]
    public void MAD_ClipsOutliers_Correctly()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(data);

        double lower = 3.0 - 3.5 * 1.4826;
        double upper = 3.0 + 3.5 * 1.4826;

        var test = MakeMatrix(new double[,] { { -10 }, { 3 }, { 20 } });
        var result = clipper.Transform(test);

        Assert.Equal(lower, result[0, 0], Tol);
        Assert.Equal(3.0, result[1, 0], Tol);
        Assert.Equal(upper, result[2, 0], Tol);
    }

    [Fact]
    public void MAD_ModifiedZScores_HandComputed()
    {
        // Median=3, MAD=1
        // Modified Z-score = 0.6745 * (x - median) / MAD
        // For x=1: 0.6745 * (1-3) / 1 = 0.6745 * (-2) = -1.349
        // For x=3: 0.6745 * (3-3) / 1 = 0
        // For x=5: 0.6745 * (5-3) / 1 = 0.6745 * 2 = 1.349
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { 1 }, { 3 }, { 5 } });
        var zScores = clipper.GetModifiedZScores(test);

        Assert.Equal(-1.349, zScores[0, 0], 1e-3);
        Assert.Equal(0.0, zScores[1, 0], Tol);
        Assert.Equal(1.349, zScores[2, 0], 1e-3);
    }

    [Fact]
    public void MAD_EvenCount_Median_IsAverage()
    {
        // Data: [1, 2, 3, 4] => Median = (2+3)/2 = 2.5
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } });
        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(data);

        Assert.NotNull(clipper.Medians);
        Assert.Equal(2.5, clipper.Medians[0], Tol);
    }

    [Fact]
    public void MAD_ConstantColumn_NoBounds()
    {
        // All same => MAD=0 => bounds [MinValue, MaxValue]
        var data = MakeMatrix(new double[,] { { 5 }, { 5 }, { 5 }, { 5 } });
        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(data);

        Assert.NotNull(clipper.MADs);
        Assert.Equal(0.0, clipper.MADs[0], Tol);

        var test = MakeMatrix(new double[,] { { 100 }, { -100 } });
        var result = clipper.Transform(test);
        Assert.Equal(100.0, result[0, 0], Tol);
        Assert.Equal(-100.0, result[1, 0], Tol);
    }

    [Fact]
    public void MAD_InvalidThreshold_Throws()
    {
        Assert.Throws<ArgumentException>(() => new MADClipper<double>(threshold: 0));
        Assert.Throws<ArgumentException>(() => new MADClipper<double>(threshold: -1));
    }

    // ========================================================================
    // ThresholdClipper - Explicit Bounds
    // ========================================================================

    [Fact]
    public void Threshold_Symmetric_ClipsCorrectly()
    {
        // Symmetric threshold of 5 => bounds [-5, 5]
        var clipper = new ThresholdClipper<double>(threshold: 5.0);
        var data = MakeMatrix(new double[,] { { 1 } });
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -10 }, { 0 }, { 10 } });
        var result = clipper.Transform(test);

        Assert.Equal(-5.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[1, 0], Tol);
        Assert.Equal(5.0, result[2, 0], Tol);
    }

    [Fact]
    public void Threshold_Asymmetric_ClipsCorrectly()
    {
        // Lower=-2, Upper=8
        var clipper = new ThresholdClipper<double>(-2.0, 8.0);
        var data = MakeMatrix(new double[,] { { 1 } });
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -5 }, { 3 }, { 15 } });
        var result = clipper.Transform(test);

        Assert.Equal(-2.0, result[0, 0], Tol);
        Assert.Equal(3.0, result[1, 0], Tol);
        Assert.Equal(8.0, result[2, 0], Tol);
    }

    [Fact]
    public void Threshold_InBoundValues_Unchanged()
    {
        var clipper = new ThresholdClipper<double>(0.0, 100.0);
        var data = MakeMatrix(new double[,] { { 50 } });
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { 0 }, { 50 }, { 100 } });
        var result = clipper.Transform(test);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(50.0, result[1, 0], Tol);
        Assert.Equal(100.0, result[2, 0], Tol);
    }

    [Fact]
    public void Threshold_OutlierMask_HandComputed()
    {
        var clipper = new ThresholdClipper<double>(0.0, 100.0);
        var data = MakeMatrix(new double[,] { { 50 } });
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -1 }, { 50 }, { 101 } });
        var mask = clipper.GetOutlierMask(test);

        Assert.True(mask[0, 0]);   // -1 < 0
        Assert.False(mask[1, 0]);  // 50 in [0, 100]
        Assert.True(mask[2, 0]);   // 101 > 100
    }

    [Fact]
    public void Threshold_CountOutliers_HandComputed()
    {
        var clipper = new ThresholdClipper<double>(0.0, 100.0);
        var data = MakeMatrix(new double[,] { { 50 } });
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -5 }, { -2 }, { 50 }, { 105 } });
        var (belowLower, aboveUpper) = clipper.CountOutliers(test);

        Assert.Equal(2, belowLower[0]); // -5 and -2 below 0
        Assert.Equal(1, aboveUpper[0]); // 105 above 100
    }

    [Fact]
    public void Threshold_InvalidBounds_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ThresholdClipper<double>(10.0, 5.0));
    }

    // ========================================================================
    // Winsorizer - Percentile-Based Clipping
    // ========================================================================

    [Fact]
    public void Winsorizer_Percentile_Bounds_HandComputed()
    {
        // Data (sorted): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        // 10th percentile: index = 0.10 * 9 = 0.9 => 1*(1-0.9) + 2*0.9 = 0.1 + 1.8 = 1.9
        // 90th percentile: index = 0.90 * 9 = 8.1 => 9*(1-0.1) + 10*0.1 = 8.1 + 1.0 = 9.1
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });

        var winsorizer = new Winsorizer<double>(lowerLimit: 10, upperLimit: 90);
        winsorizer.Fit(data);

        Assert.NotNull(winsorizer.LowerBounds);
        Assert.NotNull(winsorizer.UpperBounds);
        Assert.Equal(1.9, winsorizer.LowerBounds[0], Tol);
        Assert.Equal(9.1, winsorizer.UpperBounds[0], Tol);
    }

    [Fact]
    public void Winsorizer_Percentile_ClipsCorrectly()
    {
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });

        var winsorizer = new Winsorizer<double>(lowerLimit: 10, upperLimit: 90);
        winsorizer.Fit(data);

        var test = MakeMatrix(new double[,] { { 0 }, { 5 }, { 15 } });
        var result = winsorizer.Transform(test);

        Assert.Equal(1.9, result[0, 0], Tol);  // Clipped to 10th percentile
        Assert.Equal(5.0, result[1, 0], Tol);   // Unchanged
        Assert.Equal(9.1, result[2, 0], Tol);  // Clipped to 90th percentile
    }

    [Fact]
    public void Winsorizer_5Percent_Bounds()
    {
        // Data [1..10] (10 elements)
        // 5th percentile: index = 0.05 * 9 = 0.45 => 1*(1-0.45) + 2*0.45 = 0.55 + 0.9 = 1.45
        // 95th percentile: index = 0.95 * 9 = 8.55 => 9*(1-0.55) + 10*0.55 = 4.05 + 5.5 = 9.55
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });

        var winsorizer = new Winsorizer<double>(lowerLimit: 5, upperLimit: 95);
        winsorizer.Fit(data);

        Assert.NotNull(winsorizer.LowerBounds);
        Assert.NotNull(winsorizer.UpperBounds);
        Assert.Equal(1.45, winsorizer.LowerBounds[0], Tol);
        Assert.Equal(9.55, winsorizer.UpperBounds[0], Tol);
    }

    [Fact]
    public void Winsorizer_IQRMode_Bounds()
    {
        // Data [1..9] sorted: [1,2,3,4,5,6,7,8,9]
        // Q1 = 3, Q3 = 7, IQR = 4
        // lowerLimit=1.5 (multiplier), upperLimit=1.5
        // Lower = 3 - 1.5*4 = -3
        // Upper = 7 + 1.5*4 = 13
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var winsorizer = new Winsorizer<double>(
            lowerLimit: 1.5, upperLimit: 1.5,
            limitType: WinsorizerLimitType.IQR);
        winsorizer.Fit(data);

        Assert.NotNull(winsorizer.LowerBounds);
        Assert.NotNull(winsorizer.UpperBounds);
        Assert.Equal(-3.0, winsorizer.LowerBounds[0], Tol);
        Assert.Equal(13.0, winsorizer.UpperBounds[0], Tol);
    }

    [Fact]
    public void Winsorizer_InvalidPercentile_Throws()
    {
        // Lower must be 0-50 for percentile mode
        Assert.Throws<ArgumentException>(() => new Winsorizer<double>(lowerLimit: -1));
        Assert.Throws<ArgumentException>(() => new Winsorizer<double>(lowerLimit: 51));
        Assert.Throws<ArgumentException>(() => new Winsorizer<double>(upperLimit: 49));
        Assert.Throws<ArgumentException>(() => new Winsorizer<double>(upperLimit: 101));
    }

    // ========================================================================
    // Cross-Clipper Consistency Tests
    // ========================================================================

    [Fact]
    public void AllClippers_PreserveInBoundValues()
    {
        // All clippers should preserve values within their bounds
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var zScore = new ZScoreClipper<double>(threshold: 100);
        var iqr = new IQRClipper<double>(multiplier: 100);
        var mad = new MADClipper<double>(threshold: 100);

        zScore.Fit(data);
        iqr.Fit(data);
        mad.Fit(data);

        // With very large thresholds, all values should be preserved
        var test = MakeMatrix(new double[,] { { 3 }, { 5 }, { 7 } });

        var zResult = zScore.Transform(test);
        var iqrResult = iqr.Transform(test);
        var madResult = mad.Transform(test);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(test[i, 0], zResult[i, 0], Tol);
            Assert.Equal(test[i, 0], iqrResult[i, 0], Tol);
            Assert.Equal(test[i, 0], madResult[i, 0], Tol);
        }
    }

    [Fact]
    public void ZScore_Idempotent_ClippedValueStaysClipped()
    {
        // After clipping, re-clipping should produce the same result
        var data = MakeMatrix(new double[,] {
            { 2 }, { 4 }, { 4 }, { 4 }, { 5 }, { 5 }, { 7 }, { 9 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 2.0);
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -100 }, { 100 } });
        var firstClip = clipper.Transform(test);
        var secondClip = clipper.Transform(firstClip);

        Assert.Equal(firstClip[0, 0], secondClip[0, 0], Tol);
        Assert.Equal(firstClip[1, 0], secondClip[1, 0], Tol);
    }

    [Fact]
    public void IQR_Idempotent_ClippedValueStaysClipped()
    {
        var data = MakeMatrix(new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });

        var clipper = new IQRClipper<double>(multiplier: 1.5);
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -100 }, { 100 } });
        var firstClip = clipper.Transform(test);
        var secondClip = clipper.Transform(firstClip);

        Assert.Equal(firstClip[0, 0], secondClip[0, 0], Tol);
        Assert.Equal(firstClip[1, 0], secondClip[1, 0], Tol);
    }

    [Fact]
    public void ZScore_SpecificColumns_OnlyClipsSelected()
    {
        // Only clip column 0, leave column 1 alone
        var data = MakeMatrix(new double[,] {
            { 2, 100 }, { 4, 200 }, { 4, 300 }, { 4, 400 },
            { 5, 500 }, { 5, 600 }, { 7, 700 }, { 9, 800 }
        });

        var clipper = new ZScoreClipper<double>(threshold: 2.0, columnIndices: new[] { 0 });
        clipper.Fit(data);

        // Column 0: Mean=5, Std=2, Bounds=[1,9]
        // Column 1: not processed
        var test = MakeMatrix(new double[,] { { 0, 10000 } });
        var result = clipper.Transform(test);

        Assert.Equal(1.0, result[0, 0], Tol);     // Column 0 clipped
        Assert.Equal(10000.0, result[0, 1], Tol);  // Column 1 unchanged
    }

    [Fact]
    public void ThresholdClipper_MultipleColumns_AllClipped()
    {
        var clipper = new ThresholdClipper<double>(0.0, 100.0);
        var data = MakeMatrix(new double[,] { { 50, 50 } });
        clipper.Fit(data);

        var test = MakeMatrix(new double[,] { { -5, 105 } });
        var result = clipper.Transform(test);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(100.0, result[0, 1], Tol);
    }

    [Fact]
    public void ZScore_PopulationStd_NotSampleStd()
    {
        // The code uses population std (divides by N, not N-1)
        // Data: [1, 5] => Mean = 3
        // Population std = sqrt(((1-3)^2 + (5-3)^2)/2) = sqrt((4+4)/2) = sqrt(4) = 2
        // Sample std would be sqrt((4+4)/1) = sqrt(8) ~ 2.828
        var data = MakeMatrix(new double[,] { { 1 }, { 5 } });
        var clipper = new ZScoreClipper<double>(threshold: 3.0);
        clipper.Fit(data);

        Assert.NotNull(clipper.StandardDeviations);
        Assert.Equal(2.0, clipper.StandardDeviations[0], Tol); // Population std
    }

    [Fact]
    public void MAD_ScaleFactor_IsCorrect()
    {
        // MADScaleFactor = 1.4826 (constant in code)
        // This is 1/Phi^{-1}(3/4) where Phi^{-1} is the quantile function of standard normal
        // The bounds should be: median +/- threshold * MAD * 1.4826
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var clipper = new MADClipper<double>(threshold: 2.5);
        clipper.Fit(data);

        // Median=3, MAD=1
        // scaledMad = 1 * 1.4826
        // Lower = 3 - 2.5 * 1.4826 = 3 - 3.7065 = -0.7065
        // Upper = 3 + 2.5 * 1.4826 = 3 + 3.7065 = 6.7065
        Assert.NotNull(clipper.LowerBounds);
        Assert.NotNull(clipper.UpperBounds);
        Assert.Equal(3.0 - 2.5 * 1.4826, clipper.LowerBounds[0], Tol);
        Assert.Equal(3.0 + 2.5 * 1.4826, clipper.UpperBounds[0], Tol);
    }

    [Fact]
    public void MAD_OutlierMask_DetectsOutliers()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(data);

        double lower = 3.0 - 3.5 * 1.4826;
        double upper = 3.0 + 3.5 * 1.4826;

        // Create test with values inside and outside bounds
        var test = MakeMatrix(new double[,] { { lower - 1 }, { 3 }, { upper + 1 } });
        var mask = clipper.GetOutlierMask(test);

        Assert.True(mask[0, 0]);   // Below lower bound
        Assert.False(mask[1, 0]);  // In bounds
        Assert.True(mask[2, 0]);   // Above upper bound
    }
}
