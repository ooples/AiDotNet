using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep math-correctness integration tests for advanced categorical encoders:
/// TargetEncoder, WOEEncoder, HelmertEncoder, JamesSteinEncoder,
/// LeaveOneOutEncoder, MEstimateEncoder, BackwardDifferenceEncoder.
/// Each test hand-computes expected values and verifies code matches.
/// </summary>
public class AdvancedEncodersDeepMathIntegrationTests
{
    private const double Tol = 1e-8;

    private static Matrix<double> MakeMatrix(double[,] data) => new(data);
    private static Vector<double> MakeVector(double[] data) => new(data);

    // ========================================================================
    // TargetEncoder - Smoothed Target Mean Encoding
    // ========================================================================

    [Fact]
    public void TargetEncoder_SmoothedMean_HandComputedFormula()
    {
        // Data: category column with values [1, 1, 1, 2, 2]
        // Target: [10, 20, 30, 40, 50]
        // Global mean = (10+20+30+40+50)/5 = 30
        // Category 1: count=3, mean=(10+20+30)/3=20, smoothed = (3*20 + 1*30)/(3+1) = 90/4 = 22.5
        // Category 2: count=2, mean=(40+50)/2=45, smoothed = (2*45 + 1*30)/(2+1) = 120/3 = 40.0
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 40, 50 });

        var encoder = new TargetEncoder<double>(smoothing: 1.0);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.Equal(22.5, result[0, 0], Tol);
        Assert.Equal(22.5, result[1, 0], Tol);
        Assert.Equal(22.5, result[2, 0], Tol);
        Assert.Equal(40.0, result[3, 0], Tol);
        Assert.Equal(40.0, result[4, 0], Tol);
    }

    [Fact]
    public void TargetEncoder_HighSmoothing_PullsTowardGlobalMean()
    {
        // Data: [1, 1, 2, 2]
        // Target: [10, 20, 80, 90]
        // Global mean = (10+20+80+90)/4 = 50
        // smoothing = 100
        // Cat 1: count=2, mean=15, smoothed = (2*15 + 100*50)/(2+100) = (30+5000)/102 = 49.31372549...
        // Cat 2: count=2, mean=85, smoothed = (2*85 + 100*50)/(2+100) = (170+5000)/102 = 50.68627451...
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 80, 90 });

        var encoder = new TargetEncoder<double>(smoothing: 100.0);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double expectedCat1 = (2 * 15.0 + 100 * 50.0) / (2 + 100);
        double expectedCat2 = (2 * 85.0 + 100 * 50.0) / (2 + 100);

        Assert.Equal(expectedCat1, result[0, 0], Tol);
        Assert.Equal(expectedCat2, result[2, 0], Tol);
        // High smoothing should pull both toward 50
        Assert.True(Math.Abs(result[0, 0] - 50) < 2.0);
        Assert.True(Math.Abs(result[2, 0] - 50) < 2.0);
    }

    [Fact]
    public void TargetEncoder_ZeroSmoothing_UsesRawCategoryMean()
    {
        // smoothing = 0 means no shrinkage
        // Cat 1: (2*15 + 0*50)/(2+0) = 15
        // Cat 2: (2*85 + 0*50)/(2+0) = 85
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 80, 90 });

        var encoder = new TargetEncoder<double>(smoothing: 0.0);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.Equal(15.0, result[0, 0], Tol); // (10+20)/2
        Assert.Equal(85.0, result[2, 0], Tol); // (80+90)/2
    }

    [Fact]
    public void TargetEncoder_MinSamplesLeaf_RareCategoryGetsGlobalMean()
    {
        // Data: [1, 1, 1, 1, 2] - Category 2 has only 1 sample
        // Target: [10, 20, 30, 40, 100]
        // Global mean = (10+20+30+40+100)/5 = 40
        // minSamplesLeaf = 2
        // Cat 1: count=4 >= 2, mean=25, smoothed = (4*25 + 1*40)/(4+1) = 140/5 = 28.0
        // Cat 2: count=1 < 2, => global mean = 40
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 1 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 40, 100 });

        var encoder = new TargetEncoder<double>(smoothing: 1.0, minSamplesLeaf: 2);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.Equal(28.0, result[0, 0], Tol);
        Assert.Equal(40.0, result[4, 0], Tol); // rare category -> global mean
    }

    [Fact]
    public void TargetEncoder_UnknownCategory_UsesGlobalMean()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 80, 90 });

        var encoder = new TargetEncoder<double>(smoothing: 1.0, handleUnknown: TargetEncoderHandleUnknown.UseGlobalMean);
        encoder.Fit(data, target);

        // Transform data with unknown category 99
        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        // Global mean = 50
        Assert.Equal(50.0, result[0, 0], Tol);
    }

    [Fact]
    public void TargetEncoder_UnknownCategory_ErrorMode_Throws()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20 });

        var encoder = new TargetEncoder<double>(handleUnknown: TargetEncoderHandleUnknown.Error);
        encoder.Fit(data, target);

        var testData = MakeMatrix(new double[,] { { 99 } });
        Assert.Throws<ArgumentException>(() => encoder.Transform(testData));
    }

    [Fact]
    public void TargetEncoder_MultiColumn_IndependentEncoding()
    {
        // Two columns with different categories
        var data = MakeMatrix(new double[,] { { 1, 10 }, { 1, 20 }, { 2, 10 }, { 2, 20 } });
        var target = MakeVector(new double[] { 100, 200, 300, 400 });
        // Global mean = (100+200+300+400)/4 = 250

        var encoder = new TargetEncoder<double>(smoothing: 0.0);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        // Col 0: Cat1 mean = (100+200)/2 = 150, Cat2 mean = (300+400)/2 = 350
        Assert.Equal(150.0, result[0, 0], Tol);
        Assert.Equal(350.0, result[2, 0], Tol);

        // Col 1: Cat10 mean = (100+300)/2 = 200, Cat20 mean = (200+400)/2 = 300
        Assert.Equal(200.0, result[0, 1], Tol);
        Assert.Equal(300.0, result[1, 1], Tol);
    }

    [Fact]
    public void TargetEncoder_NegativeSmoothing_Throws()
    {
        Assert.Throws<ArgumentException>(() => new TargetEncoder<double>(smoothing: -1.0));
    }

    // ========================================================================
    // WOEEncoder - Weight of Evidence
    // ========================================================================

    [Fact]
    public void WOEEncoder_HandComputed_TwoCategories()
    {
        // Data: [1, 1, 1, 2, 2, 2]
        // Target: [1, 1, 0, 0, 0, 1]
        // totalEvents = 3 (targets = 1), totalNonEvents = 3 (targets = 0)
        // nCategories = 2, regularization = 0.5
        //
        // Cat 1: events=2, nonEvents=1
        //   distEvents = (2 + 0.5) / (3 + 0.5*2) = 2.5 / 4.0 = 0.625
        //   distNonEvents = (1 + 0.5) / (3 + 0.5*2) = 1.5 / 4.0 = 0.375
        //   WOE = ln(0.625 / 0.375) = ln(1.66666...) = 0.510825623...
        //
        // Cat 2: events=1, nonEvents=2
        //   distEvents = (1 + 0.5) / (3 + 0.5*2) = 1.5 / 4.0 = 0.375
        //   distNonEvents = (2 + 0.5) / (3 + 0.5*2) = 2.5 / 4.0 = 0.625
        //   WOE = ln(0.375 / 0.625) = ln(0.6) = -0.510825623...
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 1, 1, 0, 0, 0, 1 });

        var encoder = new WOEEncoder<double>(regularization: 0.5);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double expectedWoeCat1 = Math.Log(0.625 / 0.375);
        double expectedWoeCat2 = Math.Log(0.375 / 0.625);

        Assert.Equal(expectedWoeCat1, result[0, 0], Tol);
        Assert.Equal(expectedWoeCat1, result[1, 0], Tol);
        Assert.Equal(expectedWoeCat2, result[3, 0], Tol);
    }

    [Fact]
    public void WOEEncoder_PositiveWOE_MoreEventsInCategory()
    {
        // Category with more events than non-events should have positive WOE
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 1, 1, 1, 0, 0, 1 });
        // Cat 1: 3 events, 1 non-event -> positive WOE
        // Cat 2: 1 event, 1 non-event -> near-zero WOE

        var encoder = new WOEEncoder<double>(regularization: 0.5);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.True(result[0, 0] > 0, "Category with more events should have positive WOE");
    }

    [Fact]
    public void WOEEncoder_NegativeWOE_MoreNonEventsInCategory()
    {
        // Category with more non-events than events should have negative WOE
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 0, 0, 0, 1, 1, 0 });
        // Cat 1: 1 event, 3 non-events -> negative WOE

        var encoder = new WOEEncoder<double>(regularization: 0.5);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.True(result[0, 0] < 0, "Category with more non-events should have negative WOE");
    }

    [Fact]
    public void WOEEncoder_Clamping_ExtremeValuesClampedToFive()
    {
        // Create extreme imbalance: one category has almost all events
        // With low regularization, WOE could be extreme
        // Category 1: 99 events, 1 non-event. Category 2: 1 event, 99 non-events.
        int n = 200;
        var dataArr = new double[n, 1];
        var targetArr = new double[n];
        for (int i = 0; i < 100; i++)
        {
            dataArr[i, 0] = 1;
            targetArr[i] = i < 99 ? 1 : 0;
        }
        for (int i = 100; i < 200; i++)
        {
            dataArr[i, 0] = 2;
            targetArr[i] = i < 101 ? 1 : 0;
        }
        var data = MakeMatrix(dataArr);
        var target = MakeVector(targetArr);

        var encoder = new WOEEncoder<double>(regularization: 0.001);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        // WOE values should be clamped to [-5, 5]
        Assert.True(result[0, 0] <= 5.0 + Tol);
        Assert.True(result[0, 0] >= -5.0 - Tol);
        Assert.True(result[100, 0] <= 5.0 + Tol);
        Assert.True(result[100, 0] >= -5.0 - Tol);
    }

    [Fact]
    public void WOEEncoder_InformationValue_HandComputed()
    {
        // Using same data as HandComputed test
        // Cat 1: distEvents=0.625, distNonEvents=0.375, WOE=ln(0.625/0.375)
        //   IV_contribution = (0.625-0.375) * ln(0.625/0.375)
        // Cat 2: distEvents=0.375, distNonEvents=0.625, WOE=ln(0.375/0.625)
        //   IV_contribution = (0.375-0.625) * ln(0.375/0.625)
        // Both contributions are positive (diff*woe: positive*positive and negative*negative)
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 1, 1, 0, 0, 0, 1 });

        var encoder = new WOEEncoder<double>(regularization: 0.5);
        encoder.Fit(data, target);
        var ivValues = encoder.CalculateInformationValue(data, target);

        double woeCat1 = Math.Log(0.625 / 0.375);
        double woeCat2 = Math.Log(0.375 / 0.625);
        double expectedIV = (0.625 - 0.375) * woeCat1 + (0.375 - 0.625) * woeCat2;

        Assert.True(ivValues.ContainsKey(0));
        Assert.Equal(expectedIV, ivValues[0], Tol);
        Assert.True(ivValues[0] >= 0, "IV should be non-negative");
    }

    [Fact]
    public void WOEEncoder_BinaryTargetRequired_Throws()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });
        var target = MakeVector(new double[] { 0, 2 }); // non-binary

        var encoder = new WOEEncoder<double>();
        Assert.Throws<ArgumentException>(() => encoder.Fit(data, target));
    }

    [Fact]
    public void WOEEncoder_SingleClassTarget_Throws()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });
        var target = MakeVector(new double[] { 1, 1 }); // all events, no non-events

        var encoder = new WOEEncoder<double>();
        Assert.Throws<ArgumentException>(() => encoder.Fit(data, target));
    }

    [Fact]
    public void WOEEncoder_UnknownCategory_UseZero()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 1, 0, 0, 1 });

        var encoder = new WOEEncoder<double>(handleUnknown: WOEHandleUnknown.UseZero);
        encoder.Fit(data, target);

        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        Assert.Equal(0.0, result[0, 0], Tol); // Unknown -> 0 (neutral evidence)
    }

    [Fact]
    public void WOEEncoder_Symmetry_SwappedClassesNegateWOE()
    {
        // If we swap events and non-events for a category, WOE should negate
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target1 = MakeVector(new double[] { 1, 0, 0, 1 });
        var target2 = MakeVector(new double[] { 0, 1, 1, 0 }); // swapped

        var encoder1 = new WOEEncoder<double>(regularization: 0.5);
        encoder1.Fit(data, target1);
        var result1 = encoder1.Transform(data);

        var encoder2 = new WOEEncoder<double>(regularization: 0.5);
        encoder2.Fit(data, target2);
        var result2 = encoder2.Transform(data);

        // Swapping target classes should negate WOE for each category
        Assert.Equal(result1[0, 0], -result2[0, 0], Tol);
        Assert.Equal(result1[2, 0], -result2[2, 0], Tol);
    }

    // ========================================================================
    // HelmertEncoder - Contrast Coding
    // ========================================================================

    [Fact]
    public void HelmertEncoder_ThreeCategories_ContrastMatrixValues()
    {
        // 3 categories (10, 20, 30), creates 2 contrast columns
        // Standard Helmert matrix for k=3:
        // Col 0: compare level 0 to mean of levels 1,2
        //   nSubsequent = 2
        //   row 0: 2/3, row 1: -1/3, row 2: -1/3
        // Col 1: compare level 1 to mean of level 2
        //   nSubsequent = 1
        //   row 0: 0, row 1: 1/2, row 2: -1/2
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });

        var encoder = new HelmertEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Row 0 (cat=10): [2/3, 0]
        Assert.Equal(2.0 / 3, result[0, 0], Tol);
        Assert.Equal(0.0, result[0, 1], Tol);

        // Row 1 (cat=20): [-1/3, 1/2]
        Assert.Equal(-1.0 / 3, result[1, 0], Tol);
        Assert.Equal(1.0 / 2, result[1, 1], Tol);

        // Row 2 (cat=30): [-1/3, -1/2]
        Assert.Equal(-1.0 / 3, result[2, 0], Tol);
        Assert.Equal(-1.0 / 2, result[2, 1], Tol);
    }

    [Fact]
    public void HelmertEncoder_ReversedMode_ComparesWithPreviousLevels()
    {
        // Reversed Helmert for k=3:
        // Col 0: compare level 1 to mean of level 0
        //   nPrevious = 1
        //   row 0: -1/2, row 1: 1/2, row 2: 0
        // Col 1: compare level 2 to mean of levels 0,1
        //   nPrevious = 2
        //   row 0: -1/3, row 1: -1/3, row 2: 2/3
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });

        var encoder = new HelmertEncoder<double>(reversed: true);
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Row 0 (cat=10): [-1/2, -1/3]
        Assert.Equal(-1.0 / 2, result[0, 0], Tol);
        Assert.Equal(-1.0 / 3, result[0, 1], Tol);

        // Row 1 (cat=20): [1/2, -1/3]
        Assert.Equal(1.0 / 2, result[1, 0], Tol);
        Assert.Equal(-1.0 / 3, result[1, 1], Tol);

        // Row 2 (cat=30): [0, 2/3]
        Assert.Equal(0.0, result[2, 0], Tol);
        Assert.Equal(2.0 / 3, result[2, 1], Tol);
    }

    [Fact]
    public void HelmertEncoder_OutputDimension_KMinusOneColumns()
    {
        // 4 categories -> 3 contrast columns
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } });

        var encoder = new HelmertEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(4, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void HelmertEncoder_ColumnSumsToZero()
    {
        // Each column in the Helmert contrast matrix should sum to zero
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } });

        var encoder = new HelmertEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int col = 0; col < result.Columns; col++)
        {
            double colSum = 0;
            for (int row = 0; row < result.Rows; row++)
            {
                colSum += result[row, col];
            }
            Assert.Equal(0.0, colSum, 1e-6);
        }
    }

    [Fact]
    public void HelmertEncoder_UnknownCategory_ZeroVector()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        var encoder = new HelmertEncoder<double>(handleUnknown: HelmertHandleUnknown.UseZeros);
        encoder.Fit(data);

        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[0, 1], Tol);
    }

    // ========================================================================
    // JamesSteinEncoder - Bayesian Shrinkage
    // ========================================================================

    [Fact]
    public void JamesSteinEncoder_LargeCountLowShrinkage_NearCategoryMean()
    {
        // Large count category far from global mean -> low shrinkage -> close to category mean
        // Category 1: count=100, all targets=100 -> mean=100
        // Category 2: count=100, all targets=0 -> mean=0
        // Global mean = 50
        // globalVariance = high (many 0s and 100s)
        //
        // For cat 1: diffFromGlobal = 100-50 = 50
        //   shrinkage = max(0, 1 - (100-2)*var / (100 * 50^2))
        //   With var = sum((x-50)^2)/199, each sample contributes 2500, var ~ 2500
        //   shrinkage = max(0, 1 - 98*2500 / (100*2500)) = max(0, 1 - 0.98) = 0.02
        //   encoded ~ (1-0.02)*100 + 0.02*50 = 99
        int n = 200;
        var dataArr = new double[n, 1];
        var targetArr = new double[n];
        for (int i = 0; i < 100; i++)
        {
            dataArr[i, 0] = 1;
            targetArr[i] = 100;
        }
        for (int i = 100; i < 200; i++)
        {
            dataArr[i, 0] = 2;
            targetArr[i] = 0;
        }
        var data = MakeMatrix(dataArr);
        var target = MakeVector(targetArr);

        var encoder = new JamesSteinEncoder<double>();
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        // Cat 1 should be close to 100 (its own mean), not shrunk much toward 50
        Assert.True(result[0, 0] > 90, $"Cat 1 encoded {result[0, 0]} should be close to its mean 100");
        // Cat 2 should be close to 0 (its own mean)
        Assert.True(result[100, 0] < 10, $"Cat 2 encoded {result[100, 0]} should be close to its mean 0");
    }

    [Fact]
    public void JamesSteinEncoder_SmallCount_FullShrinkageToGlobalMean()
    {
        // Category with count <= 2 gets full shrinkage (shrinkage=1.0)
        // Encoded value = (1-1)*categoryMean + 1*globalMean = globalMean
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 2 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 999, 10, 20, 30, 40 });
        // Global mean = (999+10+20+30+40)/5 = 219.8
        // Cat 1: count=1 <= 2, shrinkage=1.0, encoded = globalMean = 219.8

        var encoder = new JamesSteinEncoder<double>();
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double globalMean = (999.0 + 10 + 20 + 30 + 40) / 5.0;
        Assert.Equal(globalMean, result[0, 0], Tol);
    }

    [Fact]
    public void JamesSteinEncoder_GlobalMean_IsCorrect()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 40 });

        var encoder = new JamesSteinEncoder<double>();
        encoder.Fit(data, target);

        Assert.Equal(25.0, encoder.GlobalMean, Tol);
    }

    [Fact]
    public void JamesSteinEncoder_UnknownCategory_UsesGlobalMean()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 40 });

        var encoder = new JamesSteinEncoder<double>(handleUnknown: JamesSteinHandleUnknown.UseGlobalMean);
        encoder.Fit(data, target);

        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        Assert.Equal(25.0, result[0, 0], Tol);
    }

    // ========================================================================
    // LeaveOneOutEncoder - LOO Target Encoding
    // ========================================================================

    [Fact]
    public void LeaveOneOutEncoder_HandComputed_ThreeSamples()
    {
        // Category "A" (value 1) has 3 samples with targets [10, 20, 30]
        // Category "B" (value 2) has 2 samples with targets [40, 50]
        // Global mean = (10+20+30+40+50)/5 = 30
        // smoothing = 0
        //
        // LOO for row 0 (cat=1, target=10): looSum = 20+30=50, looCount=2, looMean=25
        //   encoded = (2*25 + 0*30)/(2+0) = 25
        // LOO for row 1 (cat=1, target=20): looSum = 10+30=40, looCount=2, looMean=20
        //   encoded = (2*20 + 0*30)/(2+0) = 20
        // LOO for row 2 (cat=1, target=30): looSum = 10+20=30, looCount=2, looMean=15
        //   encoded = (2*15 + 0*30)/(2+0) = 15
        // LOO for row 3 (cat=2, target=40): looSum = 50, looCount=1, looMean=50
        //   encoded = (1*50 + 0*30)/(1+0) = 50
        // LOO for row 4 (cat=2, target=50): looSum = 40, looCount=1, looMean=40
        //   encoded = (1*40 + 0*30)/(1+0) = 40
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 40, 50 });

        var encoder = new LeaveOneOutEncoder<double>(smoothing: 0);
        encoder.Fit(data, target);
        var result = encoder.TransformWithTarget(data, target);

        Assert.Equal(25.0, result[0, 0], Tol);
        Assert.Equal(20.0, result[1, 0], Tol);
        Assert.Equal(15.0, result[2, 0], Tol);
        Assert.Equal(50.0, result[3, 0], Tol);
        Assert.Equal(40.0, result[4, 0], Tol);
    }

    [Fact]
    public void LeaveOneOutEncoder_WithSmoothing_HandComputed()
    {
        // Same data as above but with smoothing=1
        // LOO for row 0 (cat=1, target=10): looSum=50, looCount=2, looMean=25
        //   encoded = (2*25 + 1*30)/(2+1) = (50+30)/3 = 80/3 = 26.666...
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 40, 50 });

        var encoder = new LeaveOneOutEncoder<double>(smoothing: 1.0);
        encoder.Fit(data, target);
        var result = encoder.TransformWithTarget(data, target);

        double expected = (2 * 25.0 + 1 * 30.0) / (2 + 1);
        Assert.Equal(expected, result[0, 0], Tol);
    }

    [Fact]
    public void LeaveOneOutEncoder_SingleSampleCategory_FallsToGlobalMean()
    {
        // Category with only 1 sample: LOO count = 0, should use global mean
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 999, 10, 20, 30 });
        // Global mean = (999+10+20+30)/4 = 264.75

        var encoder = new LeaveOneOutEncoder<double>(smoothing: 0);
        encoder.Fit(data, target);
        var result = encoder.TransformWithTarget(data, target);

        // Cat 1 has only 1 sample, LOO count=0, falls to global mean
        double globalMean = (999.0 + 10 + 20 + 30) / 4.0;
        Assert.Equal(globalMean, result[0, 0], Tol);
    }

    [Fact]
    public void LeaveOneOutEncoder_TransformForTestData_UsesFullStats()
    {
        // Transform (without target) for test data uses full category statistics
        // Cat 1: sum=60, count=3, mean=20, smoothed=(3*20+1*30)/(3+1) = 90/4 = 22.5
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 40, 50 });

        var encoder = new LeaveOneOutEncoder<double>(smoothing: 1.0);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double expectedCat1 = (3 * 20.0 + 1 * 30.0) / (3 + 1);
        Assert.Equal(expectedCat1, result[0, 0], Tol);
    }

    [Fact]
    public void LeaveOneOutEncoder_EachRowGetsDifferentEncoding()
    {
        // Unlike TargetEncoder, LOO gives different encodings per row even for same category
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 } });
        var target = MakeVector(new double[] { 10, 20, 30 });

        var encoder = new LeaveOneOutEncoder<double>(smoothing: 0);
        encoder.Fit(data, target);
        var result = encoder.TransformWithTarget(data, target);

        // Row 0 LOO: (20+30)/2 = 25
        // Row 1 LOO: (10+30)/2 = 20
        // Row 2 LOO: (10+20)/2 = 15
        Assert.Equal(25.0, result[0, 0], Tol);
        Assert.Equal(20.0, result[1, 0], Tol);
        Assert.Equal(15.0, result[2, 0], Tol);
        // All different!
        Assert.NotEqual(result[0, 0], result[1, 0]);
        Assert.NotEqual(result[1, 0], result[2, 0]);
    }

    // ========================================================================
    // MEstimateEncoder - Simplified Target Encoding with M-Parameter
    // ========================================================================

    [Fact]
    public void MEstimateEncoder_HandComputed_Formula()
    {
        // Formula: (sum + m * globalMean) / (count + m)
        // Data: [1, 1, 1, 2, 2]
        // Target: [10, 20, 30, 80, 90]
        // Global mean = (10+20+30+80+90)/5 = 46
        // m = 2
        //
        // Cat 1: sum=60, count=3
        //   encoded = (60 + 2*46)/(3+2) = (60+92)/5 = 152/5 = 30.4
        // Cat 2: sum=170, count=2
        //   encoded = (170 + 2*46)/(2+2) = (170+92)/4 = 262/4 = 65.5
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 80, 90 });

        var encoder = new MEstimateEncoder<double>(m: 2.0);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.Equal(30.4, result[0, 0], Tol);
        Assert.Equal(30.4, result[1, 0], Tol);
        Assert.Equal(30.4, result[2, 0], Tol);
        Assert.Equal(65.5, result[3, 0], Tol);
        Assert.Equal(65.5, result[4, 0], Tol);
    }

    [Fact]
    public void MEstimateEncoder_MZero_EqualsRawCategoryMean()
    {
        // m=0: encoded = (sum + 0)/(count + 0) = sum/count = categoryMean
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 30, 70, 90 });

        var encoder = new MEstimateEncoder<double>(m: 0.0);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.Equal(20.0, result[0, 0], Tol); // (10+30)/2
        Assert.Equal(80.0, result[2, 0], Tol); // (70+90)/2
    }

    [Fact]
    public void MEstimateEncoder_VeryLargeM_ApproachesGlobalMean()
    {
        // m=10000: encoded ~ globalMean for all categories
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 0, 0, 100, 100 });
        // Global mean = 50

        var encoder = new MEstimateEncoder<double>(m: 10000.0);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.True(Math.Abs(result[0, 0] - 50.0) < 0.1);
        Assert.True(Math.Abs(result[2, 0] - 50.0) < 0.1);
    }

    [Fact]
    public void MEstimateEncoder_EquivalenceWithTargetEncoder()
    {
        // MEstimate with m=s and TargetEncoder with smoothing=s should produce same results
        // MEstimate: (sum + m*global) / (count + m) = (count*mean + m*global) / (count + m)
        // TargetEncoder: (count*mean + smoothing*global) / (count + smoothing)
        // These are identical!
        var data = MakeMatrix(new double[,] { { 1 }, { 1 }, { 1 }, { 2 }, { 2 } });
        var target = MakeVector(new double[] { 10, 20, 30, 40, 50 });

        var mEncoder = new MEstimateEncoder<double>(m: 3.0);
        mEncoder.Fit(data, target);
        var mResult = mEncoder.Transform(data);

        var tEncoder = new TargetEncoder<double>(smoothing: 3.0);
        tEncoder.Fit(data, target);
        var tResult = tEncoder.Transform(data);

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(tResult[i, 0], mResult[i, 0], Tol);
        }
    }

    [Fact]
    public void MEstimateEncoder_UnknownCategory_UsesGlobalMean()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });
        var target = MakeVector(new double[] { 10, 30 });

        var encoder = new MEstimateEncoder<double>(m: 1.0);
        encoder.Fit(data, target);

        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        Assert.Equal(20.0, result[0, 0], Tol); // global mean
    }

    [Fact]
    public void MEstimateEncoder_NegativeM_Throws()
    {
        Assert.Throws<ArgumentException>(() => new MEstimateEncoder<double>(m: -1.0));
    }

    // ========================================================================
    // BackwardDifferenceEncoder - Contrast Coding
    // ========================================================================

    [Fact]
    public void BackwardDifferenceEncoder_ThreeCategories_ContrastValues()
    {
        // k=3 categories (1, 2, 3), creates 2 contrast columns
        // Backward difference matrix for k=3:
        // Col 0: compares level 1 vs level 0 (encoded as differences from cumulative mean)
        //   row <= 0: -(3-0-1)/3 = -2/3
        //   row > 0: (0+1)/3 = 1/3
        // Col 1: compares level 2 vs level 1
        //   row <= 1: -(3-1-1)/3 = -1/3
        //   row > 1: (1+1)/3 = 2/3
        //
        // Matrix:
        //        Col0    Col1
        // Cat1: -2/3    -1/3
        // Cat2:  1/3    -1/3
        // Cat3:  1/3     2/3
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        var encoder = new BackwardDifferenceEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // Row 0 (cat=1): [-2/3, -1/3]
        Assert.Equal(-2.0 / 3, result[0, 0], Tol);
        Assert.Equal(-1.0 / 3, result[0, 1], Tol);

        // Row 1 (cat=2): [1/3, -1/3]
        Assert.Equal(1.0 / 3, result[1, 0], Tol);
        Assert.Equal(-1.0 / 3, result[1, 1], Tol);

        // Row 2 (cat=3): [1/3, 2/3]
        Assert.Equal(1.0 / 3, result[2, 0], Tol);
        Assert.Equal(2.0 / 3, result[2, 1], Tol);
    }

    [Fact]
    public void BackwardDifferenceEncoder_ColumnSumsToZero()
    {
        // Each column should sum to zero (balanced contrast)
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } });

        var encoder = new BackwardDifferenceEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        for (int col = 0; col < result.Columns; col++)
        {
            double colSum = 0;
            for (int row = 0; row < result.Rows; row++)
            {
                colSum += result[row, col];
            }
            Assert.Equal(0.0, colSum, 1e-6);
        }
    }

    [Fact]
    public void BackwardDifferenceEncoder_OutputDimension_KMinusOne()
    {
        // 5 categories -> 4 contrast columns
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        var encoder = new BackwardDifferenceEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        Assert.Equal(5, result.Rows);
        Assert.Equal(4, result.Columns);
    }

    [Fact]
    public void BackwardDifferenceEncoder_AdjacentDifference_IsOneOverK()
    {
        // The difference between adjacent rows in each column should be predictable
        // For k=4, col c: row c gets -(k-c-1)/k, row c+1 gets (c+1)/k
        // difference = (c+1)/k - (-(k-c-1)/k) = (c+1)/k + (k-c-1)/k = k/k = 1
        // The jump between the "transition" rows is always exactly 1
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } });
        int k = 4;

        var encoder = new BackwardDifferenceEncoder<double>();
        encoder.Fit(data);
        var result = encoder.Transform(data);

        // For each contrast column c, the transition from row c to row c+1 = jump of 1
        for (int c = 0; c < k - 1; c++)
        {
            double valueBefore = result[c, c];     // row <= col: -(k-c-1)/k
            double valueAfter = result[c + 1, c];  // row > col: (c+1)/k
            double jump = valueAfter - valueBefore;
            Assert.Equal(1.0, jump, Tol);
        }
    }

    [Fact]
    public void BackwardDifferenceEncoder_UnknownCategory_ZeroVector()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        var encoder = new BackwardDifferenceEncoder<double>(handleUnknown: BackwardDifferenceHandleUnknown.UseZeros);
        encoder.Fit(data);

        var testData = MakeMatrix(new double[,] { { 99 } });
        var result = encoder.Transform(testData);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[0, 1], Tol);
    }

    // ========================================================================
    // Cross-Encoder Properties
    // ========================================================================

    [Fact]
    public void TargetEncoder_InverseTransform_NotSupported()
    {
        var encoder = new TargetEncoder<double>();
        Assert.False(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void WOEEncoder_InverseTransform_NotSupported()
    {
        var encoder = new WOEEncoder<double>();
        Assert.False(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void HelmertEncoder_InverseTransform_NotSupported()
    {
        var encoder = new HelmertEncoder<double>();
        Assert.False(encoder.SupportsInverseTransform);
    }

    [Fact]
    public void TargetEncoder_FitWithoutTarget_Throws()
    {
        var encoder = new TargetEncoder<double>();
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });
        Assert.Throws<InvalidOperationException>(() => encoder.Fit(data));
    }

    [Fact]
    public void WOEEncoder_FitWithoutTarget_Throws()
    {
        var encoder = new WOEEncoder<double>();
        var data = MakeMatrix(new double[,] { { 1 }, { 2 } });
        Assert.Throws<InvalidOperationException>(() => encoder.Fit(data));
    }
}
