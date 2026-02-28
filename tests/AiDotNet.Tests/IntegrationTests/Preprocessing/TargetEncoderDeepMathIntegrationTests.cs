using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep mathematical correctness tests for TargetEncoder (target mean encoding with smoothing).
/// Each test verifies exact hand-calculated smoothed means against the formula:
/// smoothed_mean = (count * category_mean + smoothing * global_mean) / (count + smoothing)
/// </summary>
public class TargetEncoderDeepMathIntegrationTests
{
    #region Helpers

    private static Matrix<double> M(double[,] data) => new(data);
    private static Vector<double> V(double[] data) => new(data);

    private static void AssertClose(double actual, double expected, double tol = 1e-10)
    {
        Assert.True(
            Math.Abs(actual - expected) < tol,
            $"expected {expected}, got {actual} (diff={Math.Abs(actual - expected)})");
    }

    #endregion

    #region Basic Target Encoding

    /// <summary>
    /// Hand-computed target encoding with smoothing=1.
    /// Data: category=[1, 1, 2, 2], target=[10, 20, 30, 40]
    /// global_mean = (10+20+30+40)/4 = 25
    ///
    /// Category 1: count=2, sum=30, mean=15
    ///   smoothed = (2*15 + 1*25) / (2+1) = (30+25)/3 = 55/3 ≈ 18.333...
    ///
    /// Category 2: count=2, sum=70, mean=35
    ///   smoothed = (2*35 + 1*25) / (2+1) = (70+25)/3 = 95/3 ≈ 31.666...
    /// </summary>
    [Fact]
    public void TargetEncoder_SimpleSmoothing_HandComputed()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 10.0, 20.0, 30.0, 40.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double expectedCat1 = 55.0 / 3.0;
        double expectedCat2 = 95.0 / 3.0;

        AssertClose(result[0, 0], expectedCat1);
        AssertClose(result[1, 0], expectedCat1);
        AssertClose(result[2, 0], expectedCat2);
        AssertClose(result[3, 0], expectedCat2);
    }

    /// <summary>
    /// With smoothing=0, encoding should equal the exact category mean (no smoothing).
    /// Formula: (count * mean + 0 * global) / (count + 0) = mean
    /// </summary>
    [Fact]
    public void TargetEncoder_SmoothingZero_ExactCategoryMean()
    {
        var encoder = new TargetEncoder<double>(smoothing: 0.0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 10.0, 20.0, 30.0, 40.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        // Category 1: mean = 15
        AssertClose(result[0, 0], 15.0);
        AssertClose(result[1, 0], 15.0);
        // Category 2: mean = 35
        AssertClose(result[2, 0], 35.0);
        AssertClose(result[3, 0], 35.0);
    }

    /// <summary>
    /// With very large smoothing, encoding should approach global mean.
    /// As smoothing → ∞: smoothed = (count*mean + smoothing*global) / (count+smoothing) → global
    /// </summary>
    [Fact]
    public void TargetEncoder_VeryLargeSmoothing_ApproachesGlobalMean()
    {
        var encoder = new TargetEncoder<double>(smoothing: 10000.0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 10.0, 20.0, 30.0, 40.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double globalMean = 25.0;
        // With smoothing=10000: (2*mean + 10000*25)/(10002) ≈ 25
        Assert.True(Math.Abs(result[0, 0] - globalMean) < 0.01, $"Expected ~{globalMean}, got {result[0, 0]}");
        Assert.True(Math.Abs(result[2, 0] - globalMean) < 0.01, $"Expected ~{globalMean}, got {result[2, 0]}");
    }

    /// <summary>
    /// All categories have the same target mean → encoded value equals global mean for all.
    /// Category 1: mean=25, Category 2: mean=25, global=25
    /// smoothed = (2*25 + 1*25) / (2+1) = 75/3 = 25
    /// </summary>
    [Fact]
    public void TargetEncoder_SameTargetMean_EncodedEqualsGlobalMean()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 20.0, 30.0, 20.0, 30.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        // Both categories have mean = 25, global = 25
        AssertClose(result[0, 0], 25.0);
        AssertClose(result[2, 0], 25.0);
    }

    #endregion

    #region Smoothing Formula Verification

    /// <summary>
    /// Verify exact smoothing formula with 3 categories and smoothing=2.
    /// Data: [1,1,1, 2,2, 3], target: [10,20,30, 40,50, 60]
    /// global_mean = (10+20+30+40+50+60)/6 = 210/6 = 35
    ///
    /// Category 1: count=3, sum=60, mean=20
    ///   smoothed = (3*20 + 2*35) / (3+2) = (60+70)/5 = 130/5 = 26
    ///
    /// Category 2: count=2, sum=90, mean=45
    ///   smoothed = (2*45 + 2*35) / (2+2) = (90+70)/4 = 160/4 = 40
    ///
    /// Category 3: count=1, sum=60, mean=60
    ///   smoothed = (1*60 + 2*35) / (1+2) = (60+70)/3 = 130/3 ≈ 43.333...
    /// </summary>
    [Fact]
    public void TargetEncoder_3Categories_Smoothing2_HandComputed()
    {
        var encoder = new TargetEncoder<double>(smoothing: 2.0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 }, { 3.0 } });
        var target = V(new double[] { 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double expectedCat1 = 130.0 / 5.0; // 26.0
        double expectedCat2 = 160.0 / 4.0; // 40.0
        double expectedCat3 = 130.0 / 3.0; // 43.333...

        AssertClose(result[0, 0], expectedCat1);
        AssertClose(result[3, 0], expectedCat2);
        AssertClose(result[5, 0], expectedCat3);
    }

    /// <summary>
    /// The smoothing formula is a weighted average between category mean and global mean.
    /// Weight of category_mean = count/(count+smoothing)
    /// Weight of global_mean = smoothing/(count+smoothing)
    /// These weights always sum to 1.
    /// </summary>
    [Fact]
    public void TargetEncoder_WeightsSumToOne()
    {
        double smoothing = 3.0;
        int count = 5;
        double wCategory = (double)count / (count + smoothing); // 5/8
        double wGlobal = smoothing / (count + smoothing);        // 3/8

        AssertClose(wCategory + wGlobal, 1.0);

        // Also verify that encoded value is between category_mean and global_mean
        var encoder = new TargetEncoder<double>(smoothing: smoothing);
        var values = new double[count + 3, 1];
        var targets = new double[count + 3];
        // Category 1 (count=5): targets all = 100
        for (int i = 0; i < count; i++) { values[i, 0] = 1.0; targets[i] = 100.0; }
        // Category 2 (count=3): targets all = 0
        for (int i = count; i < count + 3; i++) { values[i, 0] = 2.0; targets[i] = 0.0; }

        var data = M(values);
        var target = V(targets);
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double globalMean = 500.0 / 8.0; // 62.5
        double cat1Mean = 100.0;
        double cat1Smoothed = (5.0 * 100.0 + 3.0 * 62.5) / 8.0; // (500 + 187.5) / 8 = 85.9375

        AssertClose(result[0, 0], cat1Smoothed);
        // Smoothed value is between category mean and global mean
        Assert.True(result[0, 0] >= globalMean && result[0, 0] <= cat1Mean,
            $"Smoothed value {result[0, 0]} should be between {globalMean} and {cat1Mean}");
    }

    #endregion

    #region MinSamplesLeaf

    /// <summary>
    /// Categories with fewer samples than minSamplesLeaf should use global mean.
    /// With minSamplesLeaf=3: category with count=1 uses global mean.
    /// </summary>
    [Fact]
    public void TargetEncoder_MinSamplesLeaf_RareCategoryUsesGlobalMean()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0, minSamplesLeaf: 3);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 1.0 }, { 2.0 } });
        var target = V(new double[] { 10.0, 20.0, 30.0, 100.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double globalMean = 160.0 / 4.0; // 40.0

        // Category 1: count=3 >= minSamplesLeaf → smoothed encoding
        double cat1Mean = 60.0 / 3.0; // 20
        double cat1Smoothed = (3.0 * 20.0 + 1.0 * 40.0) / 4.0; // (60+40)/4 = 25
        AssertClose(result[0, 0], cat1Smoothed);

        // Category 2: count=1 < minSamplesLeaf → uses global mean
        AssertClose(result[3, 0], globalMean);
    }

    #endregion

    #region Unknown Categories

    /// <summary>
    /// Unknown category with UseGlobalMean mode should return global target mean.
    /// </summary>
    [Fact]
    public void TargetEncoder_UnknownCategory_UseGlobalMean()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0, handleUnknown: TargetEncoderHandleUnknown.UseGlobalMean);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 10.0, 20.0, 30.0, 40.0 });
        encoder.Fit(data, target);

        var testData = M(new double[,] { { 99.0 } });
        var result = encoder.Transform(testData);

        double globalMean = 25.0;
        AssertClose(result[0, 0], globalMean);
    }

    /// <summary>
    /// Unknown category with Error mode should throw.
    /// </summary>
    [Fact]
    public void TargetEncoder_UnknownCategory_Error_Throws()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0, handleUnknown: TargetEncoderHandleUnknown.Error);
        var data = M(new double[,] { { 1.0 }, { 2.0 } });
        var target = V(new double[] { 10.0, 20.0 });
        encoder.Fit(data, target);

        var testData = M(new double[,] { { 99.0 } });
        Assert.Throws<ArgumentException>(() => encoder.Transform(testData));
    }

    #endregion

    #region Edge Cases and Validation

    /// <summary>
    /// Single category: smoothed mean should blend with global (which equals the category mean).
    /// global_mean = category_mean = some value
    /// smoothed = (count * mean + smoothing * mean) / (count + smoothing) = mean
    /// </summary>
    [Fact]
    public void TargetEncoder_SingleCategory_SmoothedEqualsGlobalMean()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 1.0 } });
        var target = V(new double[] { 10.0, 20.0, 30.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double globalMean = 20.0;
        double catMean = 20.0;
        // smoothed = (3*20 + 1*20) / (3+1) = 80/4 = 20
        AssertClose(result[0, 0], 20.0);
    }

    /// <summary>
    /// Binary target (0/1): encoding should give smoothed probability.
    /// Category 1: 2 events out of 3 → mean = 2/3
    /// Category 2: 1 event out of 3 → mean = 1/3
    /// global = 3/6 = 0.5, smoothing = 1
    ///
    /// Cat 1 smoothed = (3 * 2/3 + 1 * 0.5) / (3+1) = (2 + 0.5) / 4 = 2.5/4 = 0.625
    /// Cat 2 smoothed = (3 * 1/3 + 1 * 0.5) / (3+1) = (1 + 0.5) / 4 = 1.5/4 = 0.375
    /// </summary>
    [Fact]
    public void TargetEncoder_BinaryTarget_SmoothedProbabilities()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double expectedCat1 = 2.5 / 4.0; // 0.625
        double expectedCat2 = 1.5 / 4.0; // 0.375

        AssertClose(result[0, 0], expectedCat1);
        AssertClose(result[3, 0], expectedCat2);
    }

    /// <summary>
    /// Target length must match data rows.
    /// </summary>
    [Fact]
    public void TargetEncoder_MismatchedLength_Throws()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0);
        var data = M(new double[,] { { 1.0 }, { 2.0 } });
        var target = V(new double[] { 10.0 }); // length 1 != 2 rows
        Assert.Throws<ArgumentException>(() => encoder.Fit(data, target));
    }

    /// <summary>
    /// Negative smoothing should throw.
    /// </summary>
    [Fact]
    public void TargetEncoder_NegativeSmoothing_Throws()
    {
        Assert.Throws<ArgumentException>(() => new TargetEncoder<double>(smoothing: -1.0));
    }

    /// <summary>
    /// Pass-through column should preserve original values.
    /// </summary>
    [Fact]
    public void TargetEncoder_PassThroughColumn_PreservesValues()
    {
        var encoder = new TargetEncoder<double>(smoothing: 1.0, columnIndices: new[] { 0 });
        var data = M(new double[,] { { 1.0, 42.0 }, { 1.0, 88.0 }, { 2.0, 77.0 }, { 2.0, 33.0 } });
        var target = V(new double[] { 10.0, 20.0, 30.0, 40.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        AssertClose(result[0, 1], 42.0);
        AssertClose(result[1, 1], 88.0);
        AssertClose(result[2, 1], 77.0);
        AssertClose(result[3, 1], 33.0);
    }

    #endregion
}
