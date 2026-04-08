using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep mathematical correctness tests for WOEEncoder (Weight of Evidence).
/// Each test verifies exact hand-calculated WOE values and Information Values
/// against the statistical formulas used in credit scoring and binary classification.
///
/// Formula: WOE = ln(dist_events / dist_non_events)
/// where dist_events = (events + reg) / (total_events + reg * num_categories)
///       dist_non_events = (non_events + reg) / (total_non_events + reg * num_categories)
/// IV = Σ (dist_events - dist_non_events) * WOE
/// </summary>
public class WOEEncoderDeepMathIntegrationTests
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

    #region Basic WOE Computation

    /// <summary>
    /// Hand-computed WOE for a simple 2-category, 4-sample scenario.
    /// Data: category=[A, A, B, B] (encoded as [1, 1, 2, 2])
    /// Target: [1, 0, 0, 1]
    /// total_events=2, total_non_events=2
    /// num_categories=2, regularization=0.5
    ///
    /// Category A (value=1): events=1, non_events=1
    ///   dist_events = (1 + 0.5) / (2 + 0.5*2) = 1.5/3 = 0.5
    ///   dist_non_events = (1 + 0.5) / (2 + 0.5*2) = 1.5/3 = 0.5
    ///   WOE_A = ln(0.5/0.5) = ln(1) = 0
    ///
    /// Category B (value=2): events=1, non_events=1
    ///   Same calculation → WOE_B = 0
    /// </summary>
    [Fact]
    public void WOEEncoder_EqualDistribution_WOEIsZero()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        // Both categories have balanced events/non-events → WOE = 0
        AssertClose(result[0, 0], 0.0);
        AssertClose(result[1, 0], 0.0);
        AssertClose(result[2, 0], 0.0);
        AssertClose(result[3, 0], 0.0);
    }

    /// <summary>
    /// Hand-computed WOE for skewed distribution.
    /// Data: category=[1, 1, 1, 2, 2, 2] (3 each)
    /// Target: [1, 1, 0, 0, 0, 1]
    /// total_events=3, total_non_events=3
    /// num_categories=2, regularization=0.5
    ///
    /// Category 1: events=2, non_events=1
    ///   dist_events = (2 + 0.5) / (3 + 0.5*2) = 2.5/4 = 0.625
    ///   dist_non_events = (1 + 0.5) / (3 + 0.5*2) = 1.5/4 = 0.375
    ///   WOE_1 = ln(0.625/0.375) = ln(5/3) ≈ 0.5108256...
    ///
    /// Category 2: events=1, non_events=2
    ///   dist_events = (1 + 0.5) / (3 + 0.5*2) = 1.5/4 = 0.375
    ///   dist_non_events = (2 + 0.5) / (3 + 0.5*2) = 2.5/4 = 0.625
    ///   WOE_2 = ln(0.375/0.625) = ln(3/5) ≈ -0.5108256...
    /// </summary>
    [Fact]
    public void WOEEncoder_SkewedDistribution_CorrectWOE()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double expectedWoe1 = Math.Log(5.0 / 3.0); // ln(0.625/0.375)
        double expectedWoe2 = Math.Log(3.0 / 5.0); // ln(0.375/0.625)

        // Category 1 rows (indices 0, 1, 2)
        AssertClose(result[0, 0], expectedWoe1);
        AssertClose(result[1, 0], expectedWoe1);
        AssertClose(result[2, 0], expectedWoe1);

        // Category 2 rows (indices 3, 4, 5)
        AssertClose(result[3, 0], expectedWoe2);
        AssertClose(result[4, 0], expectedWoe2);
        AssertClose(result[5, 0], expectedWoe2);
    }

    /// <summary>
    /// WOE with regularization=0 and category that has all events.
    /// Without regularization, if a category has 0 non-events, dist_non_events would be 0.
    /// With reg=0: distNonEvents = 0 / total_non_events = 0, causing division by zero in log.
    /// Actually, with reg=0:
    ///   dist_events = (events + 0) / (total_events + 0*num_categories) = events/total_events
    ///   dist_non_events = (non_events + 0) / (total_non_events + 0*num_categories) = non_events/total_non_events
    /// If events=2, non_events=0 for category: dist_non_events = 0/total_non_events = 0
    /// log(dist_events/0) = +infinity, clamped to 5.
    /// </summary>
    [Fact]
    public void WOEEncoder_AllEventsInCategory_RegZero_ClampedTo5()
    {
        var encoder = new WOEEncoder<double>(regularization: 0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 1.0, 0.0, 0.0 });
        // total_events=2, total_non_events=2
        // Category 1: events=2, non_events=0 → dist_non_events=0 → WOE=+inf → clamp to 5
        // Category 2: events=0, non_events=2 → dist_events=0 → WOE=-inf → clamp to -5
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        AssertClose(result[0, 0], 5.0); // clamped +inf
        AssertClose(result[1, 0], 5.0);
        AssertClose(result[2, 0], -5.0); // clamped -inf
        AssertClose(result[3, 0], -5.0);
    }

    /// <summary>
    /// WOE antisymmetry property: for 2 categories with complementary distributions,
    /// WOE_1 = -WOE_2 (they are equal in magnitude but opposite in sign).
    /// </summary>
    [Fact]
    public void WOEEncoder_TwoCategories_AntisymmetricWOE()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double woe1 = result[0, 0];
        double woe2 = result[3, 0];

        // WOE_1 + WOE_2 should be 0 (antisymmetric)
        // Because category 1 has events=2, non=1 and category 2 has events=1, non=2
        // They're complementary, so the distributions swap
        AssertClose(woe1 + woe2, 0.0);
    }

    #endregion

    #region Regularization Effects

    /// <summary>
    /// Larger regularization should reduce the magnitude of WOE values.
    /// With reg=0.5: WOE is moderate.
    /// With reg=5.0: WOE should be closer to zero (more shrinkage).
    /// </summary>
    [Fact]
    public void WOEEncoder_LargerRegularization_SmallerMagnitudeWOE()
    {
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 1.0, 0.0, 0.0 });

        // Small regularization
        var encoder1 = new WOEEncoder<double>(regularization: 0.5);
        encoder1.Fit(data, target);
        var result1 = encoder1.Transform(data);
        double woe1 = Math.Abs(result1[0, 0]);

        // Large regularization
        var encoder2 = new WOEEncoder<double>(regularization: 5.0);
        encoder2.Fit(data, target);
        var result2 = encoder2.Transform(data);
        double woe2 = Math.Abs(result2[0, 0]);

        // Larger regularization → smaller |WOE|
        Assert.True(woe2 < woe1, $"|WOE with reg=5| = {woe2} should be < |WOE with reg=0.5| = {woe1}");
    }

    /// <summary>
    /// With very large regularization, all categories approach WOE=0.
    /// reg=1000: dist_events ≈ (events + 1000) / (total_events + 1000*k) ≈ 1000/(k*1000) = 1/k
    /// dist_non_events ≈ 1/k similarly
    /// WOE = ln(1) = 0
    /// </summary>
    [Fact]
    public void WOEEncoder_VeryLargeRegularization_WOENearZero()
    {
        var encoder = new WOEEncoder<double>(regularization: 1000.0);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 1.0, 0.0, 0.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        // With reg=1000, WOE should be very close to 0
        Assert.True(Math.Abs(result[0, 0]) < 0.01, $"WOE with reg=1000: expected ~0, got {result[0, 0]}");
        Assert.True(Math.Abs(result[2, 0]) < 0.01, $"WOE with reg=1000: expected ~0, got {result[2, 0]}");
    }

    #endregion

    #region Information Value

    /// <summary>
    /// Hand-computed IV for the skewed example.
    /// From SkewedDistribution test:
    /// Category 1: dist_events=0.625, dist_non_events=0.375, WOE=ln(5/3)
    /// Category 2: dist_events=0.375, dist_non_events=0.625, WOE=ln(3/5)
    ///
    /// IV = Σ (dist_events - dist_non_events) * WOE
    ///    = (0.625 - 0.375) * ln(5/3) + (0.375 - 0.625) * ln(3/5)
    ///    = 0.25 * ln(5/3) + (-0.25) * ln(3/5)
    ///    = 0.25 * ln(5/3) + 0.25 * ln(5/3)  (since -ln(3/5) = ln(5/3))
    ///    = 0.5 * ln(5/3)
    ///    ≈ 0.5 * 0.5108256 ≈ 0.2554128
    /// </summary>
    [Fact]
    public void WOEEncoder_InformationValue_HandComputed()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);

        var ivValues = encoder.CalculateInformationValue(data, target);

        double expectedIV = 0.5 * Math.Log(5.0 / 3.0);
        Assert.True(ivValues.ContainsKey(0));
        AssertClose(ivValues[0], expectedIV);
    }

    /// <summary>
    /// IV for balanced distribution should be 0 (no predictive power).
    /// When each category has equal event/non-event ratios, WOE=0 for all categories,
    /// so IV = Σ 0 * 0 = 0.
    /// </summary>
    [Fact]
    public void WOEEncoder_InformationValue_BalancedDistribution_IVIsZero()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 0.0, 1.0, 0.0 });
        encoder.Fit(data, target);

        var ivValues = encoder.CalculateInformationValue(data, target);

        Assert.True(ivValues.ContainsKey(0));
        AssertClose(ivValues[0], 0.0);
    }

    /// <summary>
    /// IV is always non-negative (it's a sum of products where each term is non-negative).
    /// This is because (dist_events - dist_non_events) and WOE always have the same sign:
    /// If dist_events > dist_non_events, WOE > 0 → product > 0
    /// If dist_events < dist_non_events, WOE < 0 → product > 0
    /// If dist_events = dist_non_events, both are 0 → product = 0
    /// </summary>
    [Fact]
    public void WOEEncoder_InformationValue_AlwaysNonNegative()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] {
            { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 },
            { 2.0 }, { 2.0 }, { 2.0 },
            { 3.0 }, { 3.0 }, { 3.0 }
        });
        var target = V(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);

        var ivValues = encoder.CalculateInformationValue(data, target);

        Assert.True(ivValues[0] >= -1e-10, $"IV should be non-negative, got {ivValues[0]}");
    }

    #endregion

    #region WOE Sign Property

    /// <summary>
    /// Categories with more events than expected should have positive WOE.
    /// Categories with fewer events than expected should have negative WOE.
    /// </summary>
    [Fact]
    public void WOEEncoder_PositiveWOE_MoreEvents_NegativeWOE_FewerEvents()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        // Category 1: 3 events, 1 non-event (more events → positive WOE)
        // Category 2: 1 event, 3 non-events (fewer events → negative WOE)
        var data = M(new double[,] {
            { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 },
            { 2.0 }, { 2.0 }, { 2.0 }, { 2.0 }
        });
        var target = V(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        Assert.True(result[0, 0] > 0, $"Category 1 (more events) should have WOE > 0, got {result[0, 0]}");
        Assert.True(result[4, 0] < 0, $"Category 2 (fewer events) should have WOE < 0, got {result[4, 0]}");
    }

    #endregion

    #region Multiple Categories

    /// <summary>
    /// WOE with 3 categories, each with different event ratios.
    /// Category 1: events=3, non=0 (all events)
    /// Category 2: events=0, non=3 (all non-events)
    /// Category 3: events=1, non=2 (mixed)
    /// Total events=4, total non_events=5, num_categories=3, reg=0.5
    ///
    /// Category 1: dist_events=(3+0.5)/(4+1.5)=3.5/5.5, dist_non=(0+0.5)/(5+1.5)=0.5/6.5
    /// WOE_1 = ln((3.5/5.5)/(0.5/6.5)) = ln(3.5*6.5 / (5.5*0.5)) = ln(22.75/2.75) = ln(8.2727...)
    ///
    /// Category 2: dist_events=(0+0.5)/(4+1.5)=0.5/5.5, dist_non=(3+0.5)/(5+1.5)=3.5/6.5
    /// WOE_2 = ln((0.5/5.5)/(3.5/6.5)) = ln(0.5*6.5 / (5.5*3.5)) = ln(3.25/19.25) = ln(0.16883...)
    ///
    /// Category 3: dist_events=(1+0.5)/(4+1.5)=1.5/5.5, dist_non=(2+0.5)/(5+1.5)=2.5/6.5
    /// WOE_3 = ln((1.5/5.5)/(2.5/6.5)) = ln(1.5*6.5 / (5.5*2.5)) = ln(9.75/13.75)
    /// </summary>
    [Fact]
    public void WOEEncoder_3Categories_HandComputed()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] {
            { 1.0 }, { 1.0 }, { 1.0 },
            { 2.0 }, { 2.0 }, { 2.0 },
            { 3.0 }, { 3.0 }, { 3.0 }
        });
        var target = V(new double[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        double expectedWoe1 = Math.Log((3.5 / 5.5) / (0.5 / 6.5));
        double expectedWoe2 = Math.Log((0.5 / 5.5) / (3.5 / 6.5));
        double expectedWoe3 = Math.Log((1.5 / 5.5) / (2.5 / 6.5));

        // Clamp check: expectedWoe1 = ln(22.75/2.75) ≈ 2.112 (within [-5, 5])
        AssertClose(result[0, 0], expectedWoe1, 1e-8);
        AssertClose(result[3, 0], expectedWoe2, 1e-8);
        AssertClose(result[6, 0], expectedWoe3, 1e-8);
    }

    #endregion

    #region Unknown Categories and Validation

    /// <summary>
    /// Unknown category with UseZero mode should return WOE=0.
    /// </summary>
    [Fact]
    public void WOEEncoder_UnknownCategory_UseZero_ReturnsZero()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5, handleUnknown: WOEHandleUnknown.UseZero);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);

        var testData = M(new double[,] { { 99.0 } });
        var result = encoder.Transform(testData);
        AssertClose(result[0, 0], 0.0);
    }

    /// <summary>
    /// Unknown category with Error mode should throw.
    /// </summary>
    [Fact]
    public void WOEEncoder_UnknownCategory_Error_Throws()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5, handleUnknown: WOEHandleUnknown.Error);
        var data = M(new double[,] { { 1.0 }, { 1.0 }, { 2.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);

        var testData = M(new double[,] { { 99.0 } });
        Assert.Throws<ArgumentException>(() => encoder.Transform(testData));
    }

    /// <summary>
    /// Target must be binary (0 or 1). Non-binary values should throw.
    /// </summary>
    [Fact]
    public void WOEEncoder_NonBinaryTarget_Throws()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] { { 1.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 0.5 }); // 0.5 is not binary!
        Assert.Throws<ArgumentException>(() => encoder.Fit(data, target));
    }

    /// <summary>
    /// Target with only one class should throw.
    /// </summary>
    [Fact]
    public void WOEEncoder_SingleClassTarget_Throws()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5);
        var data = M(new double[,] { { 1.0 }, { 2.0 } });
        var target = V(new double[] { 1.0, 1.0 }); // all events, no non-events
        Assert.Throws<ArgumentException>(() => encoder.Fit(data, target));
    }

    /// <summary>
    /// Negative regularization should throw.
    /// </summary>
    [Fact]
    public void WOEEncoder_NegativeRegularization_Throws()
    {
        Assert.Throws<ArgumentException>(() => new WOEEncoder<double>(regularization: -1.0));
    }

    /// <summary>
    /// Pass-through column should preserve original values.
    /// </summary>
    [Fact]
    public void WOEEncoder_PassThroughColumn_PreservesValues()
    {
        var encoder = new WOEEncoder<double>(regularization: 0.5, columnIndices: new[] { 0 });
        var data = M(new double[,] { { 1.0, 42.0 }, { 1.0, 88.0 }, { 2.0, 77.0 }, { 2.0, 33.0 } });
        var target = V(new double[] { 1.0, 0.0, 0.0, 1.0 });
        encoder.Fit(data, target);
        var result = encoder.Transform(data);

        // Column 1 should be pass-through
        AssertClose(result[0, 1], 42.0);
        AssertClose(result[1, 1], 88.0);
        AssertClose(result[2, 1], 77.0);
        AssertClose(result[3, 1], 33.0);
    }

    #endregion
}
