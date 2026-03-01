using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Deep math-correctness integration tests for NumericalStabilityHelper and GradientClippingHelper.
/// Verifies safe math operations, softmax stability, gradient clipping formulas,
/// and edge cases with NaN/Infinity values.
/// </summary>
public class NumericalStabilityAndGradientClippingDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-5;

    #region SafeLog

    [Fact]
    public void SafeLog_PositiveValue_ReturnsExactLog()
    {
        // SafeLog(e) = 1.0 (natural log of e)
        double result = NumericalStabilityHelper.SafeLog(Math.E);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void SafeLog_One_ReturnsZero()
    {
        double result = NumericalStabilityHelper.SafeLog(1.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SafeLog_Zero_ReturnsLogOfEpsilon()
    {
        // SafeLog(0) should clamp to epsilon and return log(epsilon)
        double expected = Math.Log(NumericalStabilityHelper.DefaultEpsilon);
        double result = NumericalStabilityHelper.SafeLog(0.0);
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void SafeLog_NegativeValue_ReturnsLogOfEpsilon()
    {
        // SafeLog(-5) should clamp to epsilon
        double expected = Math.Log(NumericalStabilityHelper.DefaultEpsilon);
        double result = NumericalStabilityHelper.SafeLog(-5.0);
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void SafeLog_CustomEpsilon_UsesCustomValue()
    {
        double eps = 1e-3;
        double expected = Math.Log(eps);
        double result = NumericalStabilityHelper.SafeLog(0.0, eps);
        Assert.Equal(expected, result, Tolerance);
    }

    #endregion

    #region SafeDiv

    [Fact]
    public void SafeDiv_NormalValues_ReturnsExactQuotient()
    {
        // 10 / 3 = 3.333...
        double result = NumericalStabilityHelper.SafeDiv(10.0, 3.0);
        Assert.Equal(10.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void SafeDiv_ZeroDenominator_ReturnsFiniteValue()
    {
        // 5 / 0 → should return 5 / epsilon
        double result = NumericalStabilityHelper.SafeDiv(5.0, 0.0);
        double expected = 5.0 / NumericalStabilityHelper.DefaultEpsilon;
        Assert.Equal(expected, result, Tolerance);
        Assert.True((!double.IsNaN(result) && !double.IsInfinity(result)), "Result should be finite");
    }

    [Fact]
    public void SafeDiv_NegativeDenominator_PreservesSign()
    {
        // When denominator is very small negative, result should be negative
        double result = NumericalStabilityHelper.SafeDiv(5.0, -1e-20);
        Assert.True(result < 0, "Division by small negative should give negative result");
    }

    [Fact]
    public void SafeDiv_ZeroDivZero_ReturnsOne()
    {
        // 0 / 0 → 0 / epsilon = 0
        double result = NumericalStabilityHelper.SafeDiv(0.0, 0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region SafeSqrt

    [Fact]
    public void SafeSqrt_PositiveValue_ReturnsExactSqrt()
    {
        double result = NumericalStabilityHelper.SafeSqrt(4.0);
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void SafeSqrt_Zero_ReturnsSqrtOfEpsilon()
    {
        double expected = Math.Sqrt(NumericalStabilityHelper.DefaultEpsilon);
        double result = NumericalStabilityHelper.SafeSqrt(0.0);
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void SafeSqrt_NegativeValue_ReturnsSqrtOfEpsilon()
    {
        double expected = Math.Sqrt(NumericalStabilityHelper.DefaultEpsilon);
        double result = NumericalStabilityHelper.SafeSqrt(-10.0);
        Assert.Equal(expected, result, Tolerance);
    }

    #endregion

    #region ClampProbability

    [Fact]
    public void ClampProbability_InRange_Unchanged()
    {
        double result = NumericalStabilityHelper.ClampProbability(0.5);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void ClampProbability_Zero_ClampsToEpsilon()
    {
        double result = NumericalStabilityHelper.ClampProbability(0.0);
        Assert.Equal(NumericalStabilityHelper.DefaultEpsilon, result, Tolerance);
    }

    [Fact]
    public void ClampProbability_One_ClampsToOneMinusEpsilon()
    {
        double result = NumericalStabilityHelper.ClampProbability(1.0);
        Assert.Equal(1.0 - NumericalStabilityHelper.DefaultEpsilon, result, Tolerance);
    }

    [Fact]
    public void ClampProbability_Negative_ClampsToEpsilon()
    {
        double result = NumericalStabilityHelper.ClampProbability(-0.1);
        Assert.Equal(NumericalStabilityHelper.DefaultEpsilon, result, Tolerance);
    }

    [Fact]
    public void ClampProbability_GreaterThanOne_ClampsToOneMinusEpsilon()
    {
        double result = NumericalStabilityHelper.ClampProbability(1.5);
        Assert.Equal(1.0 - NumericalStabilityHelper.DefaultEpsilon, result, Tolerance);
    }

    #endregion

    #region SafeLogProbability

    [Fact]
    public void SafeLogProbability_ValidProbability_ReturnsExactLog()
    {
        // SafeLogProbability(0.5) = log(0.5) = -ln(2)
        double result = NumericalStabilityHelper.SafeLogProbability(0.5);
        Assert.Equal(-Math.Log(2.0), result, Tolerance);
    }

    [Fact]
    public void SafeLogProbability_Zero_ReturnsLogEpsilon()
    {
        // 0 → clamped to eps → log(eps)
        double result = NumericalStabilityHelper.SafeLogProbability(0.0);
        double expected = Math.Log(NumericalStabilityHelper.DefaultEpsilon);
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void SafeLogProbability_One_ReturnsNearZero()
    {
        // 1 → clamped to 1-eps → log(1-eps) ≈ -eps
        double result = NumericalStabilityHelper.SafeLogProbability(1.0);
        // log(1-eps) is very close to 0 but slightly negative
        Assert.True(result < 0 && result > -1e-4,
            $"SafeLogProbability(1) = {result} should be very close to 0");
    }

    #endregion

    #region StableSoftmax

    [Fact]
    public void StableSoftmax_SimpleInput_HandCalculated()
    {
        // softmax([1, 2, 3]) = [e^1, e^2, e^3] / (e^1 + e^2 + e^3)
        var logits = new Vector<double>([1.0, 2.0, 3.0]);
        var result = NumericalStabilityHelper.StableSoftmax(logits);

        double e1 = Math.Exp(1), e2 = Math.Exp(2), e3 = Math.Exp(3);
        double sum = e1 + e2 + e3;

        Assert.NotNull(result);
        Assert.Equal(e1 / sum, result[0], Tolerance);
        Assert.Equal(e2 / sum, result[1], Tolerance);
        Assert.Equal(e3 / sum, result[2], Tolerance);
    }

    [Fact]
    public void StableSoftmax_SumsToOne()
    {
        var logits = new Vector<double>([1.0, 2.0, 3.0]);
        var result = NumericalStabilityHelper.StableSoftmax(logits);

        Assert.NotNull(result);
        double sum = 0;
        for (int i = 0; i < result.Length; i++) sum += result[i];
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void StableSoftmax_LargeValues_DoesNotOverflow()
    {
        // Without the max-subtraction trick, exp(1000) would overflow
        var logits = new Vector<double>([1000.0, 1001.0, 1002.0]);
        var result = NumericalStabilityHelper.StableSoftmax(logits);

        Assert.NotNull(result);
        // Result should be same as softmax([0, 1, 2]) due to shift invariance
        double e0 = Math.Exp(0), e1 = Math.Exp(1), e2 = Math.Exp(2);
        double sum = e0 + e1 + e2;

        Assert.Equal(e0 / sum, result[0], Tolerance);
        Assert.Equal(e1 / sum, result[1], Tolerance);
        Assert.Equal(e2 / sum, result[2], Tolerance);
    }

    [Fact]
    public void StableSoftmax_EqualInputs_UniformDistribution()
    {
        var logits = new Vector<double>([5.0, 5.0, 5.0, 5.0]);
        var result = NumericalStabilityHelper.StableSoftmax(logits);

        Assert.NotNull(result);
        for (int i = 0; i < result.Length; i++)
            Assert.Equal(0.25, result[i], Tolerance);
    }

    [Fact]
    public void StableSoftmax_NegativeValues_StillCorrect()
    {
        var logits = new Vector<double>([-1.0, -2.0, -3.0]);
        var result = NumericalStabilityHelper.StableSoftmax(logits);

        double em1 = Math.Exp(-1), em2 = Math.Exp(-2), em3 = Math.Exp(-3);
        double sum = em1 + em2 + em3;

        Assert.NotNull(result);
        Assert.Equal(em1 / sum, result[0], Tolerance);
        Assert.Equal(em2 / sum, result[1], Tolerance);
        Assert.Equal(em3 / sum, result[2], Tolerance);
    }

    [Fact]
    public void StableSoftmax_ShiftInvariance()
    {
        // softmax(x + c) = softmax(x) for any constant c
        var logits1 = new Vector<double>([1.0, 2.0, 3.0]);
        var logits2 = new Vector<double>([101.0, 102.0, 103.0]);

        var result1 = NumericalStabilityHelper.StableSoftmax(logits1);
        var result2 = NumericalStabilityHelper.StableSoftmax(logits2);

        Assert.NotNull(result1);
        Assert.NotNull(result2);
        for (int i = 0; i < result1.Length; i++)
            Assert.Equal(result1[i], result2[i], Tolerance);
    }

    [Fact]
    public void StableSoftmax_MonotonicInInput()
    {
        // Larger input → larger softmax output
        var logits = new Vector<double>([1.0, 3.0, 2.0]);
        var result = NumericalStabilityHelper.StableSoftmax(logits);

        Assert.NotNull(result);
        Assert.True(result[1] > result[2], "softmax(3) > softmax(2)");
        Assert.True(result[2] > result[0], "softmax(2) > softmax(1)");
    }

    [Fact]
    public void StableSoftmax_NullInput_ReturnsNull()
    {
        var result = NumericalStabilityHelper.StableSoftmax<double>(null);
        Assert.Null(result);
    }

    #endregion

    #region StableLogSoftmax

    [Fact]
    public void StableLogSoftmax_ConsistentWithSoftmax()
    {
        // log_softmax(x) should equal log(softmax(x))
        var logits = new Vector<double>([1.0, 2.0, 3.0]);
        var softmax = NumericalStabilityHelper.StableSoftmax(logits);
        var logSoftmax = NumericalStabilityHelper.StableLogSoftmax(logits);

        Assert.NotNull(softmax);
        Assert.NotNull(logSoftmax);
        for (int i = 0; i < logits.Length; i++)
            Assert.Equal(Math.Log(softmax[i]), logSoftmax[i], LooseTolerance);
    }

    [Fact]
    public void StableLogSoftmax_LogSumExpSubtraction()
    {
        // log_softmax(x_i) = x_i - log(sum(exp(x_j)))
        var logits = new Vector<double>([2.0, 4.0, 1.0]);
        var result = NumericalStabilityHelper.StableLogSoftmax(logits);

        double logSumExp = Math.Log(Math.Exp(2) + Math.Exp(4) + Math.Exp(1));
        Assert.NotNull(result);
        Assert.Equal(2.0 - logSumExp, result[0], LooseTolerance);
        Assert.Equal(4.0 - logSumExp, result[1], LooseTolerance);
        Assert.Equal(1.0 - logSumExp, result[2], LooseTolerance);
    }

    [Fact]
    public void StableLogSoftmax_AllValuesNegative()
    {
        // All log-softmax values should be <= 0
        var logits = new Vector<double>([1.0, 2.0, 3.0]);
        var result = NumericalStabilityHelper.StableLogSoftmax(logits);

        Assert.NotNull(result);
        for (int i = 0; i < result.Length; i++)
            Assert.True(result[i] <= 0, $"Log softmax[{i}]={result[i]} should be <= 0");
    }

    [Fact]
    public void StableLogSoftmax_ExpSumsToOne()
    {
        // exp(log_softmax) should sum to 1
        var logits = new Vector<double>([1.0, 2.0, 3.0]);
        var logSoftmax = NumericalStabilityHelper.StableLogSoftmax(logits);

        Assert.NotNull(logSoftmax);
        double sum = 0;
        for (int i = 0; i < logSoftmax.Length; i++)
            sum += Math.Exp(logSoftmax[i]);
        Assert.Equal(1.0, sum, LooseTolerance);
    }

    #endregion

    #region IsNaN, IsInfinity, IsFinite

    [Fact]
    public void IsNaN_DetectsNaN()
    {
        Assert.True(NumericalStabilityHelper.IsNaN(double.NaN));
        Assert.False(NumericalStabilityHelper.IsNaN(1.0));
        Assert.False(NumericalStabilityHelper.IsNaN(double.PositiveInfinity));
    }

    [Fact]
    public void IsInfinity_DetectsInfinity()
    {
        Assert.True(NumericalStabilityHelper.IsInfinity(double.PositiveInfinity));
        Assert.True(NumericalStabilityHelper.IsInfinity(double.NegativeInfinity));
        Assert.False(NumericalStabilityHelper.IsInfinity(1.0));
        Assert.False(NumericalStabilityHelper.IsInfinity(double.NaN));
    }

    [Fact]
    public void IsFinite_DetectsFinite()
    {
        Assert.True(NumericalStabilityHelper.IsFinite(1.0));
        Assert.True(NumericalStabilityHelper.IsFinite(0.0));
        Assert.True(NumericalStabilityHelper.IsFinite(-999.0));
        Assert.False(NumericalStabilityHelper.IsFinite(double.NaN));
        Assert.False(NumericalStabilityHelper.IsFinite(double.PositiveInfinity));
    }

    #endregion

    #region Vector NaN/Infinity Detection

    [Fact]
    public void ContainsNaN_DetectsNaNInVector()
    {
        var vec = new Vector<double>([1.0, double.NaN, 3.0]);
        Assert.True(NumericalStabilityHelper.ContainsNaN(vec));

        var clean = new Vector<double>([1.0, 2.0, 3.0]);
        Assert.False(NumericalStabilityHelper.ContainsNaN(clean));
    }

    [Fact]
    public void ContainsInfinity_DetectsInfinityInVector()
    {
        var vec = new Vector<double>([1.0, double.PositiveInfinity, 3.0]);
        Assert.True(NumericalStabilityHelper.ContainsInfinity(vec));

        var clean = new Vector<double>([1.0, 2.0, 3.0]);
        Assert.False(NumericalStabilityHelper.ContainsInfinity(clean));
    }

    [Fact]
    public void CountNaN_CorrectCount()
    {
        var vec = new Vector<double>([double.NaN, 2.0, double.NaN, 4.0, double.NaN]);
        Assert.Equal(3, NumericalStabilityHelper.CountNaN(vec));
    }

    [Fact]
    public void CountInfinity_CorrectCount()
    {
        var vec = new Vector<double>([double.PositiveInfinity, 2.0, double.NegativeInfinity, 4.0]);
        Assert.Equal(2, NumericalStabilityHelper.CountInfinity(vec));
    }

    #endregion

    #region ReplaceNaN / ReplaceInfinity / ReplaceNonFinite

    [Fact]
    public void ReplaceNaN_ReplacesWithZero()
    {
        var vec = new Vector<double>([1.0, double.NaN, 3.0]);
        var result = NumericalStabilityHelper.ReplaceNaN(vec);

        Assert.NotNull(result);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void ReplaceNaN_CustomReplacement()
    {
        var vec = new Vector<double>([1.0, double.NaN, 3.0]);
        var result = NumericalStabilityHelper.ReplaceNaN(vec, -1.0);

        Assert.NotNull(result);
        Assert.Equal(-1.0, result[1], Tolerance);
    }

    [Fact]
    public void ReplaceInfinity_ReplacesWithZero()
    {
        var vec = new Vector<double>([1.0, double.PositiveInfinity, double.NegativeInfinity]);
        var result = NumericalStabilityHelper.ReplaceInfinity(vec);

        Assert.NotNull(result);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(0.0, result[2], Tolerance);
    }

    [Fact]
    public void ReplaceNonFinite_ReplacesAllBad()
    {
        var vec = new Vector<double>([1.0, double.NaN, double.PositiveInfinity, 4.0]);
        var result = NumericalStabilityHelper.ReplaceNonFinite(vec);

        Assert.NotNull(result);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(0.0, result[2], Tolerance);
        Assert.Equal(4.0, result[3], Tolerance);
    }

    #endregion

    #region AssertFinite

    [Fact]
    public void AssertFinite_NaN_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            NumericalStabilityHelper.AssertFinite(double.NaN));
    }

    [Fact]
    public void AssertFinite_Infinity_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            NumericalStabilityHelper.AssertFinite(double.PositiveInfinity));
    }

    [Fact]
    public void AssertFinite_ValidValue_NoThrow()
    {
        NumericalStabilityHelper.AssertFinite(42.0);
    }

    [Fact]
    public void AssertFinite_VectorWithNaN_Throws()
    {
        var vec = new Vector<double>([1.0, double.NaN, 3.0]);
        Assert.Throws<ArgumentException>(() =>
            NumericalStabilityHelper.AssertFinite(vec));
    }

    #endregion

    #region GradientClipping - ClipByValue

    [Fact]
    public void ClipByValue_WithinRange_Unchanged()
    {
        var grads = new Vector<double>([0.5, -0.3, 0.8]);
        var clipped = GradientClippingHelper.ClipByValue(grads, 1.0);

        Assert.NotNull(clipped);
        Assert.Equal(0.5, clipped[0], Tolerance);
        Assert.Equal(-0.3, clipped[1], Tolerance);
        Assert.Equal(0.8, clipped[2], Tolerance);
    }

    [Fact]
    public void ClipByValue_ExceedsRange_Clamped()
    {
        var grads = new Vector<double>([5.0, -3.0, 0.5]);
        var clipped = GradientClippingHelper.ClipByValue(grads, 1.0);

        Assert.NotNull(clipped);
        Assert.Equal(1.0, clipped[0], Tolerance);
        Assert.Equal(-1.0, clipped[1], Tolerance);
        Assert.Equal(0.5, clipped[2], Tolerance);
    }

    [Fact]
    public void ClipByValue_CustomMaxValue()
    {
        var grads = new Vector<double>([3.0, -4.0, 0.1]);
        var clipped = GradientClippingHelper.ClipByValue(grads, 2.0);

        Assert.NotNull(clipped);
        Assert.Equal(2.0, clipped[0], Tolerance);
        Assert.Equal(-2.0, clipped[1], Tolerance);
        Assert.Equal(0.1, clipped[2], Tolerance);
    }

    [Fact]
    public void ClipByValueInPlace_ModifiesOriginal()
    {
        var grads = new Vector<double>([5.0, -3.0, 0.5]);
        GradientClippingHelper.ClipByValueInPlace(grads, 1.0);

        Assert.Equal(1.0, grads[0], Tolerance);
        Assert.Equal(-1.0, grads[1], Tolerance);
        Assert.Equal(0.5, grads[2], Tolerance);
    }

    #endregion

    #region GradientClipping - ClipByNorm

    [Fact]
    public void ClipByNorm_BelowThreshold_Unchanged()
    {
        // Norm = sqrt(0.3^2 + 0.4^2) = sqrt(0.09 + 0.16) = sqrt(0.25) = 0.5
        var grads = new Vector<double>([0.3, 0.4]);
        var clipped = GradientClippingHelper.ClipByNorm(grads, 1.0);

        Assert.NotNull(clipped);
        Assert.Equal(0.3, clipped[0], Tolerance);
        Assert.Equal(0.4, clipped[1], Tolerance);
    }

    [Fact]
    public void ClipByNorm_ExceedsThreshold_ScaledDown()
    {
        // Norm = sqrt(3^2 + 4^2) = sqrt(9+16) = sqrt(25) = 5
        // Scale = 1.0 / 5.0 = 0.2
        // Clipped = [3*0.2, 4*0.2] = [0.6, 0.8]
        var grads = new Vector<double>([3.0, 4.0]);
        var clipped = GradientClippingHelper.ClipByNorm(grads, 1.0);

        Assert.NotNull(clipped);
        Assert.Equal(0.6, clipped[0], Tolerance);
        Assert.Equal(0.8, clipped[1], Tolerance);
    }

    [Fact]
    public void ClipByNorm_PreservesDirection()
    {
        // After clipping, direction should be preserved
        var grads = new Vector<double>([3.0, 4.0]);
        var clipped = GradientClippingHelper.ClipByNorm(grads, 1.0);

        Assert.NotNull(clipped);
        // Ratio should be preserved: clipped[0]/clipped[1] = 3/4
        Assert.Equal(3.0 / 4.0, clipped[0] / clipped[1], Tolerance);
    }

    [Fact]
    public void ClipByNorm_ResultHasMaxNorm()
    {
        var grads = new Vector<double>([3.0, 4.0]);
        double maxNorm = 2.0;
        var clipped = GradientClippingHelper.ClipByNorm(grads, maxNorm);

        Assert.NotNull(clipped);
        double resultNorm = Math.Sqrt(clipped[0] * clipped[0] + clipped[1] * clipped[1]);
        Assert.Equal(maxNorm, resultNorm, Tolerance);
    }

    [Fact]
    public void ClipByNormInPlace_ReturnsTrue_WhenClipped()
    {
        var grads = new Vector<double>([3.0, 4.0]); // norm=5
        bool clipped = GradientClippingHelper.ClipByNormInPlace(grads, 1.0);

        Assert.True(clipped);
        Assert.Equal(0.6, grads[0], Tolerance);
        Assert.Equal(0.8, grads[1], Tolerance);
    }

    [Fact]
    public void ClipByNormInPlace_ReturnsFalse_WhenNotClipped()
    {
        var grads = new Vector<double>([0.3, 0.4]); // norm=0.5
        bool clipped = GradientClippingHelper.ClipByNormInPlace(grads, 1.0);

        Assert.False(clipped);
    }

    #endregion

    #region GradientClipping - ClipByGlobalNorm

    [Fact]
    public void ClipByGlobalNorm_HandCalculated()
    {
        // Two vectors: [3, 4] and [5, 12]
        // Global norm = sqrt(9+16+25+144) = sqrt(194) ≈ 13.928
        // maxNorm = 5
        // Scale = 5 / sqrt(194)
        var g1 = new Vector<double>([3.0, 4.0]);
        var g2 = new Vector<double>([5.0, 12.0]);
        var list = new List<Vector<double>> { g1, g2 };

        var clipped = GradientClippingHelper.ClipByGlobalNorm(list, 5.0);

        Assert.NotNull(clipped);
        double globalNorm = Math.Sqrt(9 + 16 + 25 + 144);
        double scale = 5.0 / globalNorm;

        Assert.Equal(3.0 * scale, clipped[0][0], Tolerance);
        Assert.Equal(4.0 * scale, clipped[0][1], Tolerance);
        Assert.Equal(5.0 * scale, clipped[1][0], Tolerance);
        Assert.Equal(12.0 * scale, clipped[1][1], Tolerance);
    }

    [Fact]
    public void ClipByGlobalNorm_BelowThreshold_Unchanged()
    {
        var g1 = new Vector<double>([0.1, 0.2]);
        var g2 = new Vector<double>([0.3, 0.4]);
        var list = new List<Vector<double>> { g1, g2 };

        var clipped = GradientClippingHelper.ClipByGlobalNorm(list, 5.0);

        Assert.NotNull(clipped);
        Assert.Equal(0.1, clipped[0][0], Tolerance);
        Assert.Equal(0.2, clipped[0][1], Tolerance);
        Assert.Equal(0.3, clipped[1][0], Tolerance);
        Assert.Equal(0.4, clipped[1][1], Tolerance);
    }

    #endregion

    #region GradientClipping - ComputeNorm

    [Fact]
    public void ComputeNorm_HandCalculated()
    {
        // L2 norm of [3, 4] = sqrt(9+16) = 5
        var grads = new Vector<double>([3.0, 4.0]);
        double norm = GradientClippingHelper.ComputeNorm(grads);
        Assert.Equal(5.0, norm, Tolerance);
    }

    [Fact]
    public void ComputeNorm_ZeroVector_IsZero()
    {
        var grads = new Vector<double>([0.0, 0.0, 0.0]);
        double norm = GradientClippingHelper.ComputeNorm(grads);
        Assert.Equal(0.0, norm, Tolerance);
    }

    [Fact]
    public void ComputeGlobalNorm_HandCalculated()
    {
        // Global norm of [[1,2],[3,4]] = sqrt(1+4+9+16) = sqrt(30)
        var g1 = new Vector<double>([1.0, 2.0]);
        var g2 = new Vector<double>([3.0, 4.0]);
        var list = new List<Vector<double>> { g1, g2 };

        double norm = GradientClippingHelper.ComputeGlobalNorm(list);
        Assert.Equal(Math.Sqrt(30.0), norm, Tolerance);
    }

    #endregion

    #region GradientClipping - Adaptive Clipping

    [Fact]
    public void ClipAdaptive_BelowThreshold_Unchanged()
    {
        // Params = [10, 10], param norm = sqrt(200) ≈ 14.14
        // Grads = [0.01, 0.01], grad norm = sqrt(0.0002) ≈ 0.0141
        // Adaptive threshold = 14.14 * 0.01 = 0.1414
        // Grad norm 0.0141 < 0.1414, so unchanged
        var grads = new Vector<double>([0.01, 0.01]);
        var params_ = new Vector<double>([10.0, 10.0]);

        var result = GradientClippingHelper.ClipAdaptive(grads, params_, 0.01);

        Assert.NotNull(result);
        Assert.Equal(0.01, result[0], Tolerance);
        Assert.Equal(0.01, result[1], Tolerance);
    }

    [Fact]
    public void ClipAdaptive_ExceedsThreshold_Scaled()
    {
        // Params = [10, 10], param norm = sqrt(200) ≈ 14.142
        // Grads = [5, 5], grad norm = sqrt(50) ≈ 7.071
        // Adaptive threshold = 14.142 * 0.01 = 0.14142
        // Grad norm 7.071 > 0.14142, so scale = 0.14142 / 7.071 ≈ 0.02
        var grads = new Vector<double>([5.0, 5.0]);
        var params_ = new Vector<double>([10.0, 10.0]);

        var result = GradientClippingHelper.ClipAdaptive(grads, params_, 0.01);

        Assert.NotNull(result);
        double paramNorm = Math.Sqrt(200.0);
        double gradNorm = Math.Sqrt(50.0);
        double threshold = paramNorm * 0.01;
        double scale = threshold / gradNorm;

        Assert.Equal(5.0 * scale, result[0], Tolerance);
        Assert.Equal(5.0 * scale, result[1], Tolerance);
    }

    [Fact]
    public void ClipAdaptive_MinimumThreshold_Respected()
    {
        // Very small params → threshold = paramNorm * ratio
        // If below 1e-3, clamped to 1e-3
        var grads = new Vector<double>([1.0, 1.0]); // norm = sqrt(2) ≈ 1.414
        var params_ = new Vector<double>([1e-10, 1e-10]); // tiny param norm

        var result = GradientClippingHelper.ClipAdaptive(grads, params_, 0.01);

        Assert.NotNull(result);
        // Min threshold = 1e-3, grad norm = 1.414
        // Scale = 1e-3 / 1.414 ≈ 7.07e-4
        double scale = 1e-3 / Math.Sqrt(2.0);
        Assert.Equal(1.0 * scale, result[0], Tolerance);
    }

    [Fact]
    public void ClipAdaptive_MismatchedLengths_Throws()
    {
        var grads = new Vector<double>([1.0, 2.0]);
        var params_ = new Vector<double>([1.0, 2.0, 3.0]);

        Assert.Throws<ArgumentException>(() =>
            GradientClippingHelper.ClipAdaptive(grads, params_, 0.01));
    }

    #endregion

    #region Gradient Explosion/Vanishing Detection

    [Fact]
    public void AreGradientsExploding_LargeNorm_True()
    {
        var grads = new Vector<double>([1e7, 0.0]);
        Assert.True(GradientClippingHelper.AreGradientsExploding(grads));
    }

    [Fact]
    public void AreGradientsExploding_NaN_True()
    {
        var grads = new Vector<double>([1.0, double.NaN]);
        Assert.True(GradientClippingHelper.AreGradientsExploding(grads));
    }

    [Fact]
    public void AreGradientsExploding_Infinity_True()
    {
        var grads = new Vector<double>([1.0, double.PositiveInfinity]);
        Assert.True(GradientClippingHelper.AreGradientsExploding(grads));
    }

    [Fact]
    public void AreGradientsExploding_Normal_False()
    {
        var grads = new Vector<double>([0.5, -0.3, 0.8]);
        Assert.False(GradientClippingHelper.AreGradientsExploding(grads));
    }

    [Fact]
    public void AreGradientsVanishing_TinyNorm_True()
    {
        var grads = new Vector<double>([1e-10, 1e-10]);
        Assert.True(GradientClippingHelper.AreGradientsVanishing(grads));
    }

    [Fact]
    public void AreGradientsVanishing_Normal_False()
    {
        var grads = new Vector<double>([0.5, -0.3]);
        Assert.False(GradientClippingHelper.AreGradientsVanishing(grads));
    }

    [Fact]
    public void AreGradientsVanishing_Null_True()
    {
        Assert.True(GradientClippingHelper.AreGradientsVanishing<double>(null));
    }

    #endregion

    #region Epsilon Constants

    [Fact]
    public void EpsilonConstants_OrderedCorrectly()
    {
        // SmallEpsilon < DefaultEpsilon < LargeEpsilon
        Assert.True(NumericalStabilityHelper.SmallEpsilon < NumericalStabilityHelper.DefaultEpsilon);
        Assert.True(NumericalStabilityHelper.DefaultEpsilon < NumericalStabilityHelper.LargeEpsilon);
    }

    [Fact]
    public void GetEpsilon_DefaultEpsilon()
    {
        double eps = NumericalStabilityHelper.GetEpsilon<double>();
        Assert.Equal(NumericalStabilityHelper.DefaultEpsilon, eps, Tolerance);
    }

    [Fact]
    public void GetEpsilon_CustomEpsilon()
    {
        double eps = NumericalStabilityHelper.GetEpsilon<double>(1e-3);
        Assert.Equal(1e-3, eps, Tolerance);
    }

    #endregion
}
