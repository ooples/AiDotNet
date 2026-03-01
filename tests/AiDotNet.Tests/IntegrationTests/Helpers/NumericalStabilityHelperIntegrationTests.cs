using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for NumericalStabilityHelper:
/// SafeLog, SafeDiv, SafeSqrt, ClampProbability, SafeLogProbability,
/// IsNaN, IsInfinity, IsFinite, ContainsNaN, ContainsInfinity,
/// ReplaceNaN, ReplaceInfinity, ReplaceNonFinite, StableSoftmax,
/// StableLogSoftmax, CountNaN, CountInfinity, AssertFinite.
/// </summary>
public class NumericalStabilityHelperIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Constants

    [Fact]
    public void DefaultEpsilon_IsSmallPositive()
    {
        Assert.Equal(1e-7, NumericalStabilityHelper.DefaultEpsilon);
    }

    [Fact]
    public void SmallEpsilon_IsSmaller()
    {
        Assert.True(NumericalStabilityHelper.SmallEpsilon < NumericalStabilityHelper.DefaultEpsilon);
        Assert.Equal(1e-15, NumericalStabilityHelper.SmallEpsilon);
    }

    [Fact]
    public void LargeEpsilon_IsLarger()
    {
        Assert.True(NumericalStabilityHelper.LargeEpsilon > NumericalStabilityHelper.DefaultEpsilon);
        Assert.Equal(1e-5, NumericalStabilityHelper.LargeEpsilon);
    }

    #endregion

    #region GetEpsilon

    [Fact]
    public void GetEpsilon_Default_ReturnsDefaultEpsilon()
    {
        double eps = NumericalStabilityHelper.GetEpsilon<double>();
        Assert.Equal(NumericalStabilityHelper.DefaultEpsilon, eps, 1e-15);
    }

    [Fact]
    public void GetEpsilon_Custom_ReturnsCustomValue()
    {
        double eps = NumericalStabilityHelper.GetEpsilon<double>(1e-3);
        Assert.Equal(1e-3, eps, 1e-15);
    }

    #endregion

    #region SafeLog

    [Fact]
    public void SafeLog_PositiveValue_ReturnsCorrectLog()
    {
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
    public void SafeLog_Zero_ReturnsFiniteValue()
    {
        double result = NumericalStabilityHelper.SafeLog(0.0);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsNegativeInfinity(result));
        Assert.True(result < 0, "SafeLog(0) should be negative (log of small epsilon)");
    }

    [Fact]
    public void SafeLog_NegativeValue_ReturnsFiniteValue()
    {
        double result = NumericalStabilityHelper.SafeLog(-5.0);
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region SafeDiv

    [Fact]
    public void SafeDiv_NormalDivision_ReturnsCorrectResult()
    {
        double result = NumericalStabilityHelper.SafeDiv(10.0, 2.0);
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void SafeDiv_ZeroDenominator_ReturnsFiniteValue()
    {
        double result = NumericalStabilityHelper.SafeDiv(1.0, 0.0);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void SafeDiv_VerySmallDenominator_ReturnsFiniteValue()
    {
        double result = NumericalStabilityHelper.SafeDiv(1.0, 1e-20);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void SafeDiv_ZeroNumerator_ReturnsZero()
    {
        double result = NumericalStabilityHelper.SafeDiv(0.0, 5.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region SafeSqrt

    [Fact]
    public void SafeSqrt_PositiveValue_ReturnsCorrectResult()
    {
        double result = NumericalStabilityHelper.SafeSqrt(4.0);
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void SafeSqrt_Zero_ReturnsFiniteValue()
    {
        double result = NumericalStabilityHelper.SafeSqrt(0.0);
        Assert.False(double.IsNaN(result));
        Assert.True(result > 0, "SafeSqrt(0) should be positive (sqrt of epsilon)");
    }

    [Fact]
    public void SafeSqrt_NegativeValue_ReturnsFiniteValue()
    {
        double result = NumericalStabilityHelper.SafeSqrt(-1.0);
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region ClampProbability

    [Fact]
    public void ClampProbability_ValidProbability_Unchanged()
    {
        double result = NumericalStabilityHelper.ClampProbability(0.5);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void ClampProbability_Zero_ClampsToEpsilon()
    {
        double result = NumericalStabilityHelper.ClampProbability(0.0);
        Assert.True(result > 0, "Clamped probability should be positive");
        Assert.True(result < 0.01, "Clamped probability should be small");
    }

    [Fact]
    public void ClampProbability_One_ClampsToLessThanOne()
    {
        double result = NumericalStabilityHelper.ClampProbability(1.0);
        Assert.True(result < 1.0, "Clamped probability should be less than 1");
        Assert.True(result > 0.99, "Clamped probability should be close to 1");
    }

    [Fact]
    public void ClampProbability_Negative_ClampsToEpsilon()
    {
        double result = NumericalStabilityHelper.ClampProbability(-0.5);
        Assert.True(result > 0);
    }

    [Fact]
    public void ClampProbability_GreaterThanOne_ClampsToLessThanOne()
    {
        double result = NumericalStabilityHelper.ClampProbability(1.5);
        Assert.True(result < 1.0);
    }

    #endregion

    #region SafeLogProbability

    [Fact]
    public void SafeLogProbability_ValidProbability_ReturnsNegative()
    {
        double result = NumericalStabilityHelper.SafeLogProbability(0.5);
        Assert.True(result < 0, "log(0.5) should be negative");
    }

    [Fact]
    public void SafeLogProbability_Zero_ReturnsFinite()
    {
        double result = NumericalStabilityHelper.SafeLogProbability(0.0);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsNegativeInfinity(result));
    }

    [Fact]
    public void SafeLogProbability_One_ReturnsNearZero()
    {
        double result = NumericalStabilityHelper.SafeLogProbability(1.0);
        Assert.True(Math.Abs(result) < 0.001, $"log(~1) should be near zero, got {result}");
    }

    #endregion

    #region IsNaN / IsInfinity / IsFinite

    [Fact]
    public void IsNaN_NaN_ReturnsTrue()
    {
        Assert.True(NumericalStabilityHelper.IsNaN(double.NaN));
    }

    [Fact]
    public void IsNaN_Normal_ReturnsFalse()
    {
        Assert.False(NumericalStabilityHelper.IsNaN(1.0));
    }

    [Fact]
    public void IsInfinity_PositiveInfinity_ReturnsTrue()
    {
        Assert.True(NumericalStabilityHelper.IsInfinity(double.PositiveInfinity));
    }

    [Fact]
    public void IsInfinity_NegativeInfinity_ReturnsTrue()
    {
        Assert.True(NumericalStabilityHelper.IsInfinity(double.NegativeInfinity));
    }

    [Fact]
    public void IsInfinity_Normal_ReturnsFalse()
    {
        Assert.False(NumericalStabilityHelper.IsInfinity(1.0));
    }

    [Fact]
    public void IsFinite_Normal_ReturnsTrue()
    {
        Assert.True(NumericalStabilityHelper.IsFinite(42.0));
    }

    [Fact]
    public void IsFinite_NaN_ReturnsFalse()
    {
        Assert.False(NumericalStabilityHelper.IsFinite(double.NaN));
    }

    [Fact]
    public void IsFinite_Infinity_ReturnsFalse()
    {
        Assert.False(NumericalStabilityHelper.IsFinite(double.PositiveInfinity));
    }

    #endregion

    #region ContainsNaN / ContainsInfinity / ContainsNonFinite (Vector)

    [Fact]
    public void ContainsNaN_Vector_CleanVector_ReturnsFalse()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        Assert.False(NumericalStabilityHelper.ContainsNaN(v));
    }

    [Fact]
    public void ContainsNaN_Vector_WithNaN_ReturnsTrue()
    {
        var v = new Vector<double>(new double[] { 1.0, double.NaN, 3.0 });
        Assert.True(NumericalStabilityHelper.ContainsNaN(v));
    }

    [Fact]
    public void ContainsInfinity_Vector_WithInfinity_ReturnsTrue()
    {
        var v = new Vector<double>(new double[] { 1.0, double.PositiveInfinity, 3.0 });
        Assert.True(NumericalStabilityHelper.ContainsInfinity(v));
    }

    [Fact]
    public void ContainsNonFinite_Vector_WithBoth_ReturnsTrue()
    {
        var v = new Vector<double>(new double[] { double.NaN, 2.0, double.NegativeInfinity });
        Assert.True(NumericalStabilityHelper.ContainsNonFinite(v));
    }

    [Fact]
    public void ContainsNonFinite_Vector_Clean_ReturnsFalse()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        Assert.False(NumericalStabilityHelper.ContainsNonFinite(v));
    }

    #endregion

    #region ContainsNaN / ContainsInfinity / ContainsNonFinite (Tensor)

    [Fact]
    public void ContainsNaN_Tensor_CleanTensor_ReturnsFalse()
    {
        var t = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        Assert.False(NumericalStabilityHelper.ContainsNaN(t));
    }

    [Fact]
    public void ContainsNaN_Tensor_WithNaN_ReturnsTrue()
    {
        var t = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, double.NaN, 3.0 }));
        Assert.True(NumericalStabilityHelper.ContainsNaN(t));
    }

    [Fact]
    public void ContainsInfinity_Tensor_WithInfinity_ReturnsTrue()
    {
        var t = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, double.PositiveInfinity, 3.0 }));
        Assert.True(NumericalStabilityHelper.ContainsInfinity(t));
    }

    #endregion

    #region ReplaceNaN / ReplaceInfinity / ReplaceNonFinite

    [Fact]
    public void ReplaceNaN_ReplacesNaNWithZero()
    {
        var v = new Vector<double>(new double[] { 1.0, double.NaN, 3.0 });
        var result = NumericalStabilityHelper.ReplaceNaN(v);
        Assert.NotNull(result);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void ReplaceNaN_CustomReplacement()
    {
        var v = new Vector<double>(new double[] { double.NaN, 2.0 });
        var result = NumericalStabilityHelper.ReplaceNaN(v, -1.0);
        Assert.NotNull(result);
        Assert.Equal(-1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
    }

    [Fact]
    public void ReplaceNaN_Null_ReturnsNull()
    {
        var result = NumericalStabilityHelper.ReplaceNaN<double>(null);
        Assert.Null(result);
    }

    [Fact]
    public void ReplaceInfinity_ReplacesInfWithZero()
    {
        var v = new Vector<double>(new double[] { 1.0, double.PositiveInfinity, double.NegativeInfinity });
        var result = NumericalStabilityHelper.ReplaceInfinity(v);
        Assert.NotNull(result);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(0.0, result[2], Tolerance);
    }

    [Fact]
    public void ReplaceNonFinite_ReplacesBoth()
    {
        var v = new Vector<double>(new double[] { double.NaN, 2.0, double.PositiveInfinity });
        var result = NumericalStabilityHelper.ReplaceNonFinite(v);
        Assert.NotNull(result);
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(0.0, result[2], Tolerance);
    }

    #endregion

    #region CountNaN / CountInfinity

    [Fact]
    public void CountNaN_NoNaN_ReturnsZero()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        Assert.Equal(0, NumericalStabilityHelper.CountNaN(v));
    }

    [Fact]
    public void CountNaN_MultipleNaN_ReturnsCount()
    {
        var v = new Vector<double>(new double[] { double.NaN, 2.0, double.NaN, 4.0 });
        Assert.Equal(2, NumericalStabilityHelper.CountNaN(v));
    }

    [Fact]
    public void CountInfinity_MultipleInf_ReturnsCount()
    {
        var v = new Vector<double>(new double[] { double.PositiveInfinity, 2.0, double.NegativeInfinity });
        Assert.Equal(2, NumericalStabilityHelper.CountInfinity(v));
    }

    #endregion

    #region StableSoftmax

    [Fact]
    public void StableSoftmax_SumsToOne()
    {
        var logits = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var result = NumericalStabilityHelper.StableSoftmax(logits);
        Assert.NotNull(result);

        double sum = 0;
        for (int i = 0; i < result.Length; i++)
            sum += result[i];
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void StableSoftmax_HighestLogitGetsHighestProbability()
    {
        var logits = new Vector<double>(new double[] { 1.0, 5.0, 2.0 });
        var result = NumericalStabilityHelper.StableSoftmax(logits);
        Assert.NotNull(result);
        Assert.True(result[1] > result[0], "Highest logit should have highest probability");
        Assert.True(result[1] > result[2], "Highest logit should have highest probability");
    }

    [Fact]
    public void StableSoftmax_EqualLogits_EqualProbabilities()
    {
        var logits = new Vector<double>(new double[] { 3.0, 3.0, 3.0 });
        var result = NumericalStabilityHelper.StableSoftmax(logits);
        Assert.NotNull(result);
        Assert.Equal(1.0 / 3.0, result[0], Tolerance);
        Assert.Equal(1.0 / 3.0, result[1], Tolerance);
        Assert.Equal(1.0 / 3.0, result[2], Tolerance);
    }

    [Fact]
    public void StableSoftmax_LargeValues_NoOverflow()
    {
        var logits = new Vector<double>(new double[] { 1000.0, 1001.0, 1002.0 });
        var result = NumericalStabilityHelper.StableSoftmax(logits);
        Assert.NotNull(result);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"Softmax result[{i}] should not be NaN");
            Assert.False(double.IsInfinity(result[i]), $"Softmax result[{i}] should not be Infinity");
        }

        double sum = 0;
        for (int i = 0; i < result.Length; i++)
            sum += result[i];
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void StableSoftmax_Null_ReturnsNull()
    {
        var result = NumericalStabilityHelper.StableSoftmax<double>(null);
        Assert.Null(result);
    }

    #endregion

    #region StableLogSoftmax

    [Fact]
    public void StableLogSoftmax_AllNegative()
    {
        var logits = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var result = NumericalStabilityHelper.StableLogSoftmax(logits);
        Assert.NotNull(result);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] <= 0, $"Log-softmax values should be non-positive, got {result[i]}");
        }
    }

    [Fact]
    public void StableLogSoftmax_ExponentiatesToSoftmax()
    {
        var logits = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var logSoftmax = NumericalStabilityHelper.StableLogSoftmax(logits);
        var softmax = NumericalStabilityHelper.StableSoftmax(logits);
        Assert.NotNull(logSoftmax);
        Assert.NotNull(softmax);

        for (int i = 0; i < logits.Length; i++)
        {
            Assert.Equal(Math.Log(softmax[i]), logSoftmax[i], 1e-4);
        }
    }

    [Fact]
    public void StableLogSoftmax_LargeValues_NoOverflow()
    {
        var logits = new Vector<double>(new double[] { 500.0, 501.0, 502.0 });
        var result = NumericalStabilityHelper.StableLogSoftmax(logits);
        Assert.NotNull(result);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
            Assert.False(double.IsPositiveInfinity(result[i]));
        }
    }

    #endregion

    #region AssertFinite

    [Fact]
    public void AssertFinite_Value_FiniteValue_DoesNotThrow()
    {
        NumericalStabilityHelper.AssertFinite(42.0);
    }

    [Fact]
    public void AssertFinite_Value_NaN_Throws()
    {
        Assert.Throws<ArgumentException>(() => NumericalStabilityHelper.AssertFinite(double.NaN));
    }

    [Fact]
    public void AssertFinite_Value_Infinity_Throws()
    {
        Assert.Throws<ArgumentException>(() => NumericalStabilityHelper.AssertFinite(double.PositiveInfinity));
    }

    [Fact]
    public void AssertFinite_Vector_AllFinite_DoesNotThrow()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        NumericalStabilityHelper.AssertFinite(v);
    }

    [Fact]
    public void AssertFinite_Vector_WithNaN_Throws()
    {
        var v = new Vector<double>(new double[] { 1.0, double.NaN, 3.0 });
        Assert.Throws<ArgumentException>(() => NumericalStabilityHelper.AssertFinite(v));
    }

    [Fact]
    public void AssertFinite_Tensor_AllFinite_DoesNotThrow()
    {
        var t = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        NumericalStabilityHelper.AssertFinite(t);
    }

    [Fact]
    public void AssertFinite_Tensor_WithInfinity_Throws()
    {
        var t = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, double.PositiveInfinity, 3.0 }));
        Assert.Throws<ArgumentException>(() => NumericalStabilityHelper.AssertFinite(t));
    }

    #endregion
}
