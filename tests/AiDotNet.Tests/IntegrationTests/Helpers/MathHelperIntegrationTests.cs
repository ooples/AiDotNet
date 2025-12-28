using Xunit;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for MathHelper to verify mathematical operations work correctly.
/// These tests validate the core mathematical functions used throughout the library.
/// </summary>
public class MathHelperIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const float FloatTolerance = 1e-5f;

    #region GetNumericOperations Tests

    [Fact]
    public void GetNumericOperations_Double_ReturnsValidOperations()
    {
        var ops = MathHelper.GetNumericOperations<double>();

        Assert.NotNull(ops);
        Assert.Equal(2.0, ops.Add(1.0, 1.0));
        Assert.Equal(6.0, ops.Multiply(2.0, 3.0));
    }

    [Fact]
    public void GetNumericOperations_Float_ReturnsValidOperations()
    {
        var ops = MathHelper.GetNumericOperations<float>();

        Assert.NotNull(ops);
        Assert.Equal(2.0f, ops.Add(1.0f, 1.0f));
        Assert.Equal(6.0f, ops.Multiply(2.0f, 3.0f));
    }

    [Fact]
    public void GetNumericOperations_Int_ReturnsValidOperations()
    {
        var ops = MathHelper.GetNumericOperations<int>();

        Assert.NotNull(ops);
        Assert.Equal(5, ops.Add(2, 3));
        Assert.Equal(12, ops.Multiply(3, 4));
    }

    #endregion

    #region Type Check Tests

    [Fact]
    public void IsFloatingPoint_Double_ReturnsTrue()
    {
        Assert.True(MathHelper.IsFloatingPoint<double>());
    }

    [Fact]
    public void IsFloatingPoint_Float_ReturnsTrue()
    {
        Assert.True(MathHelper.IsFloatingPoint<float>());
    }

    [Fact]
    public void IsFloatingPoint_Int_ReturnsFalse()
    {
        Assert.False(MathHelper.IsFloatingPoint<int>());
    }

    [Fact]
    public void IsIntegerType_Int_ReturnsTrue()
    {
        Assert.True(MathHelper.IsIntegerType<int>());
    }

    [Fact]
    public void IsIntegerType_Long_ReturnsTrue()
    {
        Assert.True(MathHelper.IsIntegerType<long>());
    }

    [Fact]
    public void IsIntegerType_Double_ReturnsFalse()
    {
        Assert.False(MathHelper.IsIntegerType<double>());
    }

    #endregion

    #region Clamp Tests

    [Fact]
    public void Clamp_ValueWithinRange_ReturnsValue()
    {
        double result = MathHelper.Clamp(5.0, 0.0, 10.0);
        Assert.Equal(5.0, result);
    }

    [Fact]
    public void Clamp_ValueBelowMin_ReturnsMin()
    {
        double result = MathHelper.Clamp(-5.0, 0.0, 10.0);
        Assert.Equal(0.0, result);
    }

    [Fact]
    public void Clamp_ValueAboveMax_ReturnsMax()
    {
        double result = MathHelper.Clamp(15.0, 0.0, 10.0);
        Assert.Equal(10.0, result);
    }

    [Fact]
    public void Clamp_Float_WorksCorrectly()
    {
        float result = MathHelper.Clamp(0.5f, 0.0f, 1.0f);
        Assert.Equal(0.5f, result);
    }

    [Fact]
    public void Clamp_Int_WorksCorrectly()
    {
        int result = MathHelper.Clamp(50, 0, 100);
        Assert.Equal(50, result);
    }

    #endregion

    #region Trigonometric Functions

    [Fact]
    public void Sin_ZeroAngle_ReturnsZero()
    {
        double result = MathHelper.Sin(0.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void Sin_PiOverTwo_ReturnsOne()
    {
        double result = MathHelper.Sin(Math.PI / 2);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void Cos_ZeroAngle_ReturnsOne()
    {
        double result = MathHelper.Cos(0.0);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void Cos_Pi_ReturnsNegativeOne()
    {
        double result = MathHelper.Cos(Math.PI);
        Assert.True(Math.Abs(result + 1.0) < Tolerance);
    }

    [Fact]
    public void Tanh_Zero_ReturnsZero()
    {
        double result = MathHelper.Tanh(0.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void Tanh_LargeValue_ReturnsNearOne()
    {
        double result = MathHelper.Tanh(10.0);
        Assert.True(Math.Abs(result - 1.0) < 0.001);
    }

    [Fact]
    public void ArcSin_Zero_ReturnsZero()
    {
        double result = MathHelper.ArcSin(0.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void ArcSin_One_ReturnsPiOverTwo()
    {
        double result = MathHelper.ArcSin(1.0);
        Assert.True(Math.Abs(result - Math.PI / 2) < Tolerance);
    }

    [Fact]
    public void ArcCos_One_ReturnsZero()
    {
        double result = MathHelper.ArcCos(1.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void ArcCos_Zero_ReturnsPiOverTwo()
    {
        double result = MathHelper.ArcCos(0.0);
        Assert.True(Math.Abs(result - Math.PI / 2) < Tolerance);
    }

    [Fact]
    public void ArcTan_Zero_ReturnsZero()
    {
        double result = MathHelper.ArcTan(0.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void ArcTan_One_ReturnsPiOverFour()
    {
        double result = MathHelper.ArcTan(1.0);
        Assert.True(Math.Abs(result - Math.PI / 4) < Tolerance);
    }

    #endregion

    #region Atanh Tests

    [Fact]
    public void Atanh_Zero_ReturnsZero()
    {
        double result = MathHelper.Atanh(0.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void Atanh_Half_ReturnsCorrectValue()
    {
        double result = MathHelper.Atanh(0.5);
        double expected = 0.5 * Math.Log((1 + 0.5) / (1 - 0.5));
        Assert.True(Math.Abs(result - expected) < Tolerance);
    }

    [Fact]
    public void Atanh_NearOne_ReturnsLargeValue()
    {
        double result = MathHelper.Atanh(0.99);
        Assert.True(result > 2.0); // Atanh(0.99) ≈ 2.65
    }

    #endregion

    #region Bessel Functions

    [Fact]
    public void BesselI0_Zero_ReturnsOne()
    {
        double result = MathHelper.BesselI0(0.0);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void BesselI0_PositiveValue_ReturnsPositive()
    {
        double result = MathHelper.BesselI0(2.0);
        Assert.True(result > 1.0); // I0(x) > 1 for x > 0
    }

    [Fact]
    public void BesselJ_OrderZero_Zero_ReturnsOne()
    {
        double result = MathHelper.BesselJ(0.0, 0.0);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void BesselJ_OrderOne_Zero_ReturnsZero()
    {
        double result = MathHelper.BesselJ(1.0, 0.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void BesselK_PositiveValue_ReturnsPositive()
    {
        double result = MathHelper.BesselK(0.0, 1.0);
        Assert.True(result > 0); // K_nu(x) > 0 for x > 0
    }

    #endregion

    #region Gamma and Factorial

    [Fact]
    public void Gamma_One_ReturnsOne()
    {
        double result = MathHelper.Gamma(1.0);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void Gamma_Two_ReturnsOne()
    {
        // Gamma(2) = 1! = 1
        double result = MathHelper.Gamma(2.0);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void Gamma_Three_ReturnsTwo()
    {
        // Gamma(3) = 2! = 2
        double result = MathHelper.Gamma(3.0);
        Assert.True(Math.Abs(result - 2.0) < Tolerance);
    }

    [Fact]
    public void Gamma_Half_ReturnsSqrtPi()
    {
        // Gamma(0.5) = sqrt(π)
        double result = MathHelper.Gamma(0.5);
        double expected = Math.Sqrt(Math.PI);
        Assert.True(Math.Abs(result - expected) < 0.01);
    }

    [Fact]
    public void Factorial_Zero_ReturnsOne()
    {
        double result = MathHelper.Factorial<double>(0);
        Assert.Equal(1.0, result);
    }

    [Fact]
    public void Factorial_Five_Returns120()
    {
        double result = MathHelper.Factorial<double>(5);
        Assert.Equal(120.0, result);
    }

    [Fact]
    public void Factorial_Ten_Returns3628800()
    {
        double result = MathHelper.Factorial<double>(10);
        Assert.Equal(3628800.0, result);
    }

    #endregion

    #region Pi and Constants

    [Fact]
    public void Pi_Double_ReturnsCorrectValue()
    {
        double result = MathHelper.Pi<double>();
        Assert.True(Math.Abs(result - Math.PI) < Tolerance);
    }

    [Fact]
    public void Pi_Float_ReturnsCorrectValue()
    {
        float result = MathHelper.Pi<float>();
        Assert.True(Math.Abs(result - (float)Math.PI) < FloatTolerance);
    }

    #endregion

    #region Reciprocal Tests

    [Fact]
    public void Reciprocal_Two_ReturnsHalf()
    {
        double result = MathHelper.Reciprocal(2.0);
        Assert.Equal(0.5, result);
    }

    [Fact]
    public void Reciprocal_Half_ReturnsTwo()
    {
        double result = MathHelper.Reciprocal(0.5);
        Assert.Equal(2.0, result);
    }

    [Fact]
    public void Reciprocal_Float_WorksCorrectly()
    {
        float result = MathHelper.Reciprocal(4.0f);
        Assert.Equal(0.25f, result);
    }

    #endregion

    #region Sinc Tests

    [Fact]
    public void Sinc_Zero_ReturnsOne()
    {
        // sinc(0) = 1 by definition (limit)
        double result = MathHelper.Sinc(0.0);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void Sinc_Pi_ReturnsZero()
    {
        // sinc(π) = sin(π)/π = 0
        double result = MathHelper.Sinc(Math.PI);
        Assert.True(Math.Abs(result) < 0.01);
    }

    #endregion

    #region Modulo Tests

    [Fact]
    public void Modulo_PositiveValues_ReturnsRemainder()
    {
        double result = MathHelper.Modulo(7.0, 3.0);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void Modulo_Float_WorksCorrectly()
    {
        float result = MathHelper.Modulo(5.5f, 2.0f);
        Assert.True(Math.Abs(result - 1.5f) < FloatTolerance);
    }

    #endregion

    #region IsInteger Tests

    [Fact]
    public void IsInteger_WholeNumber_ReturnsTrue()
    {
        Assert.True(MathHelper.IsInteger(5.0));
    }

    [Fact]
    public void IsInteger_DecimalNumber_ReturnsFalse()
    {
        Assert.False(MathHelper.IsInteger(5.5));
    }

    [Fact]
    public void IsInteger_Zero_ReturnsTrue()
    {
        Assert.True(MathHelper.IsInteger(0.0));
    }

    [Fact]
    public void IsInteger_NegativeWhole_ReturnsTrue()
    {
        Assert.True(MathHelper.IsInteger(-3.0));
    }

    #endregion

    #region Sigmoid Tests

    [Fact]
    public void Sigmoid_Zero_ReturnsHalf()
    {
        double result = MathHelper.Sigmoid(0.0);
        Assert.True(Math.Abs(result - 0.5) < Tolerance);
    }

    [Fact]
    public void Sigmoid_LargePositive_ReturnsNearOne()
    {
        double result = MathHelper.Sigmoid(10.0);
        Assert.True(result > 0.999);
    }

    [Fact]
    public void Sigmoid_LargeNegative_ReturnsNearZero()
    {
        double result = MathHelper.Sigmoid(-10.0);
        Assert.True(result < 0.001);
    }

    [Fact]
    public void Sigmoid_OutputRange_BetweenZeroAndOne()
    {
        for (double x = -5; x <= 5; x += 0.5)
        {
            double result = MathHelper.Sigmoid(x);
            Assert.True(result > 0 && result < 1, $"Sigmoid({x}) = {result} should be in (0, 1)");
        }
    }

    #endregion

    #region AlmostEqual Tests

    [Fact]
    public void AlmostEqual_ExactlyEqual_ReturnsTrue()
    {
        Assert.True(MathHelper.AlmostEqual(1.0, 1.0));
    }

    [Fact]
    public void AlmostEqual_VeryClose_ReturnsTrue()
    {
        Assert.True(MathHelper.AlmostEqual(1.0, 1.0 + 1e-15));
    }

    [Fact]
    public void AlmostEqual_Different_ReturnsFalse()
    {
        Assert.False(MathHelper.AlmostEqual(1.0, 2.0));
    }

    [Fact]
    public void AlmostEqual_WithTolerance_ReturnsTrue()
    {
        Assert.True(MathHelper.AlmostEqual(1.0, 1.1, 0.2));
    }

    [Fact]
    public void AlmostEqual_WithTolerance_ReturnsFalse()
    {
        Assert.False(MathHelper.AlmostEqual(1.0, 1.5, 0.1));
    }

    #endregion

    #region Log2 Tests

    [Fact]
    public void Log2_One_ReturnsZero()
    {
        double result = MathHelper.Log2(1.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void Log2_Two_ReturnsOne()
    {
        double result = MathHelper.Log2(2.0);
        Assert.True(Math.Abs(result - 1.0) < Tolerance);
    }

    [Fact]
    public void Log2_Eight_ReturnsThree()
    {
        double result = MathHelper.Log2(8.0);
        Assert.True(Math.Abs(result - 3.0) < Tolerance);
    }

    [Fact]
    public void Log2_PowerOfTwo_ReturnsExact()
    {
        double result = MathHelper.Log2(1024.0);
        Assert.True(Math.Abs(result - 10.0) < Tolerance);
    }

    #endregion

    #region Min/Max Tests

    [Fact]
    public void Min_FirstSmaller_ReturnsFirst()
    {
        double result = MathHelper.Min(1.0, 2.0);
        Assert.Equal(1.0, result);
    }

    [Fact]
    public void Min_SecondSmaller_ReturnsSecond()
    {
        double result = MathHelper.Min(3.0, 2.0);
        Assert.Equal(2.0, result);
    }

    [Fact]
    public void Max_FirstLarger_ReturnsFirst()
    {
        double result = MathHelper.Max(5.0, 3.0);
        Assert.Equal(5.0, result);
    }

    [Fact]
    public void Max_SecondLarger_ReturnsSecond()
    {
        double result = MathHelper.Max(2.0, 4.0);
        Assert.Equal(4.0, result);
    }

    #endregion

    #region Erf (Error Function) Tests

    [Fact]
    public void Erf_Zero_ReturnsZero()
    {
        double result = MathHelper.Erf(0.0);
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void Erf_LargePositive_ReturnsNearOne()
    {
        double result = MathHelper.Erf(3.0);
        Assert.True(Math.Abs(result - 1.0) < 0.01);
    }

    [Fact]
    public void Erf_LargeNegative_ReturnsNearNegativeOne()
    {
        double result = MathHelper.Erf(-3.0);
        Assert.True(Math.Abs(result + 1.0) < 0.01);
    }

    [Fact]
    public void Erf_OddFunction_NegatesWithInput()
    {
        double positive = MathHelper.Erf(1.5);
        double negative = MathHelper.Erf(-1.5);
        Assert.True(Math.Abs(positive + negative) < Tolerance);
    }

    #endregion

    #region GetNormalRandom Tests

    [Fact]
    public void GetNormalRandom_ZeroStdDev_ReturnsMean()
    {
        // With stdDev = 0, result should always be the mean
        var random = new Random(42);
        double result = MathHelper.GetNormalRandom(5.0, 0.0, random);
        Assert.Equal(5.0, result);
    }

    [Fact]
    public void GetNormalRandom_GeneratesVaried_WithNonZeroStdDev()
    {
        var random = new Random(42);
        var values = new List<double>();

        for (int i = 0; i < 100; i++)
        {
            values.Add(MathHelper.GetNormalRandom(0.0, 1.0, random));
        }

        // Check that we get varied values
        var distinct = values.Distinct().Count();
        Assert.True(distinct > 50, "Should generate varied random values");
    }

    [Fact]
    public void GetNormalRandom_WithSeed_IsReproducible()
    {
        var random1 = new Random(123);
        var random2 = new Random(123);

        double value1 = MathHelper.GetNormalRandom(0.0, 1.0, random1);
        double value2 = MathHelper.GetNormalRandom(0.0, 1.0, random2);

        Assert.Equal(value1, value2);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void Sin_Float_WorksCorrectly()
    {
        float result = MathHelper.Sin(0.0f);
        Assert.True(Math.Abs(result) < FloatTolerance);
    }

    [Fact]
    public void Cos_Float_WorksCorrectly()
    {
        float result = MathHelper.Cos(0.0f);
        Assert.True(Math.Abs(result - 1.0f) < FloatTolerance);
    }

    [Fact]
    public void Sigmoid_Float_WorksCorrectly()
    {
        float result = MathHelper.Sigmoid(0.0f);
        Assert.True(Math.Abs(result - 0.5f) < FloatTolerance);
    }

    [Fact]
    public void Erf_Float_WorksCorrectly()
    {
        float result = MathHelper.Erf(0.0f);
        Assert.True(Math.Abs(result) < FloatTolerance);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Factorial_NegativeInput_ThrowsOrReturnsSpecialValue()
    {
        // Depending on implementation, this might throw or return NaN/special value
        try
        {
            double result = MathHelper.Factorial<double>(-1);
            // If it doesn't throw, result should be handled gracefully
            Assert.True(double.IsNaN(result) || double.IsInfinity(result) || result >= 0);
        }
        catch (ArgumentException)
        {
            // Expected for invalid input
            Assert.True(true);
        }
    }

    [Fact]
    public void Reciprocal_Zero_ThrowsDivideByZeroException()
    {
        // MathHelper.Reciprocal properly throws exception for zero input
        Assert.Throws<DivideByZeroException>(() => MathHelper.Reciprocal(0.0));
    }

    [Fact]
    public void Clamp_MinEqualsMax_ReturnsMinMax()
    {
        double result = MathHelper.Clamp(5.0, 3.0, 3.0);
        Assert.Equal(3.0, result);
    }

    #endregion
}
