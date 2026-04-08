using System;
using AiDotNet.ActivationFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Deep mathematical correctness tests for activation functions NOT covered in the first file.
/// Each test verifies hand-calculated expected values against the implementation.
/// Covers: HardSigmoid, HardSwish, HardTanh, CELU, SoftSign, BentIdentity, Gaussian,
/// ISRU, ReLU6, ThresholdedReLU, LiSHT.
/// </summary>
public class ActivationFunctionDeepMathIntegrationTests2
{
    private const double Tolerance = 1e-8;

    #region HardSigmoid

    [Fact]
    public void HardSigmoid_AtZero_Returns0_5()
    {
        // f(0) = max(0, min(1, (0+1)/2)) = max(0, min(1, 0.5)) = 0.5
        var act = new HardSigmoidActivation<double>();
        Assert.Equal(0.5, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void HardSigmoid_AtNeg1_Returns0()
    {
        // f(-1) = max(0, min(1, (-1+1)/2)) = max(0, min(1, 0)) = 0
        var act = new HardSigmoidActivation<double>();
        Assert.Equal(0.0, act.Activate(-1.0), Tolerance);
    }

    [Fact]
    public void HardSigmoid_AtPos1_Returns1()
    {
        // f(1) = max(0, min(1, (1+1)/2)) = max(0, min(1, 1)) = 1
        var act = new HardSigmoidActivation<double>();
        Assert.Equal(1.0, act.Activate(1.0), Tolerance);
    }

    [Fact]
    public void HardSigmoid_SaturatesBelow()
    {
        // f(-5) = max(0, min(1, (-5+1)/2)) = max(0, -2) = 0
        var act = new HardSigmoidActivation<double>();
        Assert.Equal(0.0, act.Activate(-5.0), Tolerance);
    }

    [Fact]
    public void HardSigmoid_SaturatesAbove()
    {
        // f(5) = max(0, min(1, (5+1)/2)) = max(0, min(1, 3)) = 1
        var act = new HardSigmoidActivation<double>();
        Assert.Equal(1.0, act.Activate(5.0), Tolerance);
    }

    [Fact]
    public void HardSigmoid_LinearRegion_HandCalculated()
    {
        // f(0.5) = (0.5+1)/2 = 0.75
        var act = new HardSigmoidActivation<double>();
        Assert.Equal(0.75, act.Activate(0.5), Tolerance);
    }

    [Fact]
    public void HardSigmoid_Derivative_InLinearRegion()
    {
        // For -1 < x < 1: f'(x) = 0.5
        var act = new HardSigmoidActivation<double>();
        Assert.Equal(0.5, act.Derivative(0.0), Tolerance);
        Assert.Equal(0.5, act.Derivative(0.5), Tolerance);
        Assert.Equal(0.5, act.Derivative(-0.5), Tolerance);
    }

    [Fact]
    public void HardSigmoid_Derivative_InSaturatedRegion()
    {
        // For x <= -1 or x >= 1: f'(x) = 0
        var act = new HardSigmoidActivation<double>();
        Assert.Equal(0.0, act.Derivative(-2.0), Tolerance);
        Assert.Equal(0.0, act.Derivative(2.0), Tolerance);
    }

    #endregion

    #region HardSwish

    [Fact]
    public void HardSwish_AtZero_Returns0()
    {
        // f(0) = 0 * min(max(0, 0+3), 6) / 6 = 0 * 3/6 = 0
        var act = new HardSwishActivation<double>();
        Assert.Equal(0.0, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void HardSwish_HandCalculated_PositiveInRange()
    {
        // f(1) = 1 * min(max(0, 1+3), 6) / 6 = 1 * min(4, 6)/6 = 1 * 4/6 = 2/3
        var act = new HardSwishActivation<double>();
        Assert.Equal(2.0 / 3.0, act.Activate(1.0), Tolerance);
    }

    [Fact]
    public void HardSwish_HandCalculated_NegativeInRange()
    {
        // f(-1) = -1 * min(max(0, -1+3), 6) / 6 = -1 * min(2, 6)/6 = -1 * 2/6 = -1/3
        var act = new HardSwishActivation<double>();
        Assert.Equal(-1.0 / 3.0, act.Activate(-1.0), Tolerance);
    }

    [Fact]
    public void HardSwish_BelowNeg3_IsZero()
    {
        // f(-4) = -4 * min(max(0, -4+3), 6)/6 = -4 * min(max(0, -1), 6)/6 = -4 * 0/6 = 0
        var act = new HardSwishActivation<double>();
        Assert.Equal(0.0, act.Activate(-4.0), Tolerance);
    }

    [Fact]
    public void HardSwish_Above3_EqualsIdentity()
    {
        // f(5) = 5 * min(max(0, 5+3), 6)/6 = 5 * min(8, 6)/6 = 5 * 6/6 = 5
        var act = new HardSwishActivation<double>();
        Assert.Equal(5.0, act.Activate(5.0), Tolerance);
    }

    [Fact]
    public void HardSwish_Derivative_BelowNeg3_IsZero()
    {
        var act = new HardSwishActivation<double>();
        Assert.Equal(0.0, act.Derivative(-4.0), Tolerance);
    }

    [Fact]
    public void HardSwish_Derivative_Above3_IsOne()
    {
        var act = new HardSwishActivation<double>();
        Assert.Equal(1.0, act.Derivative(4.0), Tolerance);
    }

    [Fact]
    public void HardSwish_Derivative_InRange_HandCalculated()
    {
        // f'(x) = (2x+3)/6 for -3 < x < 3
        // f'(0) = 3/6 = 0.5
        // f'(1) = 5/6
        // f'(-1) = 1/6
        var act = new HardSwishActivation<double>();
        Assert.Equal(0.5, act.Derivative(0.0), Tolerance);
        Assert.Equal(5.0 / 6.0, act.Derivative(1.0), Tolerance);
        Assert.Equal(1.0 / 6.0, act.Derivative(-1.0), Tolerance);
    }

    [Fact]
    public void HardSwish_NumericalGradient()
    {
        var act = new HardSwishActivation<double>();
        double x = 1.5;
        double eps = 1e-5;
        double numGrad = (act.Activate(x + eps) - act.Activate(x - eps)) / (2 * eps);
        Assert.Equal(numGrad, act.Derivative(x), 1e-4);
    }

    #endregion

    #region HardTanh

    [Fact]
    public void HardTanh_InRange_ReturnsInput()
    {
        // f(0.5) = max(-1, min(1, 0.5)) = 0.5
        var act = new HardTanhActivation<double>();
        Assert.Equal(0.5, act.Activate(0.5), Tolerance);
        Assert.Equal(-0.5, act.Activate(-0.5), Tolerance);
        Assert.Equal(0.0, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void HardTanh_ClipsAbove()
    {
        // f(2) = max(-1, min(1, 2)) = 1
        var act = new HardTanhActivation<double>();
        Assert.Equal(1.0, act.Activate(2.0), Tolerance);
        Assert.Equal(1.0, act.Activate(100.0), Tolerance);
    }

    [Fact]
    public void HardTanh_ClipsBelow()
    {
        // f(-2) = max(-1, min(1, -2)) = -1
        var act = new HardTanhActivation<double>();
        Assert.Equal(-1.0, act.Activate(-2.0), Tolerance);
        Assert.Equal(-1.0, act.Activate(-100.0), Tolerance);
    }

    [Fact]
    public void HardTanh_Derivative_InRange()
    {
        // For -1 < x < 1: f'(x) = 1
        var act = new HardTanhActivation<double>();
        Assert.Equal(1.0, act.Derivative(0.0), Tolerance);
        Assert.Equal(1.0, act.Derivative(0.5), Tolerance);
        Assert.Equal(1.0, act.Derivative(-0.5), Tolerance);
    }

    [Fact]
    public void HardTanh_Derivative_OutOfRange()
    {
        // For |x| >= 1: f'(x) = 0
        var act = new HardTanhActivation<double>();
        Assert.Equal(0.0, act.Derivative(2.0), Tolerance);
        Assert.Equal(0.0, act.Derivative(-2.0), Tolerance);
    }

    [Fact]
    public void HardTanh_IsOddFunction()
    {
        // f(-x) = -f(x) for odd functions
        var act = new HardTanhActivation<double>();
        double[] testValues = { 0.3, 0.7, 1.5, 3.0 };
        foreach (var x in testValues)
        {
            Assert.Equal(-act.Activate(x), act.Activate(-x), Tolerance);
        }
    }

    #endregion

    #region CELU

    [Fact]
    public void CELU_PositiveInput_ReturnsInput()
    {
        // For x >= 0: f(x) = max(0,x) + min(0, a*(exp(x/a)-1)) = x + 0 = x
        var act = new CELUActivation<double>(alpha: 1.0);
        Assert.Equal(2.0, act.Activate(2.0), Tolerance);
        Assert.Equal(0.5, act.Activate(0.5), Tolerance);
    }

    [Fact]
    public void CELU_AtZero_ReturnsZero()
    {
        var act = new CELUActivation<double>(alpha: 1.0);
        Assert.Equal(0.0, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void CELU_NegativeInput_HandCalculated()
    {
        // For x = -1, a = 1: f(-1) = max(0,-1) + min(0, 1*(exp(-1)-1))
        // = 0 + min(0, exp(-1)-1) = exp(-1)-1 = 0.36788... - 1 = -0.63212...
        var act = new CELUActivation<double>(alpha: 1.0);
        Assert.Equal(Math.Exp(-1.0) - 1.0, act.Activate(-1.0), Tolerance);
    }

    [Fact]
    public void CELU_DifferentAlpha_HandCalculated()
    {
        // For x = -2, a = 2: f(-2) = 0 + min(0, 2*(exp(-2/2)-1)) = 2*(exp(-1)-1)
        // = 2*(0.36788... - 1) = 2*(-0.63212...) = -1.26424...
        var act = new CELUActivation<double>(alpha: 2.0);
        double expected = 2.0 * (Math.Exp(-1.0) - 1.0);
        Assert.Equal(expected, act.Activate(-2.0), Tolerance);
    }

    [Fact]
    public void CELU_NegativeSaturation_ApproachesMinusAlpha()
    {
        // As x -> -inf: f(x) -> -alpha
        var act = new CELUActivation<double>(alpha: 1.5);
        double result = act.Activate(-100.0);
        Assert.Equal(-1.5, result, 1e-6);
    }

    [Fact]
    public void CELU_Derivative_Positive_Returns1()
    {
        var act = new CELUActivation<double>(alpha: 1.0);
        Assert.Equal(1.0, act.Derivative(2.0), Tolerance);
        Assert.Equal(1.0, act.Derivative(0.0), Tolerance);
    }

    [Fact]
    public void CELU_Derivative_Negative_HandCalculated()
    {
        // For x < 0: f'(x) = exp(x/a)
        // x = -1, a = 1: f'(-1) = exp(-1) = 0.36788...
        var act = new CELUActivation<double>(alpha: 1.0);
        Assert.Equal(Math.Exp(-1.0), act.Derivative(-1.0), Tolerance);
    }

    [Fact]
    public void CELU_Derivative_NeverZero()
    {
        // Unlike ReLU, CELU derivative is never exactly 0
        var act = new CELUActivation<double>(alpha: 1.0);
        Assert.True(act.Derivative(-10.0) > 0);
        Assert.True(act.Derivative(-100.0) > 0);
    }

    [Fact]
    public void CELU_NumericalGradient()
    {
        var act = new CELUActivation<double>(alpha: 1.0);
        double x = -0.5;
        double eps = 1e-5;
        double numGrad = (act.Activate(x + eps) - act.Activate(x - eps)) / (2 * eps);
        Assert.Equal(numGrad, act.Derivative(x), 1e-4);
    }

    #endregion

    #region SoftSign

    [Fact]
    public void SoftSign_AtZero_Returns0()
    {
        // f(0) = 0 / (1 + 0) = 0
        var act = new SoftSignActivation<double>();
        Assert.Equal(0.0, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void SoftSign_HandCalculated()
    {
        // f(2) = 2 / (1 + 2) = 2/3
        // f(-2) = -2 / (1 + 2) = -2/3
        var act = new SoftSignActivation<double>();
        Assert.Equal(2.0 / 3.0, act.Activate(2.0), Tolerance);
        Assert.Equal(-2.0 / 3.0, act.Activate(-2.0), Tolerance);
    }

    [Fact]
    public void SoftSign_BoundedBetweenNeg1And1()
    {
        var act = new SoftSignActivation<double>();
        Assert.True(act.Activate(1000.0) < 1.0);
        Assert.True(act.Activate(-1000.0) > -1.0);
        Assert.True(act.Activate(1000.0) > 0.999);
        Assert.True(act.Activate(-1000.0) < -0.999);
    }

    [Fact]
    public void SoftSign_IsOddFunction()
    {
        // f(-x) = -f(x)
        var act = new SoftSignActivation<double>();
        double[] testValues = { 0.5, 1.0, 3.0, 10.0 };
        foreach (var x in testValues)
        {
            Assert.Equal(-act.Activate(x), act.Activate(-x), Tolerance);
        }
    }

    [Fact]
    public void SoftSign_Derivative_AtZero_IsMax()
    {
        // f'(0) = 1/(1+0)^2 = 1
        var act = new SoftSignActivation<double>();
        Assert.Equal(1.0, act.Derivative(0.0), Tolerance);
    }

    [Fact]
    public void SoftSign_Derivative_HandCalculated()
    {
        // f'(x) = 1/(1+|x|)^2
        // f'(2) = 1/(1+2)^2 = 1/9
        // f'(-3) = 1/(1+3)^2 = 1/16
        var act = new SoftSignActivation<double>();
        Assert.Equal(1.0 / 9.0, act.Derivative(2.0), Tolerance);
        Assert.Equal(1.0 / 16.0, act.Derivative(-3.0), Tolerance);
    }

    [Fact]
    public void SoftSign_Derivative_AlwaysPositive()
    {
        var act = new SoftSignActivation<double>();
        Assert.True(act.Derivative(-100.0) > 0);
        Assert.True(act.Derivative(100.0) > 0);
    }

    [Fact]
    public void SoftSign_NumericalGradient()
    {
        var act = new SoftSignActivation<double>();
        double x = 1.5;
        double eps = 1e-5;
        double numGrad = (act.Activate(x + eps) - act.Activate(x - eps)) / (2 * eps);
        Assert.Equal(numGrad, act.Derivative(x), 1e-4);
    }

    #endregion

    #region BentIdentity

    [Fact]
    public void BentIdentity_AtZero_ReturnsZero()
    {
        // f(0) = (sqrt(0+1)-1)/2 + 0 = (1-1)/2 + 0 = 0
        var act = new BentIdentityActivation<double>();
        Assert.Equal(0.0, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void BentIdentity_HandCalculated()
    {
        // f(2) = (sqrt(4+1)-1)/2 + 2 = (sqrt(5)-1)/2 + 2
        // sqrt(5) = 2.2360679..., so (2.2360679-1)/2 + 2 = 0.6180339... + 2 = 2.6180339...
        var act = new BentIdentityActivation<double>();
        double expected = (Math.Sqrt(5.0) - 1.0) / 2.0 + 2.0;
        Assert.Equal(expected, act.Activate(2.0), Tolerance);
    }

    [Fact]
    public void BentIdentity_NegativeInput_HandCalculated()
    {
        // f(-2) = (sqrt(4+1)-1)/2 + (-2) = (sqrt(5)-1)/2 - 2
        var act = new BentIdentityActivation<double>();
        double expected = (Math.Sqrt(5.0) - 1.0) / 2.0 - 2.0;
        Assert.Equal(expected, act.Activate(-2.0), Tolerance);
    }

    [Fact]
    public void BentIdentity_ApproximatesIdentity_ForLargePositive()
    {
        // For large x: f(x) ~ x + (|x|-1)/2 ~ x + x/2 - 1/2
        // But more precisely, f(x) - x = (sqrt(x^2+1)-1)/2 ~ (|x|-1)/2 for large x
        var act = new BentIdentityActivation<double>();
        double x = 100.0;
        double result = act.Activate(x);
        // f(x) ~ x + x/2 = 1.5x for large positive x, not exactly
        // Actually f(x) = (sqrt(x^2+1)-1)/2 + x, for large x ~ (x-1)/2 + x = 3x/2 - 1/2
        // But let's just check it's > x
        Assert.True(result > x);
    }

    [Fact]
    public void BentIdentity_Derivative_AtZero()
    {
        // f'(0) = 0/(2*sqrt(1)) + 1 = 0 + 1 = 1
        var act = new BentIdentityActivation<double>();
        Assert.Equal(1.0, act.Derivative(0.0), Tolerance);
    }

    [Fact]
    public void BentIdentity_Derivative_HandCalculated()
    {
        // f'(x) = x/(2*sqrt(x^2+1)) + 1
        // f'(2) = 2/(2*sqrt(5)) + 1 = 1/sqrt(5) + 1 = 0.4472135... + 1 = 1.4472135...
        var act = new BentIdentityActivation<double>();
        double expected = 1.0 / Math.Sqrt(5.0) + 1.0;
        Assert.Equal(expected, act.Derivative(2.0), Tolerance);
    }

    [Fact]
    public void BentIdentity_Derivative_AlwaysGreaterThanHalf()
    {
        // f'(x) = x/(2*sqrt(x^2+1)) + 1
        // The term x/(2*sqrt(x^2+1)) is in (-0.5, 0.5), so f'(x) > 0.5
        var act = new BentIdentityActivation<double>();
        Assert.True(act.Derivative(-100.0) > 0.5);
        Assert.True(act.Derivative(100.0) > 0.5);
        Assert.True(act.Derivative(0.0) > 0.5);
    }

    [Fact]
    public void BentIdentity_NumericalGradient()
    {
        var act = new BentIdentityActivation<double>();
        double x = -1.5;
        double eps = 1e-5;
        double numGrad = (act.Activate(x + eps) - act.Activate(x - eps)) / (2 * eps);
        Assert.Equal(numGrad, act.Derivative(x), 1e-4);
    }

    #endregion

    #region Gaussian

    [Fact]
    public void Gaussian_AtZero_Returns1()
    {
        // f(0) = exp(-0^2) = exp(0) = 1
        var act = new GaussianActivation<double>();
        Assert.Equal(1.0, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void Gaussian_HandCalculated()
    {
        // f(1) = exp(-1) = 0.36788...
        // f(2) = exp(-4) = 0.01832...
        var act = new GaussianActivation<double>();
        Assert.Equal(Math.Exp(-1.0), act.Activate(1.0), Tolerance);
        Assert.Equal(Math.Exp(-4.0), act.Activate(2.0), Tolerance);
    }

    [Fact]
    public void Gaussian_IsEvenFunction()
    {
        // f(-x) = exp(-(-x)^2) = exp(-x^2) = f(x)
        var act = new GaussianActivation<double>();
        double[] testValues = { 0.5, 1.0, 2.0, 3.0 };
        foreach (var x in testValues)
        {
            Assert.Equal(act.Activate(x), act.Activate(-x), Tolerance);
        }
    }

    [Fact]
    public void Gaussian_AlwaysPositive()
    {
        var act = new GaussianActivation<double>();
        Assert.True(act.Activate(-10.0) > 0);
        Assert.True(act.Activate(10.0) > 0);
    }

    [Fact]
    public void Gaussian_ApproachesZero_ForLargeInput()
    {
        var act = new GaussianActivation<double>();
        Assert.True(act.Activate(5.0) < 1e-10);
        Assert.True(act.Activate(-5.0) < 1e-10);
    }

    [Fact]
    public void Gaussian_Derivative_AtZero_IsZero()
    {
        // f'(0) = -2*0*exp(0) = 0
        var act = new GaussianActivation<double>();
        Assert.Equal(0.0, act.Derivative(0.0), Tolerance);
    }

    [Fact]
    public void Gaussian_Derivative_HandCalculated()
    {
        // f'(x) = -2x*exp(-x^2)
        // f'(1) = -2*1*exp(-1) = -2*0.36788... = -0.73576...
        var act = new GaussianActivation<double>();
        Assert.Equal(-2.0 * Math.Exp(-1.0), act.Derivative(1.0), Tolerance);
    }

    [Fact]
    public void Gaussian_Derivative_IsOdd()
    {
        // f'(-x) = -2(-x)*exp(-x^2) = 2x*exp(-x^2) = -f'(x)
        var act = new GaussianActivation<double>();
        Assert.Equal(-act.Derivative(1.5), act.Derivative(-1.5), Tolerance);
    }

    [Fact]
    public void Gaussian_NumericalGradient()
    {
        var act = new GaussianActivation<double>();
        double x = 0.7;
        double eps = 1e-5;
        double numGrad = (act.Activate(x + eps) - act.Activate(x - eps)) / (2 * eps);
        Assert.Equal(numGrad, act.Derivative(x), 1e-4);
    }

    #endregion

    #region ISRU

    [Fact]
    public void ISRU_AtZero_Returns0()
    {
        // f(0) = 0 / sqrt(1 + 0) = 0
        var act = new ISRUActivation<double>(alpha: 1.0);
        Assert.Equal(0.0, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void ISRU_HandCalculated()
    {
        // f(1) = 1 / sqrt(1 + 1*1^2) = 1 / sqrt(2) = 0.70710...
        var act = new ISRUActivation<double>(alpha: 1.0);
        Assert.Equal(1.0 / Math.Sqrt(2.0), act.Activate(1.0), Tolerance);
    }

    [Fact]
    public void ISRU_NegativeInput()
    {
        // f(-1) = -1 / sqrt(1 + 1) = -1/sqrt(2)
        var act = new ISRUActivation<double>(alpha: 1.0);
        Assert.Equal(-1.0 / Math.Sqrt(2.0), act.Activate(-1.0), Tolerance);
    }

    [Fact]
    public void ISRU_DifferentAlpha()
    {
        // f(2) with a=0.5: 2 / sqrt(1 + 0.5*4) = 2 / sqrt(3) = 1.1547...
        var act = new ISRUActivation<double>(alpha: 0.5);
        Assert.Equal(2.0 / Math.Sqrt(3.0), act.Activate(2.0), Tolerance);
    }

    [Fact]
    public void ISRU_BoundedBetweenNeg1And1()
    {
        // As x -> inf: f(x) -> 1/sqrt(a), so bounded by 1/sqrt(a)
        // For a=1: bounded between -1 and 1
        var act = new ISRUActivation<double>(alpha: 1.0);
        Assert.True(act.Activate(100.0) < 1.0);
        Assert.True(act.Activate(-100.0) > -1.0);
    }

    [Fact]
    public void ISRU_IsOddFunction()
    {
        var act = new ISRUActivation<double>(alpha: 1.0);
        Assert.Equal(-act.Activate(2.0), act.Activate(-2.0), Tolerance);
    }

    [Fact]
    public void ISRU_Derivative_AtZero_Is1()
    {
        // f'(0) = (1 + 0)^(-3/2) = 1
        var act = new ISRUActivation<double>(alpha: 1.0);
        Assert.Equal(1.0, act.Derivative(0.0), Tolerance);
    }

    [Fact]
    public void ISRU_Derivative_HandCalculated()
    {
        // f'(x) = (1 + a*x^2)^(-3/2)
        // f'(1) with a=1: (1+1)^(-3/2) = 2^(-1.5) = 1/(2*sqrt(2)) = 0.35355...
        var act = new ISRUActivation<double>(alpha: 1.0);
        Assert.Equal(Math.Pow(2.0, -1.5), act.Derivative(1.0), Tolerance);
    }

    [Fact]
    public void ISRU_NumericalGradient()
    {
        var act = new ISRUActivation<double>(alpha: 1.0);
        double x = 0.8;
        double eps = 1e-5;
        double numGrad = (act.Activate(x + eps) - act.Activate(x - eps)) / (2 * eps);
        Assert.Equal(numGrad, act.Derivative(x), 1e-4);
    }

    #endregion

    #region ReLU6

    [Fact]
    public void ReLU6_NegativeInput_Returns0()
    {
        var act = new ReLU6Activation<double>();
        Assert.Equal(0.0, act.Activate(-1.0), Tolerance);
        Assert.Equal(0.0, act.Activate(-100.0), Tolerance);
    }

    [Fact]
    public void ReLU6_InRange_ReturnsInput()
    {
        var act = new ReLU6Activation<double>();
        Assert.Equal(3.0, act.Activate(3.0), Tolerance);
        Assert.Equal(0.5, act.Activate(0.5), Tolerance);
        Assert.Equal(5.9, act.Activate(5.9), Tolerance);
    }

    [Fact]
    public void ReLU6_ClipsAt6()
    {
        var act = new ReLU6Activation<double>();
        Assert.Equal(6.0, act.Activate(6.0), Tolerance);
        Assert.Equal(6.0, act.Activate(7.0), Tolerance);
        Assert.Equal(6.0, act.Activate(100.0), Tolerance);
    }

    [Fact]
    public void ReLU6_Derivative_InRange()
    {
        // For 0 < x < 6: f'(x) = 1
        var act = new ReLU6Activation<double>();
        Assert.Equal(1.0, act.Derivative(3.0), Tolerance);
        Assert.Equal(1.0, act.Derivative(0.5), Tolerance);
    }

    [Fact]
    public void ReLU6_Derivative_OutOfRange()
    {
        // For x <= 0 or x >= 6: f'(x) = 0
        var act = new ReLU6Activation<double>();
        Assert.Equal(0.0, act.Derivative(-1.0), Tolerance);
        Assert.Equal(0.0, act.Derivative(7.0), Tolerance);
    }

    #endregion

    #region ThresholdedReLU

    [Fact]
    public void ThresholdedReLU_AboveThreshold_ReturnsInput()
    {
        // theta=1 (default), x=2: f(2) = 2 (since 2 > 1)
        var act = new ThresholdedReLUActivation<double>(theta: 1.0);
        Assert.Equal(2.0, act.Activate(2.0), Tolerance);
    }

    [Fact]
    public void ThresholdedReLU_BelowThreshold_Returns0()
    {
        // theta=1, x=0.5: f(0.5) = 0 (since 0.5 <= 1)
        var act = new ThresholdedReLUActivation<double>(theta: 1.0);
        Assert.Equal(0.0, act.Activate(0.5), Tolerance);
    }

    [Fact]
    public void ThresholdedReLU_AtThreshold_Returns0()
    {
        // theta=1, x=1: f(1) = 0 (since 1 is NOT > 1)
        var act = new ThresholdedReLUActivation<double>(theta: 1.0);
        Assert.Equal(0.0, act.Activate(1.0), Tolerance);
    }

    [Fact]
    public void ThresholdedReLU_NegativeInput_Returns0()
    {
        var act = new ThresholdedReLUActivation<double>(theta: 1.0);
        Assert.Equal(0.0, act.Activate(-5.0), Tolerance);
    }

    [Fact]
    public void ThresholdedReLU_CustomThreshold()
    {
        var act = new ThresholdedReLUActivation<double>(theta: 3.0);
        Assert.Equal(0.0, act.Activate(2.0), Tolerance);
        Assert.Equal(0.0, act.Activate(3.0), Tolerance);
        Assert.Equal(4.0, act.Activate(4.0), Tolerance);
    }

    [Fact]
    public void ThresholdedReLU_Derivative_AboveThreshold()
    {
        var act = new ThresholdedReLUActivation<double>(theta: 1.0);
        Assert.Equal(1.0, act.Derivative(2.0), Tolerance);
    }

    [Fact]
    public void ThresholdedReLU_Derivative_BelowThreshold()
    {
        var act = new ThresholdedReLUActivation<double>(theta: 1.0);
        Assert.Equal(0.0, act.Derivative(0.5), Tolerance);
        Assert.Equal(0.0, act.Derivative(-1.0), Tolerance);
    }

    #endregion

    #region LiSHT

    [Fact]
    public void LiSHT_AtZero_Returns0()
    {
        // f(0) = 0 * tanh(0) = 0 * 0 = 0
        var act = new LiSHTActivation<double>();
        Assert.Equal(0.0, act.Activate(0.0), Tolerance);
    }

    [Fact]
    public void LiSHT_HandCalculated()
    {
        // f(1) = 1 * tanh(1) = tanh(1) = 0.76159...
        var act = new LiSHTActivation<double>();
        Assert.Equal(Math.Tanh(1.0), act.Activate(1.0), Tolerance);
    }

    [Fact]
    public void LiSHT_IsEvenFunction()
    {
        // f(-x) = (-x) * tanh(-x) = (-x)(-tanh(x)) = x*tanh(x) = f(x)
        var act = new LiSHTActivation<double>();
        Assert.Equal(act.Activate(2.0), act.Activate(-2.0), Tolerance);
        Assert.Equal(act.Activate(0.5), act.Activate(-0.5), Tolerance);
    }

    [Fact]
    public void LiSHT_AlwaysNonNegative()
    {
        // Since f(x) = x*tanh(x) and sign(x) = sign(tanh(x)), product >= 0
        var act = new LiSHTActivation<double>();
        Assert.True(act.Activate(-5.0) >= 0);
        Assert.True(act.Activate(5.0) >= 0);
        Assert.True(act.Activate(0.0) >= 0);
    }

    [Fact]
    public void LiSHT_LargePositive_ApproachesInput()
    {
        // For large x: tanh(x) -> 1, so f(x) -> x
        var act = new LiSHTActivation<double>();
        Assert.Equal(10.0, act.Activate(10.0), 1e-4);
    }

    [Fact]
    public void LiSHT_Derivative_AtZero()
    {
        // f'(0) = tanh(0) + 0*(1-tanh^2(0)) = 0 + 0 = 0
        var act = new LiSHTActivation<double>();
        Assert.Equal(0.0, act.Derivative(0.0), Tolerance);
    }

    [Fact]
    public void LiSHT_Derivative_HandCalculated()
    {
        // f'(x) = tanh(x) + x*(1-tanh^2(x))
        // f'(1) = tanh(1) + 1*(1-tanh^2(1))
        // tanh(1) = 0.76159..., tanh^2(1) = 0.58002..., 1-tanh^2(1) = 0.41997...
        // f'(1) = 0.76159 + 0.41997 = 1.18157...
        var act = new LiSHTActivation<double>();
        double tanhVal = Math.Tanh(1.0);
        double expected = tanhVal + 1.0 * (1.0 - tanhVal * tanhVal);
        Assert.Equal(expected, act.Derivative(1.0), Tolerance);
    }

    [Fact]
    public void LiSHT_NumericalGradient()
    {
        var act = new LiSHTActivation<double>();
        double x = 1.5;
        double eps = 1e-5;
        double numGrad = (act.Activate(x + eps) - act.Activate(x - eps)) / (2 * eps);
        Assert.Equal(numGrad, act.Derivative(x), 1e-4);
    }

    #endregion

    #region Cross-Activation Relationships

    [Fact]
    public void HardSigmoid_ApproximatesSigmoid_NearZero()
    {
        // Hard sigmoid is a piecewise linear approximation of sigmoid
        // They should be close near x=0
        var hs = new HardSigmoidActivation<double>();
        var s = new SigmoidActivation<double>();
        // Both should give 0.5 at x=0
        Assert.Equal(0.5, hs.Activate(0.0), Tolerance);
        Assert.Equal(0.5, s.Activate(0.0), Tolerance);
    }

    [Fact]
    public void HardTanh_ClipsLikeReLU_MinusReLUNeg()
    {
        // HardTanh = max(-1, min(1, x)) = min(1, max(-1, x))
        // This is clip(-1, 1, x) which ReLU cannot do
        var ht = new HardTanhActivation<double>();
        Assert.Equal(0.5, ht.Activate(0.5), Tolerance);
        Assert.Equal(1.0, ht.Activate(5.0), Tolerance);
        Assert.Equal(-1.0, ht.Activate(-5.0), Tolerance);
    }

    [Fact]
    public void CELU_Alpha1_MatchesELU_Alpha1()
    {
        // CELU with alpha=1 is identical to ELU with alpha=1
        var celu = new CELUActivation<double>(alpha: 1.0);
        var elu = new ELUActivation<double>(alpha: 1.0);

        double[] inputs = { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };
        foreach (var x in inputs)
        {
            Assert.Equal(elu.Activate(x), celu.Activate(x), 1e-6);
        }
    }

    [Fact]
    public void SoftSign_SlowerSaturation_ThanTanh()
    {
        // SoftSign has polynomial tails (slower saturation) vs tanh's exponential tails
        // At x=2: softsign = 2/3 = 0.667, tanh(2) = 0.964
        // tanh is closer to 1 (faster saturation)
        var ss = new SoftSignActivation<double>();
        var tanh = new TanhActivation<double>();
        double sVal = ss.Activate(2.0);
        double tVal = tanh.Activate(2.0);
        Assert.True(tVal > sVal); // tanh saturates faster
    }

    [Fact]
    public void ReLU6_SubsetOfReLU()
    {
        // ReLU6(x) = min(6, ReLU(x))
        // So ReLU6 <= ReLU always
        var relu6 = new ReLU6Activation<double>();
        var relu = new ReLUActivation<double>();

        double[] inputs = { -2, 0, 3, 6, 10 };
        foreach (var x in inputs)
        {
            Assert.True(relu6.Activate(x) <= relu.Activate(x) + 1e-10);
        }
    }

    #endregion
}
