using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Deep mathematical integration tests for activation functions.
/// Tests hand-calculated values, numerical gradient verification, and mathematical properties.
/// </summary>
public class ActivationFunctionDeepMathIntegrationTests
{
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();
    private const double Tol = 1e-8;

    // ========================================================================
    // Sigmoid
    // ========================================================================

    [Fact]
    public void Sigmoid_AtZero_Equals0_5()
    {
        var sigmoid = new SigmoidActivation<double>();
        double result = sigmoid.Activate(0.0);
        Assert.Equal(0.5, result, Tol);
    }

    [Fact]
    public void Sigmoid_HandCalculated_MatchesFormula()
    {
        // sigmoid(1) = 1 / (1 + e^(-1))
        var sigmoid = new SigmoidActivation<double>();
        double result = sigmoid.Activate(1.0);
        double expected = 1.0 / (1.0 + Math.Exp(-1.0));
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void Sigmoid_Symmetry_SigmoidX_Plus_SigmoidNegX_Equals1()
    {
        // sigma(x) + sigma(-x) = 1
        var sigmoid = new SigmoidActivation<double>();
        double[] testValues = [-3, -1, 0, 0.5, 2, 5];
        foreach (double x in testValues)
        {
            double pos = sigmoid.Activate(x);
            double neg = sigmoid.Activate(-x);
            Assert.Equal(1.0, pos + neg, Tol);
        }
    }

    [Fact]
    public void Sigmoid_Derivative_EqualsSigmoidTimesOneMinusSigmoid()
    {
        var sigmoid = new SigmoidActivation<double>();
        double[] testValues = [-2, -1, 0, 1, 2];
        foreach (double x in testValues)
        {
            double s = sigmoid.Activate(x);
            double analytical = sigmoid.Derivative(x);
            double expected = s * (1 - s);
            Assert.Equal(expected, analytical, Tol);
        }
    }

    [Fact]
    public void Sigmoid_Derivative_MaxAtZero_Equals0_25()
    {
        var sigmoid = new SigmoidActivation<double>();
        double derAtZero = sigmoid.Derivative(0.0);
        Assert.Equal(0.25, derAtZero, Tol);
    }

    [Fact]
    public void Sigmoid_NumericalGradient_MatchesAnalytical()
    {
        var sigmoid = new SigmoidActivation<double>();
        double[] testValues = [-3, -1, 0, 1, 3];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (sigmoid.Activate(x + h) - sigmoid.Activate(x - h)) / (2 * h);
            double analytical = sigmoid.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-5);
        }
    }

    // ========================================================================
    // Tanh
    // ========================================================================

    [Fact]
    public void Tanh_AtZero_Equals0()
    {
        var tanh = new TanhActivation<double>();
        double result = tanh.Activate(0.0);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void Tanh_IsOddFunction()
    {
        // tanh(-x) = -tanh(x)
        var tanh = new TanhActivation<double>();
        double[] testValues = [0.5, 1, 2, 5];
        foreach (double x in testValues)
        {
            double pos = tanh.Activate(x);
            double neg = tanh.Activate(-x);
            Assert.Equal(-pos, neg, Tol);
        }
    }

    [Fact]
    public void Tanh_OutputBoundedBetweenMinus1And1()
    {
        var tanh = new TanhActivation<double>();
        double[] testValues = [-100, -10, -1, 0, 1, 10, 100];
        foreach (double x in testValues)
        {
            double result = tanh.Activate(x);
            Assert.True(result >= -1.0 && result <= 1.0, $"tanh({x})={result} out of [-1,1]");
        }
    }

    [Fact]
    public void Tanh_Derivative_Equals1MinusTanhSquared()
    {
        var tanh = new TanhActivation<double>();
        double[] testValues = [-2, -0.5, 0, 0.5, 2];
        foreach (double x in testValues)
        {
            double t = tanh.Activate(x);
            double analytical = tanh.Derivative(x);
            double expected = 1 - t * t;
            Assert.Equal(expected, analytical, Tol);
        }
    }

    [Fact]
    public void Tanh_Derivative_AtZero_Equals1()
    {
        var tanh = new TanhActivation<double>();
        double derAtZero = tanh.Derivative(0.0);
        Assert.Equal(1.0, derAtZero, Tol);
    }

    [Fact]
    public void Tanh_NumericalGradient_MatchesAnalytical()
    {
        var tanh = new TanhActivation<double>();
        double[] testValues = [-2, -0.5, 0, 0.5, 2];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (tanh.Activate(x + h) - tanh.Activate(x - h)) / (2 * h);
            double analytical = tanh.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-5);
        }
    }

    [Fact]
    public void Tanh_Equals_2Sigmoid2x_Minus1()
    {
        // tanh(x) = 2*sigmoid(2x) - 1
        var tanh = new TanhActivation<double>();
        var sigmoid = new SigmoidActivation<double>();
        double[] testValues = [-3, -1, 0, 1, 3];
        foreach (double x in testValues)
        {
            double tanhVal = tanh.Activate(x);
            double fromSigmoid = 2 * sigmoid.Activate(2 * x) - 1;
            Assert.Equal(fromSigmoid, tanhVal, Tol);
        }
    }

    // ========================================================================
    // ReLU
    // ========================================================================

    [Fact]
    public void ReLU_PositiveInput_ReturnsInput()
    {
        var relu = new ReLUActivation<double>();
        Assert.Equal(5.0, relu.Activate(5.0), Tol);
        Assert.Equal(0.001, relu.Activate(0.001), Tol);
    }

    [Fact]
    public void ReLU_NegativeInput_ReturnsZero()
    {
        var relu = new ReLUActivation<double>();
        Assert.Equal(0.0, relu.Activate(-5.0), Tol);
        Assert.Equal(0.0, relu.Activate(-0.001), Tol);
    }

    [Fact]
    public void ReLU_Derivative_PositiveInput_Returns1()
    {
        var relu = new ReLUActivation<double>();
        Assert.Equal(1.0, relu.Derivative(5.0), Tol);
        Assert.Equal(1.0, relu.Derivative(0.001), Tol);
    }

    [Fact]
    public void ReLU_Derivative_NegativeInput_Returns0()
    {
        var relu = new ReLUActivation<double>();
        Assert.Equal(0.0, relu.Derivative(-5.0), Tol);
        Assert.Equal(0.0, relu.Derivative(-0.001), Tol);
    }

    [Fact]
    public void ReLU_IsHomogeneous_ForPositive()
    {
        // ReLU(ax) = a*ReLU(x) for a > 0
        var relu = new ReLUActivation<double>();
        double x = 3.0;
        double a = 2.5;
        Assert.Equal(a * relu.Activate(x), relu.Activate(a * x), Tol);
    }

    // ========================================================================
    // Leaky ReLU
    // ========================================================================

    [Fact]
    public void LeakyReLU_PositiveInput_ReturnsInput()
    {
        var lrelu = new LeakyReLUActivation<double>(alpha: 0.1);
        Assert.Equal(5.0, lrelu.Activate(5.0), Tol);
    }

    [Fact]
    public void LeakyReLU_NegativeInput_ReturnsAlphaTimesInput()
    {
        var lrelu = new LeakyReLUActivation<double>(alpha: 0.1);
        Assert.Equal(-0.5, lrelu.Activate(-5.0), Tol);
    }

    [Fact]
    public void LeakyReLU_Alpha0_EqualsReLU()
    {
        var relu = new ReLUActivation<double>();
        var lrelu = new LeakyReLUActivation<double>(alpha: 0.0);
        double[] testValues = [-5, -1, 0, 1, 5];
        foreach (double x in testValues)
        {
            Assert.Equal(relu.Activate(x), lrelu.Activate(x), Tol);
        }
    }

    [Fact]
    public void LeakyReLU_Derivative_HandCalculated()
    {
        var lrelu = new LeakyReLUActivation<double>(alpha: 0.2);
        Assert.Equal(1.0, lrelu.Derivative(3.0), Tol);  // positive
        Assert.Equal(0.2, lrelu.Derivative(-3.0), Tol); // negative
    }

    [Fact]
    public void LeakyReLU_NumericalGradient_MatchesAnalytical()
    {
        var lrelu = new LeakyReLUActivation<double>(alpha: 0.1);
        // Avoid testing at x=0 where the derivative is technically undefined
        double[] testValues = [-3, -1, -0.5, 0.5, 1, 3];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (lrelu.Activate(x + h) - lrelu.Activate(x - h)) / (2 * h);
            double analytical = lrelu.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-5);
        }
    }

    // ========================================================================
    // ELU
    // ========================================================================

    [Fact]
    public void ELU_PositiveInput_ReturnsInput()
    {
        var elu = new ELUActivation<double>(alpha: 1.0);
        Assert.Equal(5.0, elu.Activate(5.0), Tol);
    }

    [Fact]
    public void ELU_NegativeInput_HandCalculated()
    {
        // ELU(-1, alpha=1) = 1 * (e^(-1) - 1) = 1/e - 1 = -0.63212...
        var elu = new ELUActivation<double>(alpha: 1.0);
        double expected = Math.Exp(-1) - 1;
        Assert.Equal(expected, elu.Activate(-1.0), Tol);
    }

    [Fact]
    public void ELU_NegativeSaturation_ApproachesMinusAlpha()
    {
        double alpha = 2.0;
        var elu = new ELUActivation<double>(alpha: alpha);
        // For very negative x: ELU(x) -> -alpha
        double result = elu.Activate(-100.0);
        Assert.Equal(-alpha, result, 1e-6);
    }

    [Fact]
    public void ELU_Derivative_Positive_Returns1()
    {
        var elu = new ELUActivation<double>(alpha: 1.0);
        Assert.Equal(1.0, elu.Derivative(5.0), Tol);
    }

    [Fact]
    public void ELU_Derivative_Negative_HandCalculated()
    {
        // ELU derivative for x < 0: ELU(x) + alpha = alpha*(e^x - 1) + alpha = alpha*e^x
        var elu = new ELUActivation<double>(alpha: 1.0);
        double x = -1.0;
        double analytical = elu.Derivative(x);
        double expected = Math.Exp(-1.0); // alpha * e^x = 1 * e^(-1)
        Assert.Equal(expected, analytical, Tol);
    }

    [Fact]
    public void ELU_NumericalGradient_MatchesAnalytical()
    {
        var elu = new ELUActivation<double>(alpha: 1.5);
        double[] testValues = [-3, -1, -0.5, 0.5, 1, 3];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (elu.Activate(x + h) - elu.Activate(x - h)) / (2 * h);
            double analytical = elu.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-5);
        }
    }

    // ========================================================================
    // SELU
    // ========================================================================

    [Fact]
    public void SELU_PositiveInput_ReturnsLambdaTimesInput()
    {
        var selu = new SELUActivation<double>();
        double lambda = 1.0507009873554804934193349852946;
        Assert.Equal(lambda * 3.0, selu.Activate(3.0), Tol);
    }

    [Fact]
    public void SELU_AtZero_ReturnsZero()
    {
        var selu = new SELUActivation<double>();
        Assert.Equal(0.0, selu.Activate(0.0), Tol);
    }

    [Fact]
    public void SELU_NegativeInput_HandCalculated()
    {
        // SELU(-1) = lambda * alpha * (e^(-1) - 1)
        var selu = new SELUActivation<double>();
        double lambda = 1.0507009873554804934193349852946;
        double alpha = 1.6732632423543772848170429916717;
        double expected = lambda * alpha * (Math.Exp(-1) - 1);
        Assert.Equal(expected, selu.Activate(-1.0), Tol);
    }

    [Fact]
    public void SELU_Derivative_Positive_ReturnsLambda()
    {
        var selu = new SELUActivation<double>();
        double lambda = 1.0507009873554804934193349852946;
        Assert.Equal(lambda, selu.Derivative(1.0), Tol);
    }

    [Fact]
    public void SELU_Derivative_Negative_HandCalculated()
    {
        // SELU'(x < 0) = lambda * alpha * e^x
        var selu = new SELUActivation<double>();
        double lambda = 1.0507009873554804934193349852946;
        double alpha = 1.6732632423543772848170429916717;
        double x = -1.0;
        double expected = lambda * alpha * Math.Exp(x);
        Assert.Equal(expected, selu.Derivative(x), Tol);
    }

    [Fact]
    public void SELU_NumericalGradient_MatchesAnalytical()
    {
        var selu = new SELUActivation<double>();
        double[] testValues = [-3, -1, -0.5, 0.5, 1, 3];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (selu.Activate(x + h) - selu.Activate(x - h)) / (2 * h);
            double analytical = selu.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-5);
        }
    }

    // ========================================================================
    // Swish (SiLU)
    // ========================================================================

    [Fact]
    public void Swish_EqualsX_TimesSigmoidX()
    {
        var swish = new SwishActivation<double>();
        var sigmoid = new SigmoidActivation<double>();
        double[] testValues = [-3, -1, 0, 1, 3];
        foreach (double x in testValues)
        {
            double swishVal = swish.Activate(x);
            double expected = x * sigmoid.Activate(x);
            Assert.Equal(expected, swishVal, Tol);
        }
    }

    [Fact]
    public void Swish_AtZero_ReturnsZero()
    {
        var swish = new SwishActivation<double>();
        Assert.Equal(0.0, swish.Activate(0.0), Tol);
    }

    [Fact]
    public void Swish_HandCalculated()
    {
        // Swish(1) = 1 * sigmoid(1) = 1 / (1 + e^(-1))
        var swish = new SwishActivation<double>();
        double expected = 1.0 / (1.0 + Math.Exp(-1.0));
        Assert.Equal(expected, swish.Activate(1.0), Tol);
    }

    [Fact]
    public void Swish_Derivative_HandCalculated()
    {
        // Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        var swish = new SwishActivation<double>();
        double x = 1.0;
        double s = 1.0 / (1.0 + Math.Exp(-x));
        double expected = s + x * s * (1 - s);
        Assert.Equal(expected, swish.Derivative(x), Tol);
    }

    [Fact]
    public void Swish_NumericalGradient_MatchesAnalytical()
    {
        var swish = new SwishActivation<double>();
        double[] testValues = [-3, -1, 0, 1, 3];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (swish.Activate(x + h) - swish.Activate(x - h)) / (2 * h);
            double analytical = swish.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-5);
        }
    }

    [Fact]
    public void Swish_HasNegativeRegion()
    {
        // Swish has a small negative region around x ≈ -1.28
        var swish = new SwishActivation<double>();
        double val = swish.Activate(-1.5);
        Assert.True(val < 0, $"Swish(-1.5)={val} should be negative");
    }

    // ========================================================================
    // GELU
    // ========================================================================

    [Fact]
    public void GELU_AtZero_ReturnsZero()
    {
        var gelu = new GELUActivation<double>();
        Assert.Equal(0.0, gelu.Activate(0.0), Tol);
    }

    [Fact]
    public void GELU_LargePositive_ApproachesInput()
    {
        var gelu = new GELUActivation<double>();
        double x = 10.0;
        double result = gelu.Activate(x);
        // GELU(x) ≈ x for large positive x
        Assert.Equal(x, result, 1e-4);
    }

    [Fact]
    public void GELU_LargeNegative_ApproachesZero()
    {
        var gelu = new GELUActivation<double>();
        double result = gelu.Activate(-10.0);
        Assert.Equal(0.0, result, 1e-4);
    }

    [Fact]
    public void GELU_HandCalculated_AtOne()
    {
        // GELU(1) = 0.5 * 1 * (1 + tanh(sqrt(2/pi) * (1 + 0.044715)))
        var gelu = new GELUActivation<double>();
        double sqrt2OverPi = Math.Sqrt(2.0 / Math.PI);
        double inner = 1.0 + 0.044715; // x + 0.044715*x^3 for x=1
        double expected = 0.5 * 1.0 * (1.0 + Math.Tanh(sqrt2OverPi * inner));
        Assert.Equal(expected, gelu.Activate(1.0), Tol);
    }

    [Fact]
    public void GELU_NumericalGradient_MatchesAnalytical()
    {
        var gelu = new GELUActivation<double>();
        double[] testValues = [-2, -1, 0, 1, 2];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (gelu.Activate(x + h) - gelu.Activate(x - h)) / (2 * h);
            double analytical = gelu.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-4);
        }
    }

    // ========================================================================
    // SoftPlus
    // ========================================================================

    [Fact]
    public void SoftPlus_HandCalculated_AtZero()
    {
        // SoftPlus(0) = ln(1 + 1) = ln(2)
        var sp = new SoftPlusActivation<double>();
        Assert.Equal(Math.Log(2), sp.Activate(0.0), Tol);
    }

    [Fact]
    public void SoftPlus_AlwaysPositive()
    {
        var sp = new SoftPlusActivation<double>();
        double[] testValues = [-100, -10, -1, 0, 1, 10, 100];
        foreach (double x in testValues)
        {
            Assert.True(sp.Activate(x) > 0, $"SoftPlus({x}) should be positive");
        }
    }

    [Fact]
    public void SoftPlus_Derivative_IsSigmoid()
    {
        // d/dx SoftPlus(x) = sigmoid(x)
        var sp = new SoftPlusActivation<double>();
        var sigmoid = new SigmoidActivation<double>();
        double[] testValues = [-3, -1, 0, 1, 3];
        foreach (double x in testValues)
        {
            Assert.Equal(sigmoid.Activate(x), sp.Derivative(x), Tol);
        }
    }

    [Fact]
    public void SoftPlus_NumericalGradient_MatchesAnalytical()
    {
        var sp = new SoftPlusActivation<double>();
        double[] testValues = [-3, -1, 0, 1, 3];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (sp.Activate(x + h) - sp.Activate(x - h)) / (2 * h);
            double analytical = sp.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-5);
        }
    }

    [Fact]
    public void SoftPlus_LargePositive_ApproachesInput()
    {
        // For large x: softplus(x) ≈ x
        var sp = new SoftPlusActivation<double>();
        double x = 50.0;
        Assert.Equal(x, sp.Activate(x), 1e-6);
    }

    [Fact]
    public void SoftPlus_GreaterThanReLU()
    {
        // SoftPlus(x) >= ReLU(x) for all x, with equality at +-inf
        var sp = new SoftPlusActivation<double>();
        var relu = new ReLUActivation<double>();
        double[] testValues = [-5, -1, 0, 1, 5];
        foreach (double x in testValues)
        {
            Assert.True(sp.Activate(x) >= relu.Activate(x) - 1e-10,
                $"SoftPlus({x})={sp.Activate(x)} should be >= ReLU({x})={relu.Activate(x)}");
        }
    }

    // ========================================================================
    // Mish
    // ========================================================================

    [Fact]
    public void Mish_AtZero_ReturnsZero()
    {
        var mish = new MishActivation<double>();
        Assert.Equal(0.0, mish.Activate(0.0), Tol);
    }

    [Fact]
    public void Mish_HandCalculated_AtOne()
    {
        // Mish(1) = 1 * tanh(softplus(1)) = 1 * tanh(ln(1+e))
        var mish = new MishActivation<double>();
        double softplus1 = Math.Log(1 + Math.E);
        double expected = 1.0 * Math.Tanh(softplus1);
        Assert.Equal(expected, mish.Activate(1.0), Tol);
    }

    [Fact]
    public void Mish_LargePositive_ApproachesInput()
    {
        // For large x: softplus(x) ≈ x, tanh(x) ≈ 1, so Mish(x) ≈ x
        var mish = new MishActivation<double>();
        double x = 20.0;
        Assert.Equal(x, mish.Activate(x), 1e-3);
    }

    [Fact]
    public void Mish_HasNegativeRegion()
    {
        // Like Swish, Mish allows some negative values
        var mish = new MishActivation<double>();
        double val = mish.Activate(-1.0);
        Assert.True(val < 0, $"Mish(-1)={val} should be negative");
    }

    [Fact]
    public void Mish_NumericalGradient_MatchesAnalytical()
    {
        var mish = new MishActivation<double>();
        double[] testValues = [-2, -1, 0, 1, 2];
        double h = 1e-7;

        foreach (double x in testValues)
        {
            double numerical = (mish.Activate(x + h) - mish.Activate(x - h)) / (2 * h);
            double analytical = mish.Derivative(x);
            Assert.Equal(numerical, analytical, 1e-4);
        }
    }

    // ========================================================================
    // Cross-activation relationships
    // ========================================================================

    [Fact]
    public void ELU_WithAlpha0_EqualsReLU_ForNonZero()
    {
        var elu = new ELUActivation<double>(alpha: 0.0);
        var relu = new ReLUActivation<double>();
        double[] testValues = [-5, -1, 0.5, 1, 5];
        foreach (double x in testValues)
        {
            Assert.Equal(relu.Activate(x), elu.Activate(x), Tol);
        }
    }

    [Fact]
    public void GELU_BetweenReLU_AndIdentity_ForPositive()
    {
        // For x > 0: 0 < GELU(x) <= x (GELU dampens positive values slightly for small x)
        var gelu = new GELUActivation<double>();
        double[] testValues = [0.5, 1, 2, 5, 10];
        foreach (double x in testValues)
        {
            double result = gelu.Activate(x);
            Assert.True(result > 0, $"GELU({x}) should be positive");
            Assert.True(result <= x + 1e-8, $"GELU({x})={result} should be <= {x}");
        }
    }

    // ========================================================================
    // Additional scalar cross-checks
    // ========================================================================

    [Fact]
    public void AllActivations_Derivative_IsNonNegative_ForCommon()
    {
        // For Sigmoid and SoftPlus, the derivative is always non-negative
        var sigmoid = new SigmoidActivation<double>();
        var softplus = new SoftPlusActivation<double>();
        double[] testValues = [-5, -2, -1, 0, 1, 2, 5];

        foreach (double x in testValues)
        {
            Assert.True(sigmoid.Derivative(x) >= 0, $"Sigmoid derivative at {x} should be non-negative");
            Assert.True(softplus.Derivative(x) >= 0, $"SoftPlus derivative at {x} should be non-negative");
        }
    }

    // ========================================================================
    // Numerical stability tests
    // ========================================================================

    [Fact]
    public void Sigmoid_VeryLargePositive_Approaches1()
    {
        var sigmoid = new SigmoidActivation<double>();
        double result = sigmoid.Activate(100.0);
        Assert.Equal(1.0, result, 1e-10);
    }

    [Fact]
    public void Sigmoid_VeryLargeNegative_Approaches0()
    {
        var sigmoid = new SigmoidActivation<double>();
        double result = sigmoid.Activate(-100.0);
        Assert.Equal(0.0, result, 1e-10);
    }

    [Fact]
    public void SoftPlus_VeryLargeNegative_ApproachesZero()
    {
        var sp = new SoftPlusActivation<double>();
        double result = sp.Activate(-50.0);
        Assert.True(result > 0 && result < 1e-10, $"SoftPlus(-50)={result} should be near zero");
    }

    // ========================================================================
    // Float type tests
    // ========================================================================

    [Fact]
    public void Sigmoid_Float_MatchesDoubleWithReasonablePrecision()
    {
        var sigmoidF = new SigmoidActivation<float>();
        var sigmoidD = new SigmoidActivation<double>();

        float resultF = sigmoidF.Activate(1.0f);
        double resultD = sigmoidD.Activate(1.0);
        Assert.Equal(resultD, resultF, 1e-6);
    }

    [Fact]
    public void ReLU_Float_MatchesDouble()
    {
        var reluF = new ReLUActivation<float>();
        var reluD = new ReLUActivation<double>();

        Assert.Equal(reluD.Activate(-3.0), reluF.Activate(-3.0f), 1e-6);
        Assert.Equal(reluD.Activate(3.0), reluF.Activate(3.0f), 1e-6);
    }

    // ========================================================================
    // Property: activation/derivative consistency via fundamental theorem
    // ========================================================================

    [Fact]
    public void Sigmoid_IntegrationConsistency()
    {
        // Integral of sigmoid derivative from a to b should equal sigmoid(b) - sigmoid(a)
        // Using trapezoidal numerical integration
        var sigmoid = new SigmoidActivation<double>();
        double a = -2.0, b = 2.0;
        int steps = 10000;
        double h = (b - a) / steps;

        double integral = 0;
        for (int i = 0; i < steps; i++)
        {
            double x0 = a + i * h;
            double x1 = x0 + h;
            integral += 0.5 * (sigmoid.Derivative(x0) + sigmoid.Derivative(x1)) * h;
        }

        double expected = sigmoid.Activate(b) - sigmoid.Activate(a);
        Assert.Equal(expected, integral, 1e-6);
    }
}
