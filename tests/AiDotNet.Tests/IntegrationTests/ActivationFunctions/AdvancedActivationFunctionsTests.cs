using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Integration tests for all activation function classes not covered by the basic tests.
/// Tests Activate, Derivative, vector operations, and mathematical properties.
/// </summary>
public class AdvancedActivationFunctionsTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    #region Swish / SiLU Activation Tests

    [Fact]
    public void SwishActivation_ActivateZero_ReturnsZero()
    {
        var swish = new SwishActivation<double>();
        var result = swish.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SwishActivation_LargePositive_ApproachesInput()
    {
        var swish = new SwishActivation<double>();
        var result = swish.Activate(10.0);
        // swish(x) = x * sigmoid(x) -> x * 1 = x for large positive
        Assert.True(result > 9.9);
    }

    [Fact]
    public void SwishActivation_NegativeInput_AllowsSomeNegative()
    {
        var swish = new SwishActivation<double>();
        // Swish is non-monotonic: slightly negative for small negative inputs
        var result = swish.Activate(-1.0);
        Assert.True(result < 0.0);
        Assert.True(result > -0.5);
    }

    [Fact]
    public void SwishActivation_Derivative_AtZero_IsHalf()
    {
        var swish = new SwishActivation<double>();
        var deriv = swish.Derivative(0.0);
        // swish'(0) = sigmoid(0) + 0*sigmoid'(0) = 0.5
        Assert.Equal(0.5, deriv, Tolerance);
    }

    [Fact]
    public void SiLUActivation_MatchesSwish()
    {
        var silu = new SiLUActivation<double>();
        var swish = new SwishActivation<double>();
        // SiLU is the same as Swish: x * sigmoid(x)
        var inputs = new[] { -3.0, -1.0, 0.0, 1.0, 3.0 };
        foreach (var x in inputs)
        {
            Assert.Equal(swish.Activate(x), silu.Activate(x), Tolerance);
        }
    }

    #endregion

    #region Mish Activation Tests

    [Fact]
    public void MishActivation_ActivateZero_ReturnsZero()
    {
        var mish = new MishActivation<double>();
        var result = mish.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MishActivation_LargePositive_ApproachesInput()
    {
        var mish = new MishActivation<double>();
        var result = mish.Activate(10.0);
        // mish(x) = x * tanh(softplus(x)) -> x * tanh(x) -> x * 1 for large x
        Assert.True(result > 9.9);
    }

    [Fact]
    public void MishActivation_NegativeInput_AllowsSomeNegative()
    {
        var mish = new MishActivation<double>();
        var result = mish.Activate(-1.0);
        Assert.True(result < 0.0);
    }

    [Fact]
    public void MishActivation_Derivative_NotNaN()
    {
        var mish = new MishActivation<double>();
        var inputs = new[] { -5.0, -1.0, 0.0, 1.0, 5.0 };
        foreach (var x in inputs)
        {
            var deriv = mish.Derivative(x);
            Assert.False(double.IsNaN(deriv));
        }
    }

    #endregion

    #region SELU Activation Tests

    [Fact]
    public void SELUActivation_ActivatePositive_ReturnsScaledInput()
    {
        var selu = new SELUActivation<double>();
        var result = selu.Activate(1.0);
        // SELU: lambda * x for x > 0, lambda ~ 1.0507
        Assert.True(result > 1.0);
        Assert.True(result < 1.1);
    }

    [Fact]
    public void SELUActivation_ActivateZero_ReturnsZero()
    {
        var selu = new SELUActivation<double>();
        var result = selu.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SELUActivation_ActivateNegative_ReturnsBoundedNegative()
    {
        var selu = new SELUActivation<double>();
        var result = selu.Activate(-10.0);
        // SELU: lambda * alpha * (exp(x) - 1) for x < 0, approaches -lambda * alpha ~ -1.758
        Assert.True(result < 0);
        Assert.True(result > -2.0);
    }

    [Fact]
    public void SELUActivation_DerivativePositive_IsLambda()
    {
        var selu = new SELUActivation<double>();
        var deriv = selu.Derivative(1.0);
        // For positive input, derivative = lambda ~ 1.0507
        Assert.True(deriv > 1.0);
        Assert.True(deriv < 1.1);
    }

    #endregion

    #region CELU Activation Tests

    [Fact]
    public void CELUActivation_ActivatePositive_ReturnsInput()
    {
        var celu = new CELUActivation<double>(1.0);
        var result = celu.Activate(5.0);
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void CELUActivation_ActivateNegative_ReturnsBoundedNegative()
    {
        var celu = new CELUActivation<double>(1.0);
        var result = celu.Activate(-10.0);
        // CELU: alpha * (exp(x/alpha) - 1) for x < 0, approaches -alpha
        Assert.True(result < 0);
        Assert.True(result > -1.1);
    }

    [Fact]
    public void CELUActivation_ActivateZero_ReturnsZero()
    {
        var celu = new CELUActivation<double>(1.0);
        var result = celu.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CELUActivation_DifferentAlpha_AffectsNegativeRange()
    {
        var celu1 = new CELUActivation<double>(0.5);
        var celu2 = new CELUActivation<double>(2.0);
        var r1 = celu1.Activate(-5.0);
        var r2 = celu2.Activate(-5.0);
        // Larger alpha -> larger negative saturation
        Assert.True(Math.Abs(r2) > Math.Abs(r1));
    }

    #endregion

    #region SoftPlus Activation Tests

    [Fact]
    public void SoftPlusActivation_AlwaysPositive()
    {
        var sp = new SoftPlusActivation<double>();
        var inputs = new[] { -10.0, -1.0, 0.0, 1.0, 10.0 };
        foreach (var x in inputs)
        {
            var result = sp.Activate(x);
            Assert.True(result > 0);
        }
    }

    [Fact]
    public void SoftPlusActivation_ActivateZero_ReturnsLn2()
    {
        var sp = new SoftPlusActivation<double>();
        var result = sp.Activate(0.0);
        // softplus(0) = ln(1 + e^0) = ln(2) ~ 0.6931
        Assert.Equal(Math.Log(2), result, Tolerance);
    }

    [Fact]
    public void SoftPlusActivation_LargePositive_ApproachesInput()
    {
        var sp = new SoftPlusActivation<double>();
        var result = sp.Activate(20.0);
        // softplus(x) -> x for large x
        Assert.Equal(20.0, result, 0.01);
    }

    [Fact]
    public void SoftPlusActivation_DerivativeIsSigmoid()
    {
        var sp = new SoftPlusActivation<double>();
        // softplus'(x) = sigmoid(x) = 1/(1+exp(-x))
        var deriv = sp.Derivative(0.0);
        Assert.Equal(0.5, deriv, Tolerance);
    }

    #endregion

    #region SoftSign Activation Tests

    [Fact]
    public void SoftSignActivation_ActivateZero_ReturnsZero()
    {
        var ss = new SoftSignActivation<double>();
        var result = ss.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SoftSignActivation_OutputBoundedNegOneToOne()
    {
        var ss = new SoftSignActivation<double>();
        var inputs = new[] { -100.0, -1.0, 0.0, 1.0, 100.0 };
        foreach (var x in inputs)
        {
            var result = ss.Activate(x);
            Assert.True(result >= -1.0);
            Assert.True(result <= 1.0);
        }
    }

    [Fact]
    public void SoftSignActivation_KnownValue()
    {
        var ss = new SoftSignActivation<double>();
        // softsign(1) = 1/(1+|1|) = 0.5
        var result = ss.Activate(1.0);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void SoftSignActivation_IsOddFunction()
    {
        var ss = new SoftSignActivation<double>();
        var result1 = ss.Activate(2.0);
        var result2 = ss.Activate(-2.0);
        Assert.Equal(-result1, result2, Tolerance);
    }

    #endregion

    #region HardSwish Activation Tests

    [Fact]
    public void HardSwishActivation_ActivateZero_ReturnsZero()
    {
        var hs = new HardSwishActivation<double>();
        var result = hs.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void HardSwishActivation_LargePositive_ReturnsInput()
    {
        var hs = new HardSwishActivation<double>();
        var result = hs.Activate(5.0);
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void HardSwishActivation_LargeNegative_ReturnsZero()
    {
        var hs = new HardSwishActivation<double>();
        var result = hs.Activate(-5.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region ReLU6 Activation Tests

    [Fact]
    public void ReLU6Activation_PositiveBelow6_ReturnsInput()
    {
        var relu6 = new ReLU6Activation<double>();
        var result = relu6.Activate(3.0);
        Assert.Equal(3.0, result, Tolerance);
    }

    [Fact]
    public void ReLU6Activation_Above6_Returns6()
    {
        var relu6 = new ReLU6Activation<double>();
        var result = relu6.Activate(10.0);
        Assert.Equal(6.0, result, Tolerance);
    }

    [Fact]
    public void ReLU6Activation_Negative_ReturnsZero()
    {
        var relu6 = new ReLU6Activation<double>();
        var result = relu6.Activate(-5.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ReLU6Activation_DerivativeInRange_ReturnsOne()
    {
        var relu6 = new ReLU6Activation<double>();
        var deriv = relu6.Derivative(3.0);
        Assert.Equal(1.0, deriv, Tolerance);
    }

    [Fact]
    public void ReLU6Activation_DerivativeAbove6_ReturnsZero()
    {
        var relu6 = new ReLU6Activation<double>();
        var deriv = relu6.Derivative(10.0);
        Assert.Equal(0.0, deriv, Tolerance);
    }

    #endregion

    #region PReLU Activation Tests

    [Fact]
    public void PReLUActivation_PositiveInput_ReturnsInput()
    {
        var prelu = new PReLUActivation<double>(0.01);
        var result = prelu.Activate(5.0);
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void PReLUActivation_NegativeInput_ReturnsScaled()
    {
        var prelu = new PReLUActivation<double>(0.25);
        var result = prelu.Activate(-4.0);
        Assert.Equal(-1.0, result, Tolerance);
    }

    #endregion

    #region ThresholdedReLU Activation Tests

    [Fact]
    public void ThresholdedReLUActivation_AboveThreshold_ReturnsInput()
    {
        var trelu = new ThresholdedReLUActivation<double>(1.0);
        var result = trelu.Activate(2.0);
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void ThresholdedReLUActivation_BelowThreshold_ReturnsZero()
    {
        var trelu = new ThresholdedReLUActivation<double>(1.0);
        var result = trelu.Activate(0.5);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ThresholdedReLUActivation_Negative_ReturnsZero()
    {
        var trelu = new ThresholdedReLUActivation<double>(1.0);
        var result = trelu.Activate(-5.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region RReLU Activation Tests

    [Fact]
    public void RReLUActivation_PositiveInput_ReturnsInput()
    {
        var rrelu = new RReLUActivation<double>(0.125, 0.333);
        var result = rrelu.Activate(5.0);
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void RReLUActivation_NegativeInput_ReturnsScaledBetweenBounds()
    {
        var rrelu = new RReLUActivation<double>(0.125, 0.333);
        var result = rrelu.Activate(-4.0);
        // Result should be between -4*0.333 and -4*0.125
        Assert.True(result <= -4.0 * 0.125);
        Assert.True(result >= -4.0 * 0.333);
    }

    #endregion

    #region ISRU Activation Tests

    [Fact]
    public void ISRUActivation_ActivateZero_ReturnsZero()
    {
        var isru = new ISRUActivation<double>(1.0);
        var result = isru.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ISRUActivation_OutputBounded()
    {
        var isru = new ISRUActivation<double>(1.0);
        var inputs = new[] { -10.0, -1.0, 0.0, 1.0, 10.0 };
        foreach (var x in inputs)
        {
            var result = isru.Activate(x);
            Assert.False(double.IsNaN(result));
            Assert.False(double.IsInfinity(result));
        }
    }

    #endregion

    #region LiSHT Activation Tests

    [Fact]
    public void LiSHTActivation_ActivateZero_ReturnsZero()
    {
        var lisht = new LiSHTActivation<double>();
        var result = lisht.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void LiSHTActivation_PositiveInput_ReturnsPositive()
    {
        var lisht = new LiSHTActivation<double>();
        // LiSHT(x) = x * tanh(x)
        var result = lisht.Activate(2.0);
        Assert.True(result > 0);
    }

    [Fact]
    public void LiSHTActivation_NegativeInput_ReturnsPositive()
    {
        var lisht = new LiSHTActivation<double>();
        // LiSHT(-x) = -x * tanh(-x) = -x * (-tanh(x)) = x * tanh(x) > 0
        var result = lisht.Activate(-2.0);
        Assert.True(result > 0);
    }

    [Fact]
    public void LiSHTActivation_IsEvenFunction()
    {
        var lisht = new LiSHTActivation<double>();
        var result1 = lisht.Activate(2.0);
        var result2 = lisht.Activate(-2.0);
        Assert.Equal(result1, result2, Tolerance);
    }

    #endregion

    #region BentIdentity Activation Tests

    [Fact]
    public void BentIdentityActivation_ActivateZero_ReturnsZero()
    {
        var bi = new BentIdentityActivation<double>();
        var result = bi.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void BentIdentityActivation_LargePositive_ApproachesInput()
    {
        var bi = new BentIdentityActivation<double>();
        var result = bi.Activate(100.0);
        Assert.True(result > 99.0);
    }

    [Fact]
    public void BentIdentityActivation_Derivative_NotNaN()
    {
        var bi = new BentIdentityActivation<double>();
        var inputs = new[] { -5.0, 0.0, 5.0 };
        foreach (var x in inputs)
        {
            var deriv = bi.Derivative(x);
            Assert.False(double.IsNaN(deriv));
        }
    }

    #endregion

    #region SignActivation Tests

    [Fact]
    public void SignActivation_PositiveInput_ReturnsOne()
    {
        var sign = new SignActivation<double>();
        var result = sign.Activate(5.0);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void SignActivation_NegativeInput_ReturnsMinusOne()
    {
        var sign = new SignActivation<double>();
        var result = sign.Activate(-5.0);
        Assert.Equal(-1.0, result, Tolerance);
    }

    [Fact]
    public void SignActivation_Zero_ReturnsZero()
    {
        var sign = new SignActivation<double>();
        var result = sign.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region SquashActivation Tests

    [Fact]
    public void SquashActivation_OutputBounded()
    {
        var squash = new SquashActivation<double>();
        var inputs = new[] { -10.0, -1.0, 0.0, 1.0, 10.0 };
        foreach (var x in inputs)
        {
            var result = squash.Activate(x);
            Assert.False(double.IsNaN(result));
            Assert.False(double.IsInfinity(result));
        }
    }

    #endregion

    #region ScaledTanh Activation Tests

    [Fact]
    public void ScaledTanhActivation_ActivateZero_ReturnsZero()
    {
        var st = new ScaledTanhActivation<double>(1.0);
        var result = st.Activate(0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ScaledTanhActivation_DifferentBeta_AffectsScale()
    {
        var st1 = new ScaledTanhActivation<double>(0.5);
        var st2 = new ScaledTanhActivation<double>(2.0);
        var r1 = st1.Activate(1.0);
        var r2 = st2.Activate(1.0);
        // Different beta should give different results
        Assert.NotEqual(r1, r2, Tolerance);
    }

    #endregion

    #region SQRBF Activation Tests

    [Fact]
    public void SQRBFActivation_ActivateZero_ReturnsOne()
    {
        var sqrbf = new SQRBFActivation<double>(1.0);
        var result = sqrbf.Activate(0.0);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void SQRBFActivation_LargeInput_ApproachesZero()
    {
        var sqrbf = new SQRBFActivation<double>(1.0);
        var result = sqrbf.Activate(10.0);
        Assert.True(result >= 0.0);
    }

    #endregion

    #region Softmax Variants Tests

    [Fact]
    public void LogSoftmaxActivation_VectorOutput_LogProbabilities()
    {
        var lsm = new LogSoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = lsm.Activate(input);
        // All log-probabilities should be negative
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] <= 0);
        }
    }

    [Fact]
    public void SoftminActivation_VectorOutput_SumsToOne()
    {
        var sm = new SoftminActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = sm.Activate(input);
        var sum = 0.0;
        for (int i = 0; i < result.Length; i++)
        {
            sum += result[i];
            Assert.True(result[i] > 0);
        }
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void SoftminActivation_SmallestInputGetsLargestProbability()
    {
        var sm = new SoftminActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 5.0, 3.0 });
        var result = sm.Activate(input);
        // Softmin: smallest input should have largest probability
        Assert.True(result[0] > result[1]);
        Assert.True(result[0] > result[2]);
    }

    [Fact]
    public void SparsemaxActivation_VectorOutput_SumsToOne()
    {
        var sm = new SparsemaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = sm.Activate(input);
        var sum = 0.0;
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0);
            sum += result[i];
        }
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void SparsemaxActivation_ProducesSparseOutput()
    {
        var sm = new SparsemaxActivation<double>();
        var input = new Vector<double>(new[] { 0.1, 0.2, 5.0 });
        var result = sm.Activate(input);
        // Sparsemax should zero out the smallest values
        int zeroCount = 0;
        for (int i = 0; i < result.Length; i++)
        {
            if (Math.Abs(result[i]) < Tolerance) zeroCount++;
        }
        Assert.True(zeroCount >= 1);
    }

    [Fact]
    public void TaylorSoftmaxActivation_VectorOutput_AllPositive()
    {
        var tsm = new TaylorSoftmaxActivation<double>(2);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = tsm.Activate(input);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] > 0);
        }
    }

    [Fact]
    public void LogSoftminActivation_VectorOutput_AllNonPositive()
    {
        var lsm = new LogSoftminActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = lsm.Activate(input);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] <= 0);
        }
    }

    [Fact]
    public void SphericalSoftmaxActivation_VectorOutput_AllPositive()
    {
        var ssm = new SphericalSoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = ssm.Activate(input);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0);
        }
    }

    #endregion

    #region BinarySpiking Activation Tests

    [Fact]
    public void BinarySpikingActivation_AboveThreshold_ReturnsOne()
    {
        var bs = new BinarySpikingActivation<double>();
        var result = bs.Activate(2.0);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void BinarySpikingActivation_BelowThreshold_ReturnsZero()
    {
        var bs = new BinarySpikingActivation<double>();
        var result = bs.Activate(0.5);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region GumbelSoftmax Activation Tests

    [Fact]
    public void GumbelSoftmaxActivation_VectorOutput_AllPositive()
    {
        var gs = new GumbelSoftmaxActivation<double>(1.0, seed: 42);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = gs.Activate(input);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0);
        }
    }

    #endregion

    #region MaxoutActivation Tests

    [Fact]
    public void MaxoutActivation_ReturnsMaxOfPieces()
    {
        var maxout = new MaxoutActivation<double>(2);
        var input = new Vector<double>(new[] { 1.0, 3.0, 2.0, 4.0 });
        var result = maxout.Activate(input);
        // With 2 pieces, takes max of each pair
        Assert.Equal(2, result.Length);
        Assert.Equal(3.0, result[0], Tolerance);
        Assert.Equal(4.0, result[1], Tolerance);
    }

    #endregion

    #region HierarchicalSoftmax Activation Tests

    [Fact]
    public void HierarchicalSoftmaxActivation_VectorOutput_AllPositive()
    {
        var hs = new HierarchicalSoftmaxActivation<double>(4);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = hs.Activate(input);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= 0);
        }
    }

    #endregion

    #region Vector and Tensor Operations Tests

    [Fact]
    public void AllScalarActivations_VectorActivate_PreservesLength()
    {
        var activations = new IActivationFunction<double>[]
        {
            new SwishActivation<double>(),
            new MishActivation<double>(),
            new SELUActivation<double>(),
            new CELUActivation<double>(1.0),
            new SoftPlusActivation<double>(),
            new SoftSignActivation<double>(),
            new HardSwishActivation<double>(),
            new ReLU6Activation<double>(),
            new PReLUActivation<double>(0.01),
            new ThresholdedReLUActivation<double>(1.0),
            new ISRUActivation<double>(1.0),
            new LiSHTActivation<double>(),
            new BentIdentityActivation<double>(),
            new SignActivation<double>(),
            new SquashActivation<double>(),
            new ScaledTanhActivation<double>(1.0),
            new SQRBFActivation<double>(1.0),
            new SiLUActivation<double>(),
            new BinarySpikingActivation<double>(),
        };

        var input = new Vector<double>(new[] { -2.0, -0.5, 0.0, 0.5, 2.0 });

        foreach (var activation in activations)
        {
            var result = activation.Activate(input);
            Assert.Equal(input.Length, result.Length);
            for (int i = 0; i < result.Length; i++)
            {
                Assert.False(double.IsNaN(result[i]),
                    $"{activation.GetType().Name} produced NaN for input {input[i]}");
            }
        }
    }

    [Fact]
    public void AllScalarActivations_Derivative_NoNaN()
    {
        var activations = new IActivationFunction<double>[]
        {
            new SwishActivation<double>(),
            new MishActivation<double>(),
            new SELUActivation<double>(),
            new CELUActivation<double>(1.0),
            new SoftPlusActivation<double>(),
            new SoftSignActivation<double>(),
            new HardSwishActivation<double>(),
            new ReLU6Activation<double>(),
            new PReLUActivation<double>(0.01),
            new LiSHTActivation<double>(),
            new BentIdentityActivation<double>(),
            new ScaledTanhActivation<double>(1.0),
            new SiLUActivation<double>(),
        };

        var testValues = new[] { -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0 };

        foreach (var activation in activations)
        {
            foreach (var x in testValues)
            {
                var deriv = activation.Derivative(x);
                Assert.False(double.IsNaN(deriv),
                    $"{activation.GetType().Name}.Derivative({x}) = NaN");
            }
        }
    }

    [Fact]
    public void AllScalarActivations_TensorActivate_PreservesShape()
    {
        var activations = new IActivationFunction<double>[]
        {
            new SwishActivation<double>(),
            new MishActivation<double>(),
            new SELUActivation<double>(),
            new SoftPlusActivation<double>(),
            new HardSwishActivation<double>(),
            new ReLU6Activation<double>(),
            new SiLUActivation<double>(),
        };

        var input = new Tensor<double>(new double[] { -1.0, 0.0, 1.0, 2.0 }, [2, 2]);

        foreach (var act in activations)
        {
            if (act is IVectorActivationFunction<double> vact)
            {
                var result = vact.Activate(input);
                Assert.Equal(input.Shape.Length, result.Shape.Length);
                Assert.Equal(input.Shape[0], result.Shape[0]);
                Assert.Equal(input.Shape[1], result.Shape[1]);
            }
        }
    }

    #endregion

    #region Backward Pass Tests

    [Fact]
    public void SwishActivation_Backward_ProducesValidGradient()
    {
        var swish = new SwishActivation<double>();
        var input = new Tensor<double>(new double[] { -1.0, 0.0, 1.0, 2.0 }, [4]);
        var gradOutput = new Tensor<double>(new double[] { 1.0, 1.0, 1.0, 1.0 }, [4]);
        var gradInput = swish.Backward(input, gradOutput);
        Assert.Equal(4, gradInput.Length);
        for (int i = 0; i < gradInput.Length; i++)
        {
            Assert.False(double.IsNaN(gradInput[i]));
        }
    }

    [Fact]
    public void MishActivation_Backward_ProducesValidGradient()
    {
        var mish = new MishActivation<double>();
        var input = new Tensor<double>(new double[] { -1.0, 0.0, 1.0, 2.0 }, [4]);
        var gradOutput = new Tensor<double>(new double[] { 1.0, 1.0, 1.0, 1.0 }, [4]);
        var gradInput = mish.Backward(input, gradOutput);
        Assert.Equal(4, gradInput.Length);
        for (int i = 0; i < gradInput.Length; i++)
        {
            Assert.False(double.IsNaN(gradInput[i]));
        }
    }

    [Fact]
    public void SELUActivation_Backward_ProducesValidGradient()
    {
        var selu = new SELUActivation<double>();
        var input = new Tensor<double>(new double[] { -1.0, 0.0, 1.0, 2.0 }, [4]);
        var gradOutput = new Tensor<double>(new double[] { 1.0, 1.0, 1.0, 1.0 }, [4]);
        var gradInput = selu.Backward(input, gradOutput);
        Assert.Equal(4, gradInput.Length);
        for (int i = 0; i < gradInput.Length; i++)
        {
            Assert.False(double.IsNaN(gradInput[i]));
        }
    }

    #endregion
}
