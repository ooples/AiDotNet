using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Integration tests for activation function classes.
/// Tests activation and derivative computations for various activation functions.
/// </summary>
public class ActivationFunctionsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Sigmoid Activation Tests

    [Fact]
    public void SigmoidActivation_Activate_ReturnsValueBetweenZeroAndOne()
    {
        // Arrange
        var sigmoid = new SigmoidActivation<double>();

        // Act
        var result = sigmoid.Activate(0.0);

        // Assert
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void SigmoidActivation_ActivatePositive_ApproachesOne()
    {
        // Arrange
        var sigmoid = new SigmoidActivation<double>();

        // Act
        var result = sigmoid.Activate(10.0);

        // Assert
        Assert.True(result > 0.99);
        Assert.True(result <= 1.0);
    }

    [Fact]
    public void SigmoidActivation_ActivateNegative_ApproachesZero()
    {
        // Arrange
        var sigmoid = new SigmoidActivation<double>();

        // Act
        var result = sigmoid.Activate(-10.0);

        // Assert
        Assert.True(result < 0.01);
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void SigmoidActivation_Derivative_MaximumAtZero()
    {
        // Arrange
        var sigmoid = new SigmoidActivation<double>();

        // Act
        var derivativeAtZero = sigmoid.Derivative(0.0);
        var derivativeAtPositive = sigmoid.Derivative(2.0);
        var derivativeAtNegative = sigmoid.Derivative(-2.0);

        // Assert
        Assert.Equal(0.25, derivativeAtZero, Tolerance);
        Assert.True(derivativeAtPositive < derivativeAtZero);
        Assert.True(derivativeAtNegative < derivativeAtZero);
    }

    [Fact]
    public void SigmoidActivation_ActivateVector_ProcessesAllElements()
    {
        // Arrange
        var sigmoid = new SigmoidActivation<double>();
        var input = new Vector<double>(new[] { -2.0, 0.0, 2.0 });

        // Act
        var result = sigmoid.Activate(input);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.True(result[0] < 0.5);
        Assert.Equal(0.5, result[1], Tolerance);
        Assert.True(result[2] > 0.5);
    }

    #endregion

    #region ReLU Activation Tests

    [Fact]
    public void ReLUActivation_ActivatePositive_ReturnsInput()
    {
        // Arrange
        var relu = new ReLUActivation<double>();

        // Act
        var result = relu.Activate(5.0);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void ReLUActivation_ActivateNegative_ReturnsZero()
    {
        // Arrange
        var relu = new ReLUActivation<double>();

        // Act
        var result = relu.Activate(-5.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ReLUActivation_ActivateZero_ReturnsZero()
    {
        // Arrange
        var relu = new ReLUActivation<double>();

        // Act
        var result = relu.Activate(0.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ReLUActivation_DerivativePositive_ReturnsOne()
    {
        // Arrange
        var relu = new ReLUActivation<double>();

        // Act
        var result = relu.Derivative(5.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ReLUActivation_DerivativeNegative_ReturnsZero()
    {
        // Arrange
        var relu = new ReLUActivation<double>();

        // Act
        var result = relu.Derivative(-5.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ReLUActivation_ActivateVector_ProcessesAllElements()
    {
        // Arrange
        var relu = new ReLUActivation<double>();
        var input = new Vector<double>(new[] { -2.0, 0.0, 3.0, -1.0, 5.0 });

        // Act
        var result = relu.Activate(input);

        // Assert
        Assert.Equal(5, result.Length);
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
        Assert.Equal(0.0, result[3], Tolerance);
        Assert.Equal(5.0, result[4], Tolerance);
    }

    #endregion

    #region Tanh Activation Tests

    [Fact]
    public void TanhActivation_ActivateZero_ReturnsZero()
    {
        // Arrange
        var tanh = new TanhActivation<double>();

        // Act
        var result = tanh.Activate(0.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void TanhActivation_ActivatePositive_ReturnsPositive()
    {
        // Arrange
        var tanh = new TanhActivation<double>();

        // Act
        var result = tanh.Activate(2.0);

        // Assert
        Assert.True(result > 0.0);
        Assert.True(result < 1.0);
    }

    [Fact]
    public void TanhActivation_ActivateNegative_ReturnsNegative()
    {
        // Arrange
        var tanh = new TanhActivation<double>();

        // Act
        var result = tanh.Activate(-2.0);

        // Assert
        Assert.True(result < 0.0);
        Assert.True(result > -1.0);
    }

    [Fact]
    public void TanhActivation_LargePositive_ApproachesOne()
    {
        // Arrange
        var tanh = new TanhActivation<double>();

        // Act
        var result = tanh.Activate(10.0);

        // Assert
        Assert.True(result > 0.99);
    }

    [Fact]
    public void TanhActivation_LargeNegative_ApproachesMinusOne()
    {
        // Arrange
        var tanh = new TanhActivation<double>();

        // Act
        var result = tanh.Activate(-10.0);

        // Assert
        Assert.True(result < -0.99);
    }

    [Fact]
    public void TanhActivation_Derivative_MaximumAtZero()
    {
        // Arrange
        var tanh = new TanhActivation<double>();

        // Act
        var derivativeAtZero = tanh.Derivative(0.0);

        // Assert
        Assert.Equal(1.0, derivativeAtZero, Tolerance);
    }

    #endregion

    #region Softmax Activation Tests

    [Fact]
    public void SoftmaxActivation_ActivateVector_SumsToOne()
    {
        // Arrange
        var softmax = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = softmax.Activate(input);

        // Assert
        var sum = 0.0;
        for (int i = 0; i < result.Length; i++)
        {
            sum += result[i];
        }
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void SoftmaxActivation_ActivateVector_AllPositive()
    {
        // Arrange
        var softmax = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { -1.0, 0.0, 1.0 });

        // Act
        var result = softmax.Activate(input);

        // Assert
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] > 0);
        }
    }

    [Fact]
    public void SoftmaxActivation_LargestInputHasLargestOutput()
    {
        // Arrange
        var softmax = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 5.0, 2.0 });

        // Act
        var result = softmax.Activate(input);

        // Assert
        Assert.True(result[1] > result[0]);
        Assert.True(result[1] > result[2]);
    }

    #endregion

    #region LeakyReLU Activation Tests

    [Fact]
    public void LeakyReLUActivation_ActivatePositive_ReturnsInput()
    {
        // Arrange
        var leakyRelu = new LeakyReLUActivation<double>(0.01);

        // Act
        var result = leakyRelu.Activate(5.0);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void LeakyReLUActivation_ActivateNegative_ReturnsScaledInput()
    {
        // Arrange
        var leakyRelu = new LeakyReLUActivation<double>(0.1);

        // Act
        var result = leakyRelu.Activate(-5.0);

        // Assert
        Assert.Equal(-0.5, result, Tolerance);
    }

    [Fact]
    public void LeakyReLUActivation_DerivativeNegative_ReturnsAlpha()
    {
        // Arrange
        var leakyRelu = new LeakyReLUActivation<double>(0.2);

        // Act
        var result = leakyRelu.Derivative(-5.0);

        // Assert
        Assert.Equal(0.2, result, Tolerance);
    }

    #endregion

    #region ELU Activation Tests

    [Fact]
    public void ELUActivation_ActivatePositive_ReturnsInput()
    {
        // Arrange
        var elu = new ELUActivation<double>(1.0);

        // Act
        var result = elu.Activate(5.0);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void ELUActivation_ActivateNegative_ReturnsExponential()
    {
        // Arrange
        var elu = new ELUActivation<double>(1.0);

        // Act
        var result = elu.Activate(-1.0);

        // Assert
        Assert.True(result < 0);
        Assert.True(result > -1.0);
    }

    [Fact]
    public void ELUActivation_LargeNegative_ApproachesMinusAlpha()
    {
        // Arrange
        var elu = new ELUActivation<double>(1.0);

        // Act
        var result = elu.Activate(-10.0);

        // Assert
        Assert.True(result > -1.0);
        Assert.True(result < -0.99);
    }

    #endregion

    #region GELU Activation Tests

    [Fact]
    public void GELUActivation_ActivateZero_ReturnsZero()
    {
        // Arrange
        var gelu = new GELUActivation<double>();

        // Act
        var result = gelu.Activate(0.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void GELUActivation_ActivatePositive_ReturnsPositive()
    {
        // Arrange
        var gelu = new GELUActivation<double>();

        // Act
        var result = gelu.Activate(2.0);

        // Assert
        Assert.True(result > 0);
    }

    [Fact]
    public void GELUActivation_LargePositive_ApproachesInput()
    {
        // Arrange
        var gelu = new GELUActivation<double>();

        // Act
        var result = gelu.Activate(5.0);

        // Assert
        Assert.True(result > 4.9);
    }

    #endregion

    #region HardSigmoid Activation Tests

    [Fact]
    public void HardSigmoidActivation_ActivateZero_ReturnsHalf()
    {
        // Arrange
        var hardSigmoid = new HardSigmoidActivation<double>();

        // Act
        var result = hardSigmoid.Activate(0.0);

        // Assert
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void HardSigmoidActivation_LargePositive_ReturnsOne()
    {
        // Arrange
        var hardSigmoid = new HardSigmoidActivation<double>();

        // Act
        var result = hardSigmoid.Activate(5.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void HardSigmoidActivation_LargeNegative_ReturnsZero()
    {
        // Arrange
        var hardSigmoid = new HardSigmoidActivation<double>();

        // Act
        var result = hardSigmoid.Activate(-5.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region HardTanh Activation Tests

    [Fact]
    public void HardTanhActivation_ActivateZero_ReturnsZero()
    {
        // Arrange
        var hardTanh = new HardTanhActivation<double>();

        // Act
        var result = hardTanh.Activate(0.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void HardTanhActivation_LargePositive_ReturnsOne()
    {
        // Arrange
        var hardTanh = new HardTanhActivation<double>();

        // Act
        var result = hardTanh.Activate(5.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void HardTanhActivation_LargeNegative_ReturnsMinusOne()
    {
        // Arrange
        var hardTanh = new HardTanhActivation<double>();

        // Act
        var result = hardTanh.Activate(-5.0);

        // Assert
        Assert.Equal(-1.0, result, Tolerance);
    }

    #endregion

    #region Gaussian Activation Tests

    [Fact]
    public void GaussianActivation_ActivateZero_ReturnsOne()
    {
        // Arrange
        var gaussian = new GaussianActivation<double>();

        // Act
        var result = gaussian.Activate(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void GaussianActivation_ActivateNonZero_ReturnsLessThanOne()
    {
        // Arrange
        var gaussian = new GaussianActivation<double>();

        // Act
        var result = gaussian.Activate(1.0);

        // Assert
        Assert.True(result < 1.0);
        Assert.True(result > 0.0);
    }

    #endregion

    #region Linear Activation Tests

    [Fact]
    public void IdentityActivation_Activate_ReturnsInput()
    {
        // Arrange
        var linear = new IdentityActivation<double>();

        // Act
        var result = linear.Activate(5.0);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void IdentityActivation_Derivative_ReturnsOne()
    {
        // Arrange
        var linear = new IdentityActivation<double>();

        // Act
        var result = linear.Derivative(5.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllActivationFunctions_HandleLargeValues()
    {
        // Arrange - test only bounded activation functions
        var activations = new IActivationFunction<double>[]
        {
            new SigmoidActivation<double>(),
            new ReLUActivation<double>(),
            new TanhActivation<double>(),
            new LeakyReLUActivation<double>(0.01),
            new ELUActivation<double>(1.0),
            new IdentityActivation<double>()
        };

        // Act & Assert - only check for NaN, as some activations may return infinity for extreme values
        foreach (var activation in activations)
        {
            var result = activation.Activate(100.0);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void AllActivationFunctions_HandleNegativeValues()
    {
        // Arrange
        var activations = new IActivationFunction<double>[]
        {
            new SigmoidActivation<double>(),
            new ReLUActivation<double>(),
            new TanhActivation<double>(),
            new LeakyReLUActivation<double>(0.01),
            new ELUActivation<double>(1.0),
            new GELUActivation<double>(),
            new IdentityActivation<double>()
        };

        // Act & Assert
        foreach (var activation in activations)
        {
            var result = activation.Activate(-100.0);
            Assert.False(double.IsNaN(result));
            Assert.False(double.IsInfinity(result));
        }
    }

    #endregion
}
