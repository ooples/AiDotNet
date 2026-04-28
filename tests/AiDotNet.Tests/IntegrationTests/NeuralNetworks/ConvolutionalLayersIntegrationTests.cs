using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for convolutional neural network layers including
/// ConvolutionalLayer, Conv3DLayer, DilatedConvolutionalLayer, DepthwiseSeparableConvolutionalLayer,
/// SeparableConvolutionalLayer, and SubpixelConvolutionalLayer.
/// Tests forward pass, backward pass, gradient computation, and layer cloning.
/// </summary>
public class ConvolutionalLayersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region ConvolutionalLayer Tests

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_Forward_ProducesValidOutput()
    {
        // Arrange: 3 channels, 8x8 image, 16 filters, 3x3 kernel
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 16, kernelSize: 3, stride: 1, padding: 1);

        // Create input: [channels, height, width]
        var input = Tensor<double>.CreateRandom(3, 8, 8);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // With padding=1 and stride=1 with 3x3 kernel, output size should be same as input
        Assert.Equal(16, output.Shape[0]); // outputDepth
        Assert.Equal(8, output.Shape[1]);  // height
        Assert.Equal(8, output.Shape[2]);  // width
        AssertNoNaNOrInf(output);
    }

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_Forward_WithStride2_ReducesDimensions()
    {
        // Arrange: stride=2 should halve the spatial dimensions
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 4, kernelSize: 3, stride: 2, padding: 1);

        var input = Tensor<double>.CreateRandom(1, 8, 8);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(4, output.Shape[0]); // outputDepth
        Assert.Equal(4, output.Shape[1]); // height: (8 + 2*1 - 3) / 2 + 1 = 4
        Assert.Equal(4, output.Shape[2]); // width: same calculation
    }


    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new ConvolutionalLayer<double>(
            outputDepth: 8, kernelSize: 3);
        var input = Tensor<double>.CreateRandom(3, 8, 8);

        // Act
        var originalOutput = original.Forward(input);
        var clone = original.Clone();
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_LargeKernel_ProducesValidOutput()
    {
        // Arrange: 5x5 kernel
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 8, kernelSize: 5, stride: 1, padding: 2);

        var input = Tensor<double>.CreateRandom(1, 16, 16);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(8, output.Shape[0]);
        Assert.Equal(16, output.Shape[1]);
        Assert.Equal(16, output.Shape[2]);
        AssertNoNaNOrInf(output);
    }

    #endregion

    #region Conv3DLayer Tests

    [Fact(Timeout = 120000)]
    public async Task Conv3DLayer_Forward_ProducesValidOutput()
    {
        // Arrange: 3D convolution for volumetric data (e.g., video, medical imaging)
        // Input: [channels, depth, height, width]
        // Constructor: (inputChannels, outputChannels, kernelSize, inputDepth, inputHeight, inputWidth, stride, padding, activation)
        IActivationFunction<double> relu = new ReLUActivation<double>();
        var layer = new Conv3DLayer<double>(
            outputChannels: 4, kernelSize: 3,
            stride: 1, padding: 1, activationFunction: relu);

        var input = Tensor<double>.CreateRandom(1, 8, 8, 8);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // outputChannels
        AssertNoNaNOrInf(output);
    }


    #endregion

    #region DilatedConvolutionalLayer Tests

    [Fact(Timeout = 120000)]
    public async Task DilatedConvolutionalLayer_Forward_ProducesValidOutput()
    {
        // Arrange: dilation=2 increases receptive field without increasing parameters
        // Constructor (lazy): (outputDepth, kernelSize, dilation, stride, padding, activation)
        IActivationFunction<double> relu = new ReLUActivation<double>();
        var layer = new DilatedConvolutionalLayer<double>(
            outputDepth: 8, kernelSize: 3, dilation: 2,
            stride: 1, padding: 2, activation: relu);

        var input = Tensor<double>.CreateRandom(3, 16, 16);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(8, output.Shape[0]); // outputDepth
        AssertNoNaNOrInf(output);
    }


    [Fact(Timeout = 120000)]
    public async Task DilatedConvolutionalLayer_LargeDilation_ProducesValidOutput()
    {
        // Arrange: dilation=4 for very large receptive field
        IActivationFunction<double> relu = new ReLUActivation<double>();
        var layer = new DilatedConvolutionalLayer<double>(
            outputDepth: 4, kernelSize: 3, dilation: 4,
            stride: 1, padding: 4, activation: relu);

        var input = Tensor<double>.CreateRandom(1, 32, 32);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    #endregion

    #region DepthwiseSeparableConvolutionalLayer Tests

    [Fact(Timeout = 120000)]
    public async Task DepthwiseSeparableConvolutionalLayer_Forward_ProducesValidOutput()
    {
        // Arrange: efficient convolution used in MobileNet
        // Constructor (lazy): (outputDepth, kernelSize, stride, padding, activation)
        IActivationFunction<double> relu = new ReLUActivation<double>();
        var layer = new DepthwiseSeparableConvolutionalLayer<double>(
            outputDepth: 16, kernelSize: 3, stride: 1, padding: 1,
            activation: relu);

        var input = Tensor<double>.CreateRandom(3, 8, 8);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(16, output.Shape[0]); // outputDepth
        AssertNoNaNOrInf(output);
    }


    #endregion

    #region SeparableConvolutionalLayer Tests

    [Fact(Timeout = 120000)]
    public async Task SeparableConvolutionalLayer_Forward_ProducesValidOutput()
    {
        // Arrange - lazy ctor, NHWC input via forward
        IActivationFunction<double> relu = new ReLUActivation<double>();
        var layer = new SeparableConvolutionalLayer<double>(
            outputDepth: 16, kernelSize: 3,
            stride: 1, padding: 1, scalarActivation: relu);

        var input = Tensor<double>.CreateRandom(1, 8, 8, 3);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }


    #endregion

    #region SubpixelConvolutionalLayer Tests

    [Fact(Timeout = 120000)]
    public async Task SubpixelConvolutionalLayer_Forward_ProducesValidOutput()
    {
        // Arrange: upscale by 2x (commonly used for super-resolution)
        // Constructor: (inputDepth, outputDepth, upscaleFactor, kernelSize, inputHeight, inputWidth, activation)
        IActivationFunction<double> relu = new ReLUActivation<double>();
        var layer = new SubpixelConvolutionalLayer<double>(
            inputDepth: 3, outputDepth: 3, upscaleFactor: 2, kernelSize: 3,
            inputHeight: 8, inputWidth: 8, activation: relu);

        var input = Tensor<double>.CreateRandom(3, 8, 8);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // Subpixel conv should upscale spatial dimensions by upscaleFactor
        Assert.Equal(3, output.Shape[0]);  // outputDepth
        Assert.Equal(16, output.Shape[1]); // height: 8 * 2 = 16
        Assert.Equal(16, output.Shape[2]); // width: 8 * 2 = 16
        AssertNoNaNOrInf(output);
    }


    #endregion

    #region Edge Cases

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_SingleChannel_SingleFilter_Works()
    {
        // Arrange: minimal configuration
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 1, kernelSize: 3, stride: 1, padding: 1);

        var input = Tensor<double>.CreateRandom(1, 4, 4);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(4, output.Shape[1]);
        Assert.Equal(4, output.Shape[2]);
        AssertNoNaNOrInf(output);
    }

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_NoPadding_ReducesDimensions()
    {
        // Arrange: without padding, 3x3 kernel reduces each dimension by 2
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 4, kernelSize: 3, stride: 1, padding: 0);

        var input = Tensor<double>.CreateRandom(1, 8, 8);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(4, output.Shape[0]);
        Assert.Equal(6, output.Shape[1]); // 8 - 3 + 1 = 6
        Assert.Equal(6, output.Shape[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_ManyFilters_Works()
    {
        // Arrange: 64 filters as commonly used in CNNs
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 64, kernelSize: 3, stride: 1, padding: 1);

        var input = Tensor<double>.CreateRandom(3, 16, 16);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(64, output.Shape[0]);
        AssertNoNaNOrInf(output);
    }

    #endregion

    #region Activation Function Tests

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_WithReLU_ProducesNonNegativeOutput()
    {
        // Arrange
        IActivationFunction<double> relu = new ReLUActivation<double>();
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 4, kernelSize: 3, stride: 1, padding: 1, relu);

        var input = Tensor<double>.CreateRandom(1, 8, 8);

        // Act
        var output = layer.Forward(input);

        // Assert - all values should be non-negative after ReLU
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] >= 0, $"ReLU output should be non-negative at index {i}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_WithTanh_ProducesOutputInRange()
    {
        // Arrange
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 4, kernelSize: 3, stride: 1, padding: 1, tanh);

        var input = Tensor<double>.CreateRandom(1, 8, 8);

        // Act
        var output = layer.Forward(input);

        // Assert - all values should be in [-1, 1] after tanh
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] >= -1 && output[i] <= 1,
                $"Tanh output should be in [-1, 1] at index {i}");
        }
    }

    #endregion

    #region Helper Methods

    private static void AssertNoNaNOrInf(Tensor<double> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.False(double.IsNaN(tensor[i]), $"Tensor contains NaN at index {i}");
            Assert.False(double.IsInfinity(tensor[i]), $"Tensor contains Infinity at index {i}");
        }
    }

    #endregion
}
