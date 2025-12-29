namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Xunit;

/// <summary>
/// Integration tests for ResNet-style block layers including ResidualLayer, BasicBlock,
/// BottleneckBlock, and related residual connection components.
/// </summary>
public class ResNetBlocksIntegrationTests
{
    #region Helper Methods

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var length = 1;
        foreach (var dim in shape) length *= dim;
        var flatData = new float[length];
        for (int i = 0; i < flatData.Length; i++)
        {
            flatData[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return new Tensor<float>(flatData, shape);
    }

    private static bool ContainsNaN(Tensor<float> tensor)
    {
        foreach (var value in tensor.Data)
        {
            if (float.IsNaN(value)) return true;
        }
        return false;
    }

    private static bool ContainsInf(Tensor<float> tensor)
    {
        foreach (var value in tensor.Data)
        {
            if (float.IsInfinity(value)) return true;
        }
        return false;
    }

    #endregion

    #region ResidualLayer Tests

    [Fact]
    public void ResidualLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [batch, features]
        int[] inputShape = [64];
        var innerLayer = new DenseLayer<float>(64, 64, (IActivationFunction<float>)new ReLUActivation<float>());
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, 64]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void ResidualLayer_ForwardPass_4D_ProducesValidOutput()
    {
        // Arrange - 4D input [batch, channels, height, width]
        int[] inputShape = [32, 8, 8];
        var innerLayer = new ConvolutionalLayer<float>(32, 32, 3, 8, 8, padding: 1);
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, 32, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void ResidualLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [32];
        var innerLayer = new DenseLayer<float>(32, 32, (IActivationFunction<float>)new ReLUActivation<float>());
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, 32]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void ResidualLayer_WithActivation_AppliesCorrectly()
    {
        // Arrange
        int[] inputShape = [32];
        var innerLayer = new DenseLayer<float>(32, 32, (IActivationFunction<float>)new IdentityActivation<float>());
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = CreateRandomTensor([4, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void ResidualLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [32];
        var innerLayer = new DenseLayer<float>(32, 32, (IActivationFunction<float>)new ReLUActivation<float>());
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, 32]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void ResidualLayer_GetParameters_ReturnsInnerLayerParameters()
    {
        // Arrange
        int[] inputShape = [32];
        var innerLayer = new DenseLayer<float>(32, 32, (IActivationFunction<float>)new ReLUActivation<float>());
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);

        // Act
        var parameters = layer.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0);
    }

    [Fact]
    public void ResidualLayer_SkipConnectionOnly_PassesThroughInput()
    {
        // Arrange - residual layer with no inner layer (should just pass through)
        int[] inputShape = [32];
        var layer = new ResidualLayer<float>(inputShape, null, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region BasicBlock Tests

    [Fact]
    public void BasicBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inChannels = 64;
        int outChannels = 64;
        int height = 16;
        int width = 16;
        var layer = new BasicBlock<float>(inChannels, outChannels, height, width);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, outChannels, height, width], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BasicBlock_WithDownsample_ProducesCorrectShape()
    {
        // Arrange - with stride 2 for downsampling
        int inChannels = 64;
        int outChannels = 128;
        int height = 16;
        int width = 16;
        int stride = 2;
        var layer = new BasicBlock<float>(inChannels, outChannels, height, width, stride);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert - output should be halved spatially
        Assert.Equal([2, outChannels, height / stride, width / stride], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BasicBlock_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inChannels = 32;
        int outChannels = 32;
        int height = 8;
        int width = 8;
        var layer = new BasicBlock<float>(inChannels, outChannels, height, width);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void BasicBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inChannels = 32;
        int outChannels = 32;
        int height = 8;
        int width = 8;
        var layer = new BasicBlock<float>(inChannels, outChannels, height, width);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void BasicBlock_GetParameters_ReturnsParameters()
    {
        // Arrange
        var layer = new BasicBlock<float>(32, 32, 8, 8);

        // Act
        var parameters = layer.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0);
    }

    #endregion

    #region BottleneckBlock Tests

    [Fact]
    public void BottleneckBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inChannels = 256;
        int bottleneckChannels = 64;
        int outChannels = 256;
        int height = 16;
        int width = 16;
        var layer = new BottleneckBlock<float>(inChannels, bottleneckChannels, outChannels, height, width);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, outChannels, height, width], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BottleneckBlock_WithExpansion_ProducesCorrectShape()
    {
        // Arrange - typical bottleneck with 4x expansion
        int inChannels = 64;
        int bottleneckChannels = 64;
        int outChannels = 256; // 4x expansion
        int height = 16;
        int width = 16;
        var layer = new BottleneckBlock<float>(inChannels, bottleneckChannels, outChannels, height, width);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, outChannels, height, width], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BottleneckBlock_WithStride_DownsamplesCorrectly()
    {
        // Arrange
        // BottleneckBlock constructor: (inChannels, baseChannels, stride, inputHeight, inputWidth, zeroInitResidual)
        // Output channels = baseChannels * 4 (expansion factor)
        int inChannels = 128;
        int baseChannels = 64;  // Output will be 256 channels (baseChannels * 4)
        int stride = 2;
        int height = 16;
        int width = 16;
        var layer = new BottleneckBlock<float>(inChannels, baseChannels, stride, height, width, true);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert - output channels = baseChannels * 4 = 256
        Assert.Equal([2, baseChannels * 4, height / stride, width / stride], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BottleneckBlock_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inChannels = 64;
        int bottleneckChannels = 16;
        int outChannels = 64;
        int height = 8;
        int width = 8;
        var layer = new BottleneckBlock<float>(inChannels, bottleneckChannels, outChannels, height, width);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void BottleneckBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inChannels = 64;
        int bottleneckChannels = 16;
        int outChannels = 64;
        int height = 8;
        int width = 8;
        var layer = new BottleneckBlock<float>(inChannels, bottleneckChannels, outChannels, height, width);
        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region HighwayLayer Tests

    [Fact]
    public void HighwayLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 64;
        var layer = new HighwayLayer<float>(inputSize, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void HighwayLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 32;
        var layer = new HighwayLayer<float>(inputSize, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, inputSize]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void HighwayLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 32;
        var layer = new HighwayLayer<float>(inputSize, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, inputSize]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void HighwayLayer_GetParameters_ReturnsParameters()
    {
        // Arrange
        var layer = new HighwayLayer<float>(32, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);

        // Act
        var parameters = layer.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0);
    }

    #endregion

    #region Stacked Block Tests

    [Fact]
    public void StackedResidualLayers_ProducesValidOutput()
    {
        // Arrange - stack 3 residual layers
        int[] inputShape = [32];
        var layers = new ResidualLayer<float>[3];
        for (int i = 0; i < 3; i++)
        {
            var inner = new DenseLayer<float>(32, 32, (IActivationFunction<float>)new ReLUActivation<float>());
            layers[i] = new ResidualLayer<float>(inputShape, inner, (IActivationFunction<float>?)null);
        }

        var input = CreateRandomTensor([4, 32]);

        // Act
        var output = input;
        foreach (var layer in layers)
        {
            output = layer.Forward(output);
        }

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    [Fact]
    public void StackedBasicBlocks_ProducesValidOutput()
    {
        // Arrange - stack 3 basic blocks (ResNet-18 style)
        int channels = 64;
        int height = 16;
        int width = 16;
        var blocks = new BasicBlock<float>[3];
        for (int i = 0; i < 3; i++)
        {
            blocks[i] = new BasicBlock<float>(channels, channels, height, width);
        }

        var input = CreateRandomTensor([2, channels, height, width]);

        // Act
        var output = input;
        foreach (var block in blocks)
        {
            output = block.Forward(output);
        }

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    [Fact]
    public void StackedBottleneckBlocks_ProducesValidOutput()
    {
        // Arrange - stack 3 bottleneck blocks (ResNet-50 style)
        int inChannels = 256;
        int bottleneck = 64;
        int outChannels = 256;
        int height = 16;
        int width = 16;
        var blocks = new BottleneckBlock<float>[3];
        for (int i = 0; i < 3; i++)
        {
            blocks[i] = new BottleneckBlock<float>(inChannels, bottleneck, outChannels, height, width);
        }

        var input = CreateRandomTensor([2, inChannels, height, width]);

        // Act
        var output = input;
        foreach (var block in blocks)
        {
            output = block.Forward(output);
        }

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void ResidualLayer_SmallBatch_HandlesCorrectly()
    {
        // Arrange - batch size 1
        int[] inputShape = [32];
        var innerLayer = new DenseLayer<float>(32, 32, (IActivationFunction<float>)new ReLUActivation<float>());
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([1, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BasicBlock_SmallSpatialSize_HandlesCorrectly()
    {
        // Arrange - small spatial dimensions
        var layer = new BasicBlock<float>(32, 32, 4, 4);
        var input = CreateRandomTensor([2, 32, 4, 4]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion
}
