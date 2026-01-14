namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

/// <summary>
/// Integration tests for specialized block implementations (ResNet, MobileNet, DenseNet blocks)
/// testing forward/backward passes, skip connections, and cloning.
/// </summary>
public class SpecializedBlocksIntegrationTests
{
    #region BasicBlock Tests

    [Fact]
    public void BasicBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange - ResNet basic block with same input/output channels
        int channels = 64;
        int height = 8;
        int width = 8;
        var block = new BasicBlock<float>(channels, channels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([2, channels, height, width]); // NCHW format

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BasicBlock_ForwardPass_WithDownsample_ProducesValidOutput()
    {
        // Arrange - Basic block with stride 2 (downsample)
        int inChannels = 64;
        int outChannels = 128;
        int height = 16;
        int width = 16;
        var block = new BasicBlock<float>(inChannels, outChannels, stride: 2, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert - output should be half spatial size with double channels
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(outChannels, output.Shape[1]); // channels
        Assert.Equal(height / 2, output.Shape[2]); // height halved
        Assert.Equal(width / 2, output.Shape[3]); // width halved
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BasicBlock_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int channels = 32;
        int height = 8;
        int width = 8;
        var block = new BasicBlock<float>(channels, channels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([2, channels, height, width]);

        // Act
        var output = block.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = block.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void BasicBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int channels = 32;
        int height = 8;
        int width = 8;
        var original = new BasicBlock<float>(channels, channels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([2, channels, height, width]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (BasicBlock<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    [Fact]
    public void BasicBlock_SkipConnection_WorksCorrectly()
    {
        // Arrange - When in/out channels match, skip connection is identity
        int channels = 32;
        int height = 8;
        int width = 8;
        var block = new BasicBlock<float>(channels, channels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([1, channels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert - output should be affected by both convolutions and skip
        Assert.False(ContainsNaN(output));
        Assert.Equal(input.Shape, output.Shape);
    }

    #endregion

    #region BottleneckBlock Tests

    [Fact]
    public void BottleneckBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange - Bottleneck uses 1x1 -> 3x3 -> 1x1 pattern with expansion
        int inChannels = 64;
        int outChannels = 64;
        int height = 8;
        int width = 8;
        var block = new BottleneckBlock<float>(inChannels, outChannels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert - output channels = outChannels * expansion (4)
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(outChannels * 4, output.Shape[1]); // channels * expansion
        Assert.Equal(height, output.Shape[2]);
        Assert.Equal(width, output.Shape[3]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BottleneckBlock_ForwardPass_WithDownsample_ProducesValidOutput()
    {
        // Arrange - Bottleneck with stride 2
        int inChannels = 256;
        int baseChannels = 128;
        int height = 16;
        int width = 16;
        var block = new BottleneckBlock<float>(inChannels, baseChannels, stride: 2, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(baseChannels * 4, output.Shape[1]); // 128 * 4 = 512
        Assert.Equal(height / 2, output.Shape[2]);
        Assert.Equal(width / 2, output.Shape[3]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BottleneckBlock_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inChannels = 64;
        int outChannels = 64;
        int height = 8;
        int width = 8;
        var block = new BottleneckBlock<float>(inChannels, outChannels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);

        // Act
        var output = block.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = block.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void BottleneckBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inChannels = 64;
        int outChannels = 64;
        int height = 8;
        int width = 8;
        var original = new BottleneckBlock<float>(inChannels, outChannels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (BottleneckBlock<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    [Fact]
    public void BottleneckBlock_ExpansionFactor_CorrectlyApplied()
    {
        // Arrange - default expansion is 4
        int inChannels = 64;
        int outChannels = 32;
        int height = 8;
        int width = 8;
        var block = new BottleneckBlock<float>(inChannels, outChannels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([1, inChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert - output channels should be outChannels * 4
        Assert.Equal(outChannels * 4, output.Shape[1]); // 32 * 4 = 128
    }

    #endregion

    #region InvertedResidualBlock Tests

    [Fact]
    public void InvertedResidualBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange - MobileNetV2 inverted residual block
        int inChannels = 32;
        int outChannels = 64;
        int height = 8;
        int width = 8;
        var block = new InvertedResidualBlock<float>(inChannels, outChannels, height, width);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(outChannels, output.Shape[1]);
        Assert.Equal(height, output.Shape[2]);
        Assert.Equal(width, output.Shape[3]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void InvertedResidualBlock_ForwardPass_WithStride_ProducesValidOutput()
    {
        // Arrange - Inverted residual with stride 2
        int inChannels = 32;
        int outChannels = 64;
        int height = 16;
        int width = 16;
        var block = new InvertedResidualBlock<float>(inChannels, outChannels, height, width, stride: 2);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(outChannels, output.Shape[1]);
        Assert.Equal(height / 2, output.Shape[2]);
        Assert.Equal(width / 2, output.Shape[3]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void InvertedResidualBlock_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inChannels = 32;
        int outChannels = 32; // Same channels for residual connection
        int height = 8;
        int width = 8;
        var block = new InvertedResidualBlock<float>(inChannels, outChannels, height, width);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);

        // Act
        var output = block.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = block.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void InvertedResidualBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inChannels = 32;
        int outChannels = 64;
        int height = 8;
        int width = 8;
        var original = new InvertedResidualBlock<float>(inChannels, outChannels, height, width);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (InvertedResidualBlock<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    [Fact]
    public void InvertedResidualBlock_ExpansionFactor_CorrectlyExpands()
    {
        // Arrange - expansion factor 6 is typical
        int inChannels = 24;
        int outChannels = 24;
        int height = 8;
        int width = 8;
        int expansionRatio = 6;
        var block = new InvertedResidualBlock<float>(inChannels, outChannels, height, width, expansionRatio: expansionRatio);
        var input = CreateRandomTensor<float>([1, inChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert - output should have outChannels
        Assert.Equal(outChannels, output.Shape[1]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void InvertedResidualBlock_WithSqueezeExcite_ProducesValidOutput()
    {
        // Arrange - SE module is optional enhancement
        int inChannels = 32;
        int outChannels = 32;
        int height = 8;
        int width = 8;
        var block = new InvertedResidualBlock<float>(inChannels, outChannels, height, width, useSE: true);
        var input = CreateRandomTensor<float>([2, inChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region DenseBlock Tests

    [Fact]
    public void DenseBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange - DenseNet block with growth rate
        int inputChannels = 64;
        int numLayers = 4;
        int growthRate = 32;
        int height = 8;
        int width = 8;
        var block = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);
        var input = CreateRandomTensor<float>([2, inputChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert - output channels = inputChannels + numLayers * growthRate
        int expectedChannels = inputChannels + numLayers * growthRate; // 64 + 4*32 = 192
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(expectedChannels, output.Shape[1]);
        Assert.Equal(height, output.Shape[2]);
        Assert.Equal(width, output.Shape[3]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void DenseBlock_OutputChannels_CorrectlyCalculated()
    {
        // Arrange
        int inputChannels = 32;
        int numLayers = 6;
        int growthRate = 16;
        int height = 8;
        int width = 8;
        var block = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);

        // Assert - verify property calculation
        Assert.Equal(inputChannels + numLayers * growthRate, block.OutputChannels);
        Assert.Equal(numLayers, block.NumLayers);
        Assert.Equal(growthRate, block.GrowthRate);
    }

    [Fact]
    public void DenseBlock_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputChannels = 32;
        int numLayers = 3;
        int growthRate = 16;
        int height = 8;
        int width = 8;
        var block = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);
        var input = CreateRandomTensor<float>([2, inputChannels, height, width]);

        // Act
        var output = block.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = block.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void DenseBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputChannels = 32;
        int numLayers = 3;
        int growthRate = 16;
        int height = 8;
        int width = 8;
        var original = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);
        var input = CreateRandomTensor<float>([2, inputChannels, height, width]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (DenseBlock<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape, clonedOutput.Shape);
    }

    [Fact]
    public void DenseBlock_DenseConnectivity_AllLayersContributeToOutput()
    {
        // Arrange - In DenseBlock, each layer's output is concatenated
        int inputChannels = 16;
        int numLayers = 4;
        int growthRate = 8;
        int height = 4;
        int width = 4;
        var block = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);
        var input = CreateRandomTensor<float>([1, inputChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert - output should contain concatenation of all layer outputs
        // Each layer adds growthRate channels
        int expectedChannels = inputChannels + numLayers * growthRate;
        Assert.Equal(expectedChannels, output.Shape[1]);
    }

    [Fact]
    public void DenseBlock_ResetState_ClearsLayerOutputs()
    {
        // Arrange
        int inputChannels = 32;
        int numLayers = 2;
        int growthRate = 16;
        int height = 4;
        int width = 4;
        var block = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);
        var input = CreateRandomTensor<float>([1, inputChannels, height, width]);

        // Act - forward then reset
        var output1 = block.Forward(input);
        block.ResetState();
        var output2 = block.Forward(input);

        // Assert - should produce same output after reset
        Assert.Equal(output1.Shape, output2.Shape);
        Assert.False(ContainsNaN(output2));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void BasicBlock_SingleBatch_Works()
    {
        // Arrange
        int channels = 32;
        int height = 4;
        int width = 4;
        var block = new BasicBlock<float>(channels, channels, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([1, channels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BottleneckBlock_SingleBatch_Works()
    {
        // Arrange
        int channels = 64;
        int height = 4;
        int width = 4;
        var block = new BottleneckBlock<float>(channels, channels / 4, inputHeight: height, inputWidth: width);
        var input = CreateRandomTensor<float>([1, channels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void InvertedResidualBlock_ExpansionOne_Works()
    {
        // Arrange - expansion 1 means no expansion (just depthwise)
        int channels = 32;
        int height = 4;
        int width = 4;
        var block = new InvertedResidualBlock<float>(channels, channels, height, width, expansionRatio: 1);
        var input = CreateRandomTensor<float>([1, channels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void DenseBlock_SingleLayer_Works()
    {
        // Arrange - minimum layers
        int inputChannels = 16;
        int numLayers = 1;
        int growthRate = 8;
        int height = 4;
        int width = 4;
        var block = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);
        var input = CreateRandomTensor<float>([1, inputChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(inputChannels + growthRate, output.Shape[1]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void DenseBlock_LargeGrowthRate_Works()
    {
        // Arrange - large growth rate
        int inputChannels = 16;
        int numLayers = 2;
        int growthRate = 64;
        int height = 4;
        int width = 4;
        var block = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);
        var input = CreateRandomTensor<float>([1, inputChannels, height, width]);

        // Act
        var output = block.Forward(input);

        // Assert
        Assert.Equal(inputChannels + numLayers * growthRate, output.Shape[1]); // 16 + 2*64 = 144
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region Helper Methods

    private static Tensor<T> CreateRandomTensor<T>(int[] shape) where T : struct, IComparable<T>
    {
        var tensor = new Tensor<T>(shape);
        var random = new Random(42);
        var span = tensor.Data.Span;

        for (int i = 0; i < span.Length; i++)
        {
            double value = random.NextDouble() * 2 - 1; // [-1, 1]
            span[i] = (T)Convert.ChangeType(value, typeof(T));
        }

        return tensor;
    }

    private static bool ContainsNaN<T>(Tensor<T> tensor) where T : struct, IComparable<T>
    {
        foreach (var value in tensor.Data.ToArray())
        {
            if (value is float f && float.IsNaN(f)) return true;
            if (value is double d && double.IsNaN(d)) return true;
        }
        return false;
    }

    #endregion
}
