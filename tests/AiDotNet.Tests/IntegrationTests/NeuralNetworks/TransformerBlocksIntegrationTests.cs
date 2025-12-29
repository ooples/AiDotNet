namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Xunit;

/// <summary>
/// Integration tests for Transformer architecture layers including TransformerEncoderLayer,
/// TransformerDecoderLayer, FeedForwardLayer, and related components.
/// </summary>
public class TransformerBlocksIntegrationTests
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

    #region TransformerEncoderLayer Tests

    [Fact]
    public void TransformerEncoderLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [batch, embedding]
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);
        var input = CreateRandomTensor([2, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    [Fact]
    public void TransformerEncoderLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, sequence, embedding]
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);
        var input = CreateRandomTensor([2, 10, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    [Fact]
    public void TransformerEncoderLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int embeddingSize = 32;
        int numHeads = 4;
        int feedForwardDim = 64;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);
        var input = CreateRandomTensor([2, 8, embeddingSize]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void TransformerEncoderLayer_GetParameters_ReturnsParameters()
    {
        // Arrange
        int embeddingSize = 32;
        int numHeads = 4;
        int feedForwardDim = 64;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);

        // Act
        var parameters = layer.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0);
    }

    [Fact]
    public void TransformerEncoderLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int embeddingSize = 32;
        int numHeads = 4;
        int feedForwardDim = 64;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);
        var input = CreateRandomTensor([2, 8, embeddingSize]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void TransformerEncoderLayer_SupportsTraining_ReturnsTrue()
    {
        // Arrange
        var layer = new TransformerEncoderLayer<float>(64, 4, 128);

        // Assert
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void TransformerEncoderLayer_DifferentHeadConfigurations_Work()
    {
        // Test various head configurations
        var configs = new[] { (64, 1), (64, 2), (64, 4), (64, 8), (128, 8) };

        foreach (var (embeddingSize, numHeads) in configs)
        {
            // Arrange
            var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, embeddingSize * 2);
            var input = CreateRandomTensor([2, 10, embeddingSize]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(input.Shape, output.Shape);
            Assert.False(ContainsNaN(output));
        }
    }

    #endregion

    #region TransformerDecoderLayer Tests

    [Fact]
    public void TransformerDecoderLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [batch, embedding]
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        var layer = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, 0, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    [Fact]
    public void TransformerDecoderLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, sequence, embedding]
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        var layer = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, 0, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, 10, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    [Fact]
    public void TransformerDecoderLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int embeddingSize = 32;
        int numHeads = 4;
        int feedForwardDim = 64;
        var layer = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, 0, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, 8, embeddingSize]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void TransformerDecoderLayer_WithEncoderOutput_ProducesValidOutput()
    {
        // Arrange - decoder with cross-attention to encoder output
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        int sequenceLength = 10;
        var layer = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, sequenceLength, (IActivationFunction<float>?)null);
        var decoderInput = CreateRandomTensor([2, 8, embeddingSize]);
        var encoderOutput = CreateRandomTensor([2, 12, embeddingSize], seed: 456);

        // Act
        var output = layer.Forward(decoderInput, encoderOutput);

        // Assert
        Assert.Equal([2, 8, embeddingSize], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void TransformerDecoderLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int embeddingSize = 32;
        var layer = new TransformerDecoderLayer<float>(embeddingSize, 4, 64, 0, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, 8, embeddingSize]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void TransformerDecoderLayer_GetParameters_ReturnsParameters()
    {
        // Arrange
        var layer = new TransformerDecoderLayer<float>(32, 4, 64, 0, (IActivationFunction<float>?)null);

        // Act
        var parameters = layer.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0);
    }

    #endregion

    #region FeedForwardLayer Tests

    [Fact]
    public void FeedForwardLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 64;
        int hiddenSize = 128;
        var layer = new FeedForwardLayer<float>(inputSize, hiddenSize, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = CreateRandomTensor([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([4, hiddenSize], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void FeedForwardLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input for sequence data
        int inputSize = 64;
        int hiddenSize = 128;
        var layer = new FeedForwardLayer<float>(inputSize, hiddenSize, (IActivationFunction<float>)new GELUActivation<float>());
        var input = CreateRandomTensor([2, 10, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, 10, hiddenSize], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void FeedForwardLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 32;
        int hiddenSize = 64;
        var layer = new FeedForwardLayer<float>(inputSize, hiddenSize, (IActivationFunction<float>)new ReLUActivation<float>());
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
    public void FeedForwardLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 32;
        int hiddenSize = 64;
        var layer = new FeedForwardLayer<float>(inputSize, hiddenSize, (IActivationFunction<float>)new ReLUActivation<float>());
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
    public void FeedForwardLayer_WithDifferentActivations_Work()
    {
        // Test with various activations
        var activations = new IActivationFunction<float>[]
        {
            new ReLUActivation<float>(),
            new GELUActivation<float>(),
            new TanhActivation<float>(),
            new SigmoidActivation<float>()
        };

        foreach (var activation in activations)
        {
            // Arrange
            var layer = new FeedForwardLayer<float>(32, 64, activation);
            var input = CreateRandomTensor([4, 32]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal([4, 64], output.Shape);
            Assert.False(ContainsNaN(output));
        }
    }

    #endregion

    #region DecoderLayer Tests

    [Fact]
    public void DecoderLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int embeddingSize = 64;
        int attentionSize = 64;
        int feedForwardSize = 128;
        var layer = new DecoderLayer<float>(embeddingSize, attentionSize, feedForwardSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, 10, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void DecoderLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int embeddingSize = 32;
        int attentionSize = 32;
        int feedForwardSize = 64;
        var layer = new DecoderLayer<float>(embeddingSize, attentionSize, feedForwardSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, 8, embeddingSize]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void DecoderLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int embeddingSize = 32;
        int attentionSize = 32;
        int feedForwardSize = 64;
        var layer = new DecoderLayer<float>(embeddingSize, attentionSize, feedForwardSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, 8, embeddingSize]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region Integration Tests - Stacked Layers

    [Fact]
    public void TransformerEncoder_StackedLayers_ProducesValidOutput()
    {
        // Arrange - Stack multiple encoder layers
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        int numLayers = 3;

        var encoderLayers = new TransformerEncoderLayer<float>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            encoderLayers[i] = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);
        }

        var input = CreateRandomTensor([2, 10, embeddingSize]);

        // Act - Pass through all layers
        var output = input;
        foreach (var layer in encoderLayers)
        {
            output = layer.Forward(output);
        }

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    [Fact]
    public void TransformerDecoder_StackedLayers_ProducesValidOutput()
    {
        // Arrange - Stack multiple decoder layers
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 128;
        int numLayers = 3;

        var decoderLayers = new TransformerDecoderLayer<float>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            decoderLayers[i] = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, 0, (IActivationFunction<float>?)null);
        }

        var input = CreateRandomTensor([2, 10, embeddingSize]);

        // Act - Pass through all layers
        var output = input;
        foreach (var layer in decoderLayers)
        {
            output = layer.Forward(output);
        }

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.False(ContainsInf(output));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void TransformerEncoderLayer_SmallBatch_HandlesCorrectly()
    {
        // Arrange - batch size of 1
        int embeddingSize = 64;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, 4, 128);
        var input = CreateRandomTensor([1, 5, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void TransformerDecoderLayer_ShortSequence_HandlesCorrectly()
    {
        // Arrange - sequence length of 1
        int embeddingSize = 64;
        var layer = new TransformerDecoderLayer<float>(embeddingSize, 4, 128, 0, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([2, 1, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void FeedForwardLayer_LargeHiddenDim_HandlesCorrectly()
    {
        // Arrange - large expansion ratio (4x)
        int inputSize = 64;
        int hiddenSize = 256;
        var layer = new FeedForwardLayer<float>(inputSize, hiddenSize, (IActivationFunction<float>)new GELUActivation<float>());
        var input = CreateRandomTensor([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([4, hiddenSize], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion
}
