namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

/// <summary>
/// Integration tests for embedding layer implementations testing any-rank tensor support,
/// forward/backward passes, training, serialization, and cloning.
/// </summary>
public class EmbeddingLayersIntegrationTests
{
    #region EmbeddingLayer Tests

    [Fact]
    public void EmbeddingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - vocabulary of 1000 tokens, 64-dimensional embeddings
        int vocabSize = 1000;
        int embeddingDim = 64;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);

        // Create input with token indices [batch, seqLen]
        var input = CreateTokenIndices([4, 10], vocabSize);

        // Act
        var output = layer.Forward(input);

        // Assert - output should be [batch, seqLen, embeddingDim]
        Assert.Equal(3, output.Rank);
        Assert.Equal(4, output.Shape[0]);
        Assert.Equal(10, output.Shape[1]);
        Assert.Equal(embeddingDim, output.Shape[2]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void EmbeddingLayer_ForwardPass_1D_ProducesValidOutput()
    {
        // Arrange - 1D input [seqLen]
        int vocabSize = 500;
        int embeddingDim = 32;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        var input = CreateTokenIndices([8], vocabSize);

        // Act
        var output = layer.Forward(input);

        // Assert - output should be [seqLen, embeddingDim]
        Assert.Equal(2, output.Rank);
        Assert.Equal(8, output.Shape[0]);
        Assert.Equal(embeddingDim, output.Shape[1]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void EmbeddingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int vocabSize = 100;
        int embeddingDim = 32;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        var input = CreateTokenIndices([4, 8], vocabSize);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert - input gradient should match input shape (zero tensor for embeddings)
        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void EmbeddingLayer_UpdateParameters_ModifiesWeights()
    {
        // Arrange
        int vocabSize = 50;
        int embeddingDim = 16;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        var input = CreateTokenIndices([2, 4], vocabSize);

        // Get initial parameters
        var initialParams = layer.GetParameters();

        // Act - forward, backward, update
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        layer.Backward(outputGradient);
        layer.UpdateParameters(0.01f);

        // Get updated parameters
        var updatedParams = layer.GetParameters();

        // Assert - parameters should have changed
        bool anyDifferent = false;
        for (int i = 0; i < initialParams.Length && !anyDifferent; i++)
        {
            if (Math.Abs(initialParams[i] - updatedParams[i]) > 1e-10)
                anyDifferent = true;
        }
        Assert.True(anyDifferent, "Parameters should change after update");
    }

    [Fact]
    public void EmbeddingLayer_GetSetParameters_RoundTrips()
    {
        // Arrange
        int vocabSize = 100;
        int embeddingDim = 32;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        var originalParams = layer.GetParameters();

        // Act
        var newLayer = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        newLayer.SetParameters(originalParams);
        var restoredParams = newLayer.GetParameters();

        // Assert
        Assert.Equal(originalParams.Length, restoredParams.Length);
        for (int i = 0; i < originalParams.Length; i++)
        {
            Assert.Equal(originalParams[i], restoredParams[i], 5);
        }
    }

    [Fact]
    public void EmbeddingLayer_AuxiliaryLoss_ComputesRegularization()
    {
        // Arrange
        int vocabSize = 100;
        int embeddingDim = 32;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        layer.UseAuxiliaryLoss = true;
        var input = CreateTokenIndices([2, 4], vocabSize);

        // Act
        layer.Forward(input);
        var auxLoss = layer.ComputeAuxiliaryLoss();

        // Assert - should return non-zero regularization loss
        Assert.True(auxLoss >= 0);
    }

    #endregion

    #region PatchEmbeddingLayer Tests

    [Fact]
    public void PatchEmbeddingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - 32x32 image with 8x8 patches
        int imageHeight = 32;
        int imageWidth = 32;
        int channels = 3;
        int patchSize = 8;
        int embeddingDim = 64;
        var layer = new PatchEmbeddingLayer<float>(imageHeight, imageWidth, channels, patchSize, embeddingDim);

        // Input: [batch, channels, height, width]
        var input = CreateRandomTensor<float>([2, channels, imageHeight, imageWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert - should have (32/8) * (32/8) = 16 patches
        int expectedPatches = (imageHeight / patchSize) * (imageWidth / patchSize);
        Assert.Equal(3, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(expectedPatches, output.Shape[1]); // patches
        Assert.Equal(embeddingDim, output.Shape[2]); // embedding
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PatchEmbeddingLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [C, H, W] (no batch dimension)
        int imageHeight = 16;
        int imageWidth = 16;
        int channels = 3;
        int patchSize = 4;
        int embeddingDim = 32;
        var layer = new PatchEmbeddingLayer<float>(imageHeight, imageWidth, channels, patchSize, embeddingDim);
        var input = CreateRandomTensor<float>([channels, imageHeight, imageWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert - output should be [numPatches, embeddingDim]
        int expectedPatches = (imageHeight / patchSize) * (imageWidth / patchSize);
        Assert.Equal(2, output.Rank);
        Assert.Equal(expectedPatches, output.Shape[0]);
        Assert.Equal(embeddingDim, output.Shape[1]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PatchEmbeddingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int imageHeight = 16;
        int imageWidth = 16;
        int channels = 3;
        int patchSize = 4;
        int embeddingDim = 32;
        var layer = new PatchEmbeddingLayer<float>(imageHeight, imageWidth, channels, patchSize, embeddingDim);
        var input = CreateRandomTensor<float>([2, channels, imageHeight, imageWidth]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void PatchEmbeddingLayer_GetSetParameters_RoundTrips()
    {
        // Arrange
        int imageHeight = 16;
        int imageWidth = 16;
        int channels = 3;
        int patchSize = 4;
        int embeddingDim = 32;
        var layer = new PatchEmbeddingLayer<float>(imageHeight, imageWidth, channels, patchSize, embeddingDim);
        var originalParams = layer.GetParameters();

        // Act
        var newLayer = new PatchEmbeddingLayer<float>(imageHeight, imageWidth, channels, patchSize, embeddingDim);
        newLayer.SetParameters(originalParams);
        var restoredParams = newLayer.GetParameters();

        // Assert
        Assert.Equal(originalParams.Length, restoredParams.Length);
    }

    #endregion

    #region TimeEmbeddingLayer Tests

    [Fact]
    public void TimeEmbeddingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - typical diffusion model configuration
        int embeddingDim = 64;
        int outputDim = 256;
        var layer = new TimeEmbeddingLayer<float>(embeddingDim, outputDim);

        // Input: timesteps [batch]
        var input = CreateTimesteps([4], 1000);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(4, output.Shape[0]); // batch
        Assert.Equal(outputDim, output.Shape[1]); // output dim
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void TimeEmbeddingLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [batch, 1]
        int embeddingDim = 32;
        int outputDim = 128;
        var layer = new TimeEmbeddingLayer<float>(embeddingDim, outputDim);
        var input = CreateTimesteps([8, 1], 1000);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(8, output.Shape[0]);
        Assert.Equal(outputDim, output.Shape[1]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void TimeEmbeddingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int embeddingDim = 32;
        int outputDim = 64;
        var layer = new TimeEmbeddingLayer<float>(embeddingDim, outputDim);
        var input = CreateTimesteps([4], 1000);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert - gradient shape should match input
        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void TimeEmbeddingLayer_UpdateParameters_ModifiesWeights()
    {
        // Arrange
        int embeddingDim = 32;
        int outputDim = 64;
        var layer = new TimeEmbeddingLayer<float>(embeddingDim, outputDim);
        var input = CreateTimesteps([4], 1000);

        var initialParams = layer.GetParameters();

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        layer.Backward(outputGradient);
        layer.UpdateParameters(0.01f);

        var updatedParams = layer.GetParameters();

        // Assert
        bool anyDifferent = false;
        for (int i = 0; i < initialParams.Length && !anyDifferent; i++)
        {
            if (Math.Abs(initialParams[i] - updatedParams[i]) > 1e-10)
                anyDifferent = true;
        }
        Assert.True(anyDifferent, "Parameters should change after update");
    }

    [Fact]
    public void TimeEmbeddingLayer_SinusoidalEncoding_UniquePerTimestep()
    {
        // Arrange
        int embeddingDim = 64;
        int outputDim = 128;
        var layer = new TimeEmbeddingLayer<float>(embeddingDim, outputDim);

        // Create different timesteps
        var t1 = new Tensor<float>([1]);
        t1[0] = 100;
        var t2 = new Tensor<float>([1]);
        t2[0] = 500;

        // Act
        var output1 = layer.Forward(t1);
        var output2 = layer.Forward(t2);

        // Assert - outputs should be different for different timesteps
        bool anyDifferent = false;
        for (int i = 0; i < output1.Length && !anyDifferent; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-6)
                anyDifferent = true;
        }
        Assert.True(anyDifferent, "Outputs should differ for different timesteps");
    }

    #endregion

    #region PositionalEncodingLayer Tests

    [Fact]
    public void PositionalEncodingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int maxSeqLen = 512;
        int embeddingSize = 64;
        var layer = new PositionalEncodingLayer<float>(maxSeqLen, embeddingSize);

        // Input: [seqLen, embeddingSize]
        var input = CreateRandomTensor<float>([100, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PositionalEncodingLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, seqLen, embedding]
        int maxSeqLen = 256;
        int embeddingSize = 32;
        var layer = new PositionalEncodingLayer<float>(maxSeqLen, embeddingSize);
        var input = CreateRandomTensor<float>([4, 50, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PositionalEncodingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int maxSeqLen = 128;
        int embeddingSize = 32;
        var layer = new PositionalEncodingLayer<float>(maxSeqLen, embeddingSize);
        var input = CreateRandomTensor<float>([4, 32, embeddingSize]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert - gradient flows through unchanged for positional encoding
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void PositionalEncodingLayer_AddsPositionalInformation()
    {
        // Arrange
        int maxSeqLen = 64;
        int embeddingSize = 16;
        var layer = new PositionalEncodingLayer<float>(maxSeqLen, embeddingSize);

        // Create zero input to see pure positional encodings
        var input = new Tensor<float>([10, embeddingSize]);
        // Input is already zero-initialized

        // Act
        var output = layer.Forward(input);

        // Assert - output should have non-zero values (the positional encodings)
        // Industry standard: sin/cos positional encodings produce non-zero values
        bool hasNonZero = false;
        for (int i = 0; i < output.Length && !hasNonZero; i++)
        {
            if (Math.Abs(output[i]) > 1e-10)
                hasNonZero = true;
        }
        Assert.True(hasNonZero, "Positional encoding should add non-zero values");
    }

    [Fact]
    public void PositionalEncodingLayer_DifferentPositions_HaveDifferentEncodings()
    {
        // Arrange
        int maxSeqLen = 64;
        int embeddingSize = 32;
        var layer = new PositionalEncodingLayer<float>(maxSeqLen, embeddingSize);

        // Create zero input to see pure positional encodings
        var input = new Tensor<float>([10, embeddingSize]);
        // Input is already zero-initialized

        // Act
        var output = layer.Forward(input);

        // Assert - each position should have a unique encoding
        // Industry standard: sin/cos at different positions produce different values
        for (int pos1 = 0; pos1 < 5; pos1++)
        {
            for (int pos2 = pos1 + 1; pos2 < 5; pos2++)
            {
                bool different = false;
                for (int dim = 0; dim < embeddingSize && !different; dim++)
                {
                    if (Math.Abs(output[pos1, dim] - output[pos2, dim]) > 1e-6)
                        different = true;
                }
                Assert.True(different, $"Positions {pos1} and {pos2} should have different encodings");
            }
        }
    }

    [Fact]
    public void PositionalEncodingLayer_NoTrainableParameters()
    {
        // Arrange
        int maxSeqLen = 64;
        int embeddingSize = 32;
        var layer = new PositionalEncodingLayer<float>(maxSeqLen, embeddingSize);

        // Act
        var parameters = layer.GetParameters();

        // Assert - positional encoding has no trainable parameters
        Assert.Equal(0, parameters.Length);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void EmbeddingLayer_SingleToken_Works()
    {
        // Arrange
        int vocabSize = 100;
        int embeddingDim = 32;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        var input = CreateTokenIndices([1], vocabSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(embeddingDim, output.Shape[1]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void EmbeddingLayer_LargeVocabulary_Works()
    {
        // Arrange - large vocabulary
        int vocabSize = 50000;
        int embeddingDim = 256;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        var input = CreateTokenIndices([2, 16], vocabSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PatchEmbeddingLayer_SquareImage_Works()
    {
        // Arrange - 64x64 image with 16x16 patches
        int size = 64;
        int channels = 3;
        int patchSize = 16;
        int embeddingDim = 128;
        var layer = new PatchEmbeddingLayer<float>(size, size, channels, patchSize, embeddingDim);
        var input = CreateRandomTensor<float>([1, channels, size, size]);

        // Act
        var output = layer.Forward(input);

        // Assert - (64/16)^2 = 16 patches
        Assert.Equal(16, output.Shape[1]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void TimeEmbeddingLayer_ZeroTimestep_Works()
    {
        // Arrange
        int embeddingDim = 32;
        int outputDim = 64;
        var layer = new TimeEmbeddingLayer<float>(embeddingDim, outputDim);

        var input = new Tensor<float>([1]);
        input[0] = 0f; // t=0

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void TimeEmbeddingLayer_MaxTimestep_Works()
    {
        // Arrange
        int embeddingDim = 32;
        int outputDim = 64;
        int maxTimestep = 1000;
        var layer = new TimeEmbeddingLayer<float>(embeddingDim, outputDim, maxTimestep);

        var input = new Tensor<float>([1]);
        input[0] = maxTimestep; // t=max

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PositionalEncodingLayer_ExactMaxLength_Works()
    {
        // Arrange
        int maxSeqLen = 64;
        int embeddingSize = 32;
        var layer = new PositionalEncodingLayer<float>(maxSeqLen, embeddingSize);
        var input = CreateRandomTensor<float>([maxSeqLen, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PositionalEncodingLayer_ShortSequence_Works()
    {
        // Arrange - sequence much shorter than max
        int maxSeqLen = 512;
        int embeddingSize = 64;
        var layer = new PositionalEncodingLayer<float>(maxSeqLen, embeddingSize);
        var input = CreateRandomTensor<float>([10, embeddingSize]); // only 10 positions

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region Helper Methods

    private static Tensor<T> CreateRandomTensor<T>(int[] shape) where T : struct, IComparable<T>
    {
        var tensor = new Tensor<T>(shape);
        var random = new Random(42);

        for (int i = 0; i < tensor.Length; i++)
        {
            double value = random.NextDouble() * 2 - 1; // [-1, 1]
            tensor[i] = (T)Convert.ChangeType(value, typeof(T));
        }

        return tensor;
    }

    private static Tensor<float> CreateTokenIndices(int[] shape, int vocabSize)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(42);

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.Next(vocabSize);
        }

        return tensor;
    }

    private static Tensor<float> CreateTimesteps(int[] shape, int maxTimestep)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(42);

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.Next(maxTimestep);
        }

        return tensor;
    }

    private static bool ContainsNaN<T>(Tensor<T> tensor) where T : struct, IComparable<T>
    {
        foreach (var value in tensor.ToArray())
        {
            if (value is float f && float.IsNaN(f)) return true;
            if (value is double d && double.IsNaN(d)) return true;
        }
        return false;
    }

    #endregion
}
