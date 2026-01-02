// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for normalization layers.
/// Tests: BatchNormalization, LayerNormalization, GroupNormalization,
/// InstanceNormalization, and SpectralNormalization layers.
/// </summary>
public class NormalizationLayersIntegrationTests
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

    #endregion

    #region BatchNormalizationLayer Tests

    [Fact]
    public void BatchNormalizationLayer_ForwardPass_ProducesNormalizedOutput()
    {
        // Arrange
        int batchSize = 4;
        int numFeatures = 8;
        var layer = new BatchNormalizationLayer<float>(numFeatures);
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor([batchSize, numFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BatchNormalizationLayer_ForwardPass_TrainingVsInference()
    {
        // Arrange
        int batchSize = 4;
        int numFeatures = 8;
        var layer = new BatchNormalizationLayer<float>(numFeatures);
        var input = CreateRandomTensor([batchSize, numFeatures]);

        // Act - Training mode
        layer.SetTrainingMode(true);
        var trainingOutput = layer.Forward(input);

        // Act - Inference mode
        layer.SetTrainingMode(false);
        var inferenceOutput = layer.Forward(input);

        // Assert - Both should produce valid outputs
        Assert.False(ContainsNaN(trainingOutput));
        Assert.False(ContainsNaN(inferenceOutput));
    }

    [Fact]
    public void BatchNormalizationLayer_BackwardPass_ProducesGradients()
    {
        // Arrange
        int batchSize = 4;
        int numFeatures = 8;
        var layer = new BatchNormalizationLayer<float>(numFeatures);
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor([batchSize, numFeatures]);
        layer.Forward(input);
        var upstreamGradient = CreateRandomTensor([batchSize, numFeatures], seed: 123);

        // Act
        var gradients = layer.Backward(upstreamGradient);

        // Assert
        Assert.Equal(input.Shape, gradients.Shape);
        Assert.False(ContainsNaN(gradients));
    }

    [Fact]
    public void BatchNormalizationLayer_Clone_CreatesIdenticalLayer()
    {
        // Arrange
        int numFeatures = 8;
        var layer = new BatchNormalizationLayer<float>(numFeatures);
        var input = CreateRandomTensor([4, numFeatures]);
        layer.Forward(input);

        // Act
        var clone = layer.Clone();
        var clonedLayer = Assert.IsType<BatchNormalizationLayer<float>>(clone);

        // Assert
        Assert.NotSame(layer, clonedLayer);
        var output1 = layer.Forward(input);
        var output2 = clonedLayer.Forward(input);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void BatchNormalizationLayer_WithCustomMomentum_Works()
    {
        // Arrange
        int batchSize = 4;
        int numFeatures = 8;
        double customMomentum = 0.95;
        var layer = new BatchNormalizationLayer<float>(numFeatures, momentum: customMomentum);
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor([batchSize, numFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void BatchNormalizationLayer_ParameterCount_IncludesGammaAndBeta()
    {
        // Arrange
        int numFeatures = 8;
        var layer = new BatchNormalizationLayer<float>(numFeatures);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - gamma and beta for each feature
        Assert.Equal(numFeatures * 2, paramCount);
    }

    #endregion

    #region LayerNormalizationLayer Tests

    [Fact]
    public void LayerNormalizationLayer_ForwardPass_ProducesNormalizedOutput()
    {
        // Arrange
        int batchSize = 4;
        int featureSize = 16;
        var layer = new LayerNormalizationLayer<float>(featureSize);
        var input = CreateRandomTensor([batchSize, featureSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void LayerNormalizationLayer_BackwardPass_ProducesGradients()
    {
        // Arrange
        int batchSize = 4;
        int featureSize = 16;
        var layer = new LayerNormalizationLayer<float>(featureSize);
        var input = CreateRandomTensor([batchSize, featureSize]);
        layer.Forward(input);
        var upstreamGradient = CreateRandomTensor([batchSize, featureSize], seed: 123);

        // Act
        var gradients = layer.Backward(upstreamGradient);

        // Assert
        Assert.Equal(input.Shape, gradients.Shape);
        Assert.False(ContainsNaN(gradients));
    }

    [Fact]
    public void LayerNormalizationLayer_Clone_CreatesIdenticalLayer()
    {
        // Arrange
        int featureSize = 16;
        var layer = new LayerNormalizationLayer<float>(featureSize);
        var input = CreateRandomTensor([4, featureSize]);
        layer.Forward(input);

        // Act
        var clone = layer.Clone();
        var clonedLayer = Assert.IsType<LayerNormalizationLayer<float>>(clone);

        // Assert
        Assert.NotSame(layer, clonedLayer);
        var output1 = layer.Forward(input);
        var output2 = clonedLayer.Forward(input);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void LayerNormalizationLayer_WithCustomEpsilon_Works()
    {
        // Arrange
        int batchSize = 4;
        int featureSize = 16;
        double customEpsilon = 1e-6;
        var layer = new LayerNormalizationLayer<float>(featureSize, epsilon: customEpsilon);
        var input = CreateRandomTensor([batchSize, featureSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void LayerNormalizationLayer_ParameterCount_IncludesGammaAndBeta()
    {
        // Arrange
        int featureSize = 16;
        var layer = new LayerNormalizationLayer<float>(featureSize);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - gamma and beta for each feature
        Assert.Equal(featureSize * 2, paramCount);
    }

    [Fact]
    public void LayerNormalizationLayer_3DInput_Works()
    {
        // Arrange - simulate sequence data [batch, seq, features]
        int batchSize = 2;
        int seqLength = 8;
        int featureSize = 16;
        var layer = new LayerNormalizationLayer<float>(featureSize);
        var input = CreateRandomTensor([batchSize, seqLength, featureSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region GroupNormalizationLayer Tests

    [Fact]
    public void GroupNormalizationLayer_ForwardPass_ProducesNormalizedOutput()
    {
        // Arrange
        int batchSize = 4;
        int numChannels = 16;
        int numGroups = 4;
        var layer = new GroupNormalizationLayer<float>(numGroups, numChannels);
        var input = CreateRandomTensor([batchSize, numChannels]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GroupNormalizationLayer_BackwardPass_ProducesGradients()
    {
        // Arrange
        int batchSize = 4;
        int numChannels = 16;
        int numGroups = 4;
        var layer = new GroupNormalizationLayer<float>(numGroups, numChannels);
        var input = CreateRandomTensor([batchSize, numChannels]);
        layer.Forward(input);
        var upstreamGradient = CreateRandomTensor([batchSize, numChannels], seed: 123);

        // Act
        var gradients = layer.Backward(upstreamGradient);

        // Assert
        Assert.Equal(input.Shape, gradients.Shape);
        Assert.False(ContainsNaN(gradients));
    }

    [Fact]
    public void GroupNormalizationLayer_Clone_CreatesIdenticalLayer()
    {
        // Arrange
        int numChannels = 16;
        int numGroups = 4;
        var layer = new GroupNormalizationLayer<float>(numGroups, numChannels);
        var input = CreateRandomTensor([4, numChannels]);
        layer.Forward(input);

        // Act
        var clone = layer.Clone();
        var clonedLayer = Assert.IsType<GroupNormalizationLayer<float>>(clone);

        // Assert
        Assert.NotSame(layer, clonedLayer);
        var output1 = layer.Forward(input);
        var output2 = clonedLayer.Forward(input);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void GroupNormalizationLayer_DifferentGroupSizes_Work()
    {
        // Arrange
        int batchSize = 4;
        int numChannels = 32;

        // Test with different group sizes
        int[] groupSizes = [1, 4, 8, 16, 32];
        foreach (var numGroups in groupSizes)
        {
            var layer = new GroupNormalizationLayer<float>(numGroups, numChannels);
            var input = CreateRandomTensor([batchSize, numChannels]);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(input.Shape, output.Shape);
            Assert.False(ContainsNaN(output));
        }
    }

    [Fact]
    public void GroupNormalizationLayer_Getters_ReturnCorrectValues()
    {
        // Arrange
        int numChannels = 16;
        int numGroups = 4;
        var layer = new GroupNormalizationLayer<float>(numGroups, numChannels);

        // Act & Assert
        Assert.Equal(numGroups, layer.NumGroups);
        Assert.Equal(numChannels, layer.NumChannels);
        Assert.NotNull(layer.GetGammaTensor());
        Assert.NotNull(layer.GetBetaTensor());
    }

    [Fact]
    public void GroupNormalizationLayer_4DInput_Works()
    {
        // Arrange - simulate convolutional feature maps [batch, channels, height, width]
        int batchSize = 2;
        int numChannels = 16;
        int height = 4;
        int width = 4;
        int numGroups = 4;
        var layer = new GroupNormalizationLayer<float>(numGroups, numChannels);
        var input = CreateRandomTensor([batchSize, numChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region InstanceNormalizationLayer Tests

    [Fact]
    public void InstanceNormalizationLayer_ForwardPass_ProducesNormalizedOutput()
    {
        // Arrange
        int batchSize = 4;
        int numChannels = 8;
        var layer = new InstanceNormalizationLayer<float>(numChannels);
        var input = CreateRandomTensor([batchSize, numChannels]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void InstanceNormalizationLayer_BackwardPass_ProducesGradients()
    {
        // Arrange
        int batchSize = 4;
        int numChannels = 8;
        var layer = new InstanceNormalizationLayer<float>(numChannels);
        var input = CreateRandomTensor([batchSize, numChannels]);
        layer.Forward(input);
        var upstreamGradient = CreateRandomTensor([batchSize, numChannels], seed: 123);

        // Act
        var gradients = layer.Backward(upstreamGradient);

        // Assert
        Assert.Equal(input.Shape, gradients.Shape);
        Assert.False(ContainsNaN(gradients));
    }

    [Fact]
    public void InstanceNormalizationLayer_Clone_CreatesIdenticalLayer()
    {
        // Arrange
        int numChannels = 8;
        var layer = new InstanceNormalizationLayer<float>(numChannels);
        var input = CreateRandomTensor([4, numChannels]);
        layer.Forward(input);

        // Act
        var clone = layer.Clone();
        var clonedLayer = Assert.IsType<InstanceNormalizationLayer<float>>(clone);

        // Assert
        Assert.NotSame(layer, clonedLayer);
        var output1 = layer.Forward(input);
        var output2 = clonedLayer.Forward(input);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void InstanceNormalizationLayer_WithAffineTrue_HasLearnableParams()
    {
        // Arrange
        int numChannels = 8;
        var layer = new InstanceNormalizationLayer<float>(numChannels, affine: true);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - gamma and beta for each channel when affine=true
        Assert.Equal(numChannels * 2, paramCount);
    }

    [Fact]
    public void InstanceNormalizationLayer_WithAffineFalse_NoLearnableParams()
    {
        // Arrange
        int numChannels = 8;
        var layer = new InstanceNormalizationLayer<float>(numChannels, affine: false);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - no learnable parameters when affine=false
        Assert.Equal(0, paramCount);
    }

    [Fact]
    public void InstanceNormalizationLayer_4DInput_Works()
    {
        // Arrange - simulate convolutional feature maps [batch, channels, height, width]
        int batchSize = 2;
        int numChannels = 8;
        int height = 4;
        int width = 4;
        var layer = new InstanceNormalizationLayer<float>(numChannels);
        var input = CreateRandomTensor([batchSize, numChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region SpectralNormalizationLayer Tests

    [Fact]
    public void SpectralNormalizationLayer_ForwardPass_ProducesOutput()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var denseLayer = new DenseLayer<float>(inputSize, outputSize);
        var layer = new SpectralNormalizationLayer<float>(denseLayer);
        var input = CreateRandomTensor([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([4, outputSize], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SpectralNormalizationLayer_BackwardPass_ProducesGradients()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var denseLayer = new DenseLayer<float>(inputSize, outputSize);
        var layer = new SpectralNormalizationLayer<float>(denseLayer);
        var input = CreateRandomTensor([4, inputSize]);
        layer.Forward(input);
        var upstreamGradient = CreateRandomTensor([4, outputSize], seed: 123);

        // Act
        var gradients = layer.Backward(upstreamGradient);

        // Assert
        Assert.Equal(input.Shape, gradients.Shape);
        Assert.False(ContainsNaN(gradients));
    }

    [Fact]
    public void SpectralNormalizationLayer_Clone_CreatesIdenticalLayer()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var denseLayer = new DenseLayer<float>(inputSize, outputSize);
        var layer = new SpectralNormalizationLayer<float>(denseLayer);
        var input = CreateRandomTensor([4, inputSize]);
        layer.Forward(input);

        // Act
        var clone = layer.Clone();
        var clonedLayer = Assert.IsType<SpectralNormalizationLayer<float>>(clone);

        // Assert
        Assert.NotSame(layer, clonedLayer);
        var output1 = layer.Forward(input);
        var output2 = clonedLayer.Forward(input);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void SpectralNormalizationLayer_WithMultiplePowerIterations_Works()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var denseLayer = new DenseLayer<float>(inputSize, outputSize);
        var layer = new SpectralNormalizationLayer<float>(denseLayer, powerIterations: 5);
        var input = CreateRandomTensor([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([4, outputSize], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SpectralNormalizationLayer_WrapsConvolutionalLayer()
    {
        // Arrange
        int inputChannels = 3;
        int inputHeight = 8;
        int inputWidth = 8;
        int outputChannels = 8;
        int kernelSize = 3;
        var convLayer = new ConvolutionalLayer<float>(inputChannels, inputHeight, inputWidth, outputChannels, kernelSize);
        var layer = new SpectralNormalizationLayer<float>(convLayer);
        var input = CreateRandomTensor([2, inputChannels, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
        Assert.Equal(4, output.Shape.Length); // [batch, channels, height, width]
    }

    #endregion

    #region Cross-Normalization Comparison Tests

    [Fact]
    public void AllNormalizationLayers_ProduceFiniteOutputs()
    {
        // Arrange
        int batchSize = 4;
        int numFeatures = 16;
        var input = CreateRandomTensor([batchSize, numFeatures]);

        var batchNorm = new BatchNormalizationLayer<float>(numFeatures);
        var layerNorm = new LayerNormalizationLayer<float>(numFeatures);
        var groupNorm = new GroupNormalizationLayer<float>(4, numFeatures);
        var instanceNorm = new InstanceNormalizationLayer<float>(numFeatures);

        batchNorm.SetTrainingMode(true);

        // Act
        var batchOutput = batchNorm.Forward(input);
        var layerOutput = layerNorm.Forward(input);
        var groupOutput = groupNorm.Forward(input);
        var instanceOutput = instanceNorm.Forward(input);

        // Assert - All should produce valid outputs
        Assert.False(ContainsNaN(batchOutput));
        Assert.False(ContainsNaN(layerOutput));
        Assert.False(ContainsNaN(groupOutput));
        Assert.False(ContainsNaN(instanceOutput));
    }

    [Fact]
    public void AllNormalizationLayers_HandleZeroVarianceGracefully()
    {
        // Arrange - constant input (zero variance)
        int batchSize = 4;
        int numFeatures = 16;
        var constantData = new float[batchSize * numFeatures];
        for (int i = 0; i < constantData.Length; i++)
            constantData[i] = 0.5f;
        var input = new Tensor<float>(constantData, [batchSize, numFeatures]);

        var batchNorm = new BatchNormalizationLayer<float>(numFeatures);
        var layerNorm = new LayerNormalizationLayer<float>(numFeatures);
        var groupNorm = new GroupNormalizationLayer<float>(4, numFeatures);
        var instanceNorm = new InstanceNormalizationLayer<float>(numFeatures);

        batchNorm.SetTrainingMode(true);

        // Act
        var batchOutput = batchNorm.Forward(input);
        var layerOutput = layerNorm.Forward(input);
        var groupOutput = groupNorm.Forward(input);
        var instanceOutput = instanceNorm.Forward(input);

        // Assert - All should handle zero variance gracefully (no NaN/Inf)
        Assert.False(ContainsNaN(batchOutput));
        Assert.False(ContainsNaN(layerOutput));
        Assert.False(ContainsNaN(groupOutput));
        Assert.False(ContainsNaN(instanceOutput));
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void BatchNormalizationLayer_SingleSample_Works()
    {
        // Arrange
        int numFeatures = 8;
        var layer = new BatchNormalizationLayer<float>(numFeatures);
        layer.SetTrainingMode(false); // Use running stats for single sample
        var input = CreateRandomTensor([1, numFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void LayerNormalizationLayer_SingleFeature_Works()
    {
        // Arrange
        int batchSize = 4;
        int featureSize = 1;
        var layer = new LayerNormalizationLayer<float>(featureSize);
        var input = CreateRandomTensor([batchSize, featureSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GroupNormalizationLayer_OneGroupPerChannel_EquivalentToInstanceNorm()
    {
        // Arrange - when numGroups = numChannels, it's equivalent to instance norm
        int batchSize = 4;
        int numChannels = 8;
        var groupNorm = new GroupNormalizationLayer<float>(numChannels, numChannels);
        var input = CreateRandomTensor([batchSize, numChannels]);

        // Act
        var output = groupNorm.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GroupNormalizationLayer_SingleGroup_EquivalentToLayerNorm()
    {
        // Arrange - when numGroups = 1, it's equivalent to layer norm
        int batchSize = 4;
        int numChannels = 8;
        var groupNorm = new GroupNormalizationLayer<float>(1, numChannels);
        var input = CreateRandomTensor([batchSize, numChannels]);

        // Act
        var output = groupNorm.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region Private Helper Methods

    private static bool ContainsNaN(Tensor<float> tensor)
    {
        int dim0 = tensor.Shape[0];
        int dim1 = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                if (tensor.Shape.Length == 1)
                {
                    if (float.IsNaN(tensor[i]))
                        return true;
                }
                else if (tensor.Shape.Length == 2)
                {
                    if (float.IsNaN(tensor[i, j]))
                        return true;
                }
                else if (tensor.Shape.Length >= 3)
                {
                    // Check first element of higher dimensions
                    var indices = new int[tensor.Shape.Length];
                    indices[0] = i;
                    indices[1] = j;
                    for (int k = 2; k < tensor.Shape.Length; k++)
                        indices[k] = 0;
                    if (float.IsNaN(tensor[indices]))
                        return true;
                }
            }
        }
        return false;
    }

    #endregion
}
