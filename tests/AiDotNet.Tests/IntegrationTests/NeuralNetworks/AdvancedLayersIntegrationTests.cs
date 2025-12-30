using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for advanced neural network layers that were previously untested.
/// These include transformer layers, utility layers, and specialized architectural components.
/// </summary>
public class AdvancedLayersIntegrationTests
{
    #region TransformerEncoderLayer Tests

    [Fact]
    public void TransformerEncoderLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 256;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);

        int batchSize = 2;
        var input = Tensor<float>.CreateRandom([batchSize, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(embeddingSize, output.Shape[1]);
    }

    [Fact]
    public void TransformerEncoderLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 8;
        int feedForwardDim = 256;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);

        int batchSize = 2;
        int seqLen = 10;
        var input = Tensor<float>.CreateRandom([batchSize, seqLen, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(seqLen, output.Shape[1]);
        Assert.Equal(embeddingSize, output.Shape[2]);
    }

    [Fact]
    public void TransformerEncoderLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int embeddingSize = 32;
        int numHeads = 4;
        int feedForwardDim = 128;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);

        var input = Tensor<float>.CreateRandom([2, embeddingSize]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void TransformerEncoderLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 256;
        var original = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<TransformerEncoderLayer<float>>(clone);
    }

    [Fact]
    public void TransformerEncoderLayer_ParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 256;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region TransformerDecoderLayer Tests

    [Fact]
    public void TransformerDecoderLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 256;
        int sequenceLength = 10;
        var layer = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, sequenceLength, (IActivationFunction<float>?)null);

        // Decoder input and encoder output (both needed for cross-attention)
        var input = Tensor<float>.CreateRandom([2, embeddingSize]);
        var encoderOutput = Tensor<float>.CreateRandom([2, embeddingSize]);

        // Act
        var output = layer.Forward(input, encoderOutput);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(embeddingSize, output.Shape[^1]);
    }

    [Fact]
    public void TransformerDecoderLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 8;
        int feedForwardDim = 256;
        int sequenceLength = 10;
        var layer = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, sequenceLength, (IActivationFunction<float>?)null);

        int batchSize = 2;
        var input = Tensor<float>.CreateRandom([batchSize, sequenceLength, embeddingSize]);
        var encoderOutput = Tensor<float>.CreateRandom([batchSize, sequenceLength, embeddingSize]);

        // Act
        var output = layer.Forward(input, encoderOutput);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(embeddingSize, output.Shape[^1]);
    }

    [Fact]
    public void TransformerDecoderLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 256;
        int sequenceLength = 10;
        var original = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, sequenceLength, (IActivationFunction<float>?)null);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<TransformerDecoderLayer<float>>(clone);
    }

    #endregion

    #region FeedForwardLayer Tests

    [Fact]
    public void FeedForwardLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new FeedForwardLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

        int batchSize = 4;
        var input = Tensor<float>.CreateRandom([batchSize, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(outputSize, output.Shape[1]);
    }

    [Fact]
    public void FeedForwardLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var layer = new FeedForwardLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, inputSize]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void FeedForwardLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var original = new FeedForwardLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<FeedForwardLayer<float>>(clone);
    }

    [Fact]
    public void FeedForwardLayer_ParameterCount_ReturnsCorrectValue()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new FeedForwardLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        // weights (64 * 32) + biases (32) = 2080
        int expectedParams = inputSize * outputSize + outputSize;
        Assert.Equal(expectedParams, paramCount);
    }

    [Fact]
    public void FeedForwardLayer_WithReLU_ProducesNonNegativeOutput()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var layer = new FeedForwardLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());

        // Use negative inputs to verify ReLU
        var input = Tensor<float>.CreateDefault([2, inputSize], -1.0f);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output.Data[i] >= 0, $"Output at index {i} should be non-negative but was {output.Data[i]}");
        }
    }

    #endregion

    #region ResidualLayer Tests

    [Fact]
    public void ResidualLayer_ForwardPass_WithoutInnerLayer_PreservesInput()
    {
        // Arrange
        int[] inputShape = [32];
        var layer = new ResidualLayer<float>(inputShape, null, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void ResidualLayer_ForwardPass_WithInnerLayer_ProducesValidOutput()
    {
        // Arrange
        int size = 32;
        int[] inputShape = [size];
        var innerLayer = new DenseLayer<float>(size, size);
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, size]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void ResidualLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int size = 16;
        int[] inputShape = [size];
        var innerLayer = new DenseLayer<float>(size, size);
        var layer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, size]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void ResidualLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int size = 32;
        int[] inputShape = [size];
        var innerLayer = new DenseLayer<float>(size, size);
        var original = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<ResidualLayer<float>>(clone);
    }

    #endregion

    #region DeconvolutionalLayer Tests

    [Fact]
    public void DeconvolutionalLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int[] inputShape = [1, 4, 8, 8]; // batch, channels, height, width
        int outputDepth = 2;
        int kernelSize = 3;
        int stride = 2;
        int padding = 1;
        var layer = new DeconvolutionalLayer<float>(inputShape, outputDepth, kernelSize, stride, padding, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([1, 4, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]); // batch size
        Assert.Equal(outputDepth, output.Shape[1]); // output channels
        Assert.True(output.Shape[2] > inputShape[2], "Output height should be larger than input height");
        Assert.True(output.Shape[3] > inputShape[3], "Output width should be larger than input width");
    }

    [Fact]
    public void DeconvolutionalLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [1, 4, 8, 8];
        int outputDepth = 2;
        int kernelSize = 3;
        int stride = 2;
        int padding = 1;
        var layer = new DeconvolutionalLayer<float>(inputShape, outputDepth, kernelSize, stride, padding, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([1, 4, 8, 8]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(4, gradient.Shape.Length);
    }

    [Fact]
    public void DeconvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [1, 4, 8, 8];
        var original = new DeconvolutionalLayer<float>(inputShape, 2, 3, 2, 1, (IActivationFunction<float>?)null);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<DeconvolutionalLayer<float>>(clone);
    }

    #endregion

    #region UpsamplingLayer Tests

    [Fact]
    public void UpsamplingLayer_ForwardPass_IncreasesSize()
    {
        // Arrange
        int[] inputShape = [4, 8, 8]; // channels, height, width
        int scaleFactor = 2;
        var layer = new UpsamplingLayer<float>(inputShape, scaleFactor);

        var input = Tensor<float>.CreateRandom([1, 4, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]); // batch size preserved
        Assert.Equal(4, output.Shape[1]); // channels preserved
        Assert.Equal(16, output.Shape[2]); // height doubled
        Assert.Equal(16, output.Shape[3]); // width doubled
    }

    [Fact]
    public void UpsamplingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [4, 8, 8];
        int scaleFactor = 2;
        var layer = new UpsamplingLayer<float>(inputShape, scaleFactor);

        var input = Tensor<float>.CreateRandom([1, 4, 8, 8]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(4, gradient.Shape.Length);
    }

    [Fact]
    public void UpsamplingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [4, 8, 8];
        var original = new UpsamplingLayer<float>(inputShape, 2);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<UpsamplingLayer<float>>(clone);
    }

    [Fact]
    public void UpsamplingLayer_ScaleFactor4_ProducesCorrectSize()
    {
        // Arrange
        int[] inputShape = [2, 4, 4];
        int scaleFactor = 4;
        var layer = new UpsamplingLayer<float>(inputShape, scaleFactor);

        var input = Tensor<float>.CreateRandom([1, 2, 4, 4]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(16, output.Shape[2]); // height * 4
        Assert.Equal(16, output.Shape[3]); // width * 4
    }

    #endregion

    #region AddLayer Tests

    [Fact]
    public void AddLayer_ForwardPass_AddsTwoInputs()
    {
        // Arrange
        int[] shape = [4, 8];
        int[][] inputShapes = [shape, shape];
        var layer = new AddLayer<float>(inputShapes, (IActivationFunction<float>?)null);

        var input1 = Tensor<float>.CreateDefault([2, 4, 8], 1.0f);
        var input2 = Tensor<float>.CreateDefault([2, 4, 8], 2.0f);

        // Act - AddLayer requires multiple inputs via params overload
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.NotNull(output);
        // Output should have same shape as single input
        Assert.Equal(input1.Shape, output.Shape);
        // Values should be sum of inputs (1.0 + 2.0 = 3.0)
        Assert.Equal(3.0f, output[0, 0, 0], 1e-5f);
    }

    [Fact]
    public void AddLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] shape = [4, 8];
        int[][] inputShapes = [shape, shape];
        var original = new AddLayer<float>(inputShapes, (IActivationFunction<float>?)null);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<AddLayer<float>>(clone);
    }

    #endregion

    #region ConcatenateLayer Tests

    [Fact]
    public void ConcatenateLayer_ForwardPass_ConcatenatesAlongAxis()
    {
        // Arrange
        int[] shape1 = [4, 8];
        int[] shape2 = [4, 8];
        int[][] inputShapes = [shape1, shape2];
        int axis = 1; // Concatenate along axis 1 (second dimension)
        var layer = new ConcatenateLayer<float>(inputShapes, axis, (IActivationFunction<float>?)null);

        var input1 = Tensor<float>.CreateRandom([2, 4, 8]);
        var input2 = Tensor<float>.CreateRandom([2, 4, 8]);

        // Act - ConcatenateLayer requires multiple inputs via params overload
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.NotNull(output);
        // Concatenated along axis 1: [2, 4, 8] + [2, 4, 8] -> [2, 8, 8]
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(8, output.Shape[1]); // axis 1 doubled (4 + 4)
        Assert.Equal(8, output.Shape[2]); // features preserved
    }

    [Fact]
    public void ConcatenateLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] shape = [4, 8];
        int[][] inputShapes = [shape, shape];
        var original = new ConcatenateLayer<float>(inputShapes, 0, (IActivationFunction<float>?)null);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<ConcatenateLayer<float>>(clone);
    }

    #endregion

    #region SqueezeAndExcitationLayer Tests

    [Fact]
    public void SqueezeAndExcitationLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int channels = 64;
        int reductionRatio = 4;
        var layer = new SqueezeAndExcitationLayer<float>(channels, reductionRatio, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);

        // SE layer expects BHWC format: [batch, height, width, channels]
        var input = Tensor<float>.CreateRandom([2, 8, 8, channels]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape.Length, output.Shape.Length);
        Assert.Equal(input.Shape[0], output.Shape[0]); // batch preserved
        Assert.Equal(input.Shape[1], output.Shape[1]); // height preserved
        Assert.Equal(input.Shape[2], output.Shape[2]); // width preserved
        Assert.Equal(channels, output.Shape[3]); // channels preserved
    }

    [Fact]
    public void SqueezeAndExcitationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int channels = 64;
        int reductionRatio = 4;
        var original = new SqueezeAndExcitationLayer<float>(channels, reductionRatio, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.IsType<SqueezeAndExcitationLayer<float>>(clone);
    }

    [Fact]
    public void SqueezeAndExcitationLayer_ParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        int channels = 64;
        int reductionRatio = 4;
        var layer = new SqueezeAndExcitationLayer<float>(channels, reductionRatio, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region Cross-Layer Integration Tests

    [Fact]
    public void TransformerEncoderDecoderStack_ForwardPass_Works()
    {
        // Arrange
        int embeddingSize = 64;
        int numHeads = 4;
        int feedForwardDim = 256;
        int seqLen = 10;

        var encoder = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);
        var decoder = new TransformerDecoderLayer<float>(embeddingSize, numHeads, feedForwardDim, seqLen, (IActivationFunction<float>?)null);

        var sourceInput = Tensor<float>.CreateRandom([2, seqLen, embeddingSize]);
        var targetInput = Tensor<float>.CreateRandom([2, seqLen, embeddingSize]);

        // Act
        var encoderOutput = encoder.Forward(sourceInput);
        var decoderOutput = decoder.Forward(targetInput, encoderOutput);

        // Assert
        Assert.NotNull(decoderOutput);
        Assert.Equal(embeddingSize, decoderOutput.Shape[^1]);
    }

    [Fact]
    public void ResidualWithFeedForward_ForwardPass_Works()
    {
        // Arrange
        int size = 64;
        int[] inputShape = [size];
        var innerLayer = new FeedForwardLayer<float>(size, size, (IActivationFunction<float>?)null);
        var residualLayer = new ResidualLayer<float>(inputShape, innerLayer, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, size]);

        // Act
        var output = residualLayer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void ConvDeconvPipeline_ForwardPass_Works()
    {
        // Arrange
        int[] convInputShape = [1, 4, 16, 16]; // batch, channels, height, width
        var conv = new ConvolutionalLayer<float>(4, 16, 16, 8, 3, 2, 1); // Downsample

        var convOutput = conv.Forward(Tensor<float>.CreateRandom(convInputShape));
        int[] deconvInputShape = [convOutput.Shape[0], convOutput.Shape[1], convOutput.Shape[2], convOutput.Shape[3]];

        var deconv = new DeconvolutionalLayer<float>(deconvInputShape, 4, 3, 2, 1, (IActivationFunction<float>?)null); // Upsample

        var input = Tensor<float>.CreateRandom(convInputShape);

        // Act
        var downsampled = conv.Forward(input);
        var upsampled = deconv.Forward(downsampled);

        // Assert
        Assert.NotNull(upsampled);
        Assert.True(upsampled.Shape[2] > downsampled.Shape[2], "Deconv should increase spatial size");
    }

    [Fact]
    public void ResidualChain_MultipleBlocks_Works()
    {
        // Arrange
        int size = 32;
        int[] inputShape = [size];

        var block1 = new ResidualLayer<float>(inputShape, new DenseLayer<float>(size, size), (IActivationFunction<float>?)null);
        var block2 = new ResidualLayer<float>(inputShape, new DenseLayer<float>(size, size), (IActivationFunction<float>?)null);
        var block3 = new ResidualLayer<float>(inputShape, new DenseLayer<float>(size, size), (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, size]);

        // Act
        var out1 = block1.Forward(input);
        var out2 = block2.Forward(out1);
        var output = block3.Forward(out2);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void TransformerEncoderLayer_SmallEmbedding_Works()
    {
        // Arrange - minimum valid configuration
        int embeddingSize = 8;
        int numHeads = 2;
        int feedForwardDim = 16;
        var layer = new TransformerEncoderLayer<float>(embeddingSize, numHeads, feedForwardDim);

        var input = Tensor<float>.CreateRandom([1, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(embeddingSize, output.Shape[^1]);
    }

    [Fact]
    public void FeedForwardLayer_LargeInputOutput_Works()
    {
        // Arrange
        int inputSize = 1024;
        int outputSize = 2048;
        var layer = new FeedForwardLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([1, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(outputSize, output.Shape[1]);
    }

    [Fact]
    public void UpsamplingLayer_SmallInput_Works()
    {
        // Arrange
        int[] inputShape = [2, 2, 2]; // Very small spatial dimensions
        var layer = new UpsamplingLayer<float>(inputShape, 2);

        var input = Tensor<float>.CreateRandom([1, 2, 2, 2]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[2]);
        Assert.Equal(4, output.Shape[3]);
    }

    #endregion
}
