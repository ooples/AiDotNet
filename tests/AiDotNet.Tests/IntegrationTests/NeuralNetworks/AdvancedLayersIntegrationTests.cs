using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.RadialBasisFunctions;
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
            Assert.True(output[i] >= 0, $"Output at index {i} should be non-negative but was {output[i]}");
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

    #region ConvLSTMLayer Tests

    [Fact]
    public void ConvLSTMLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - ConvLSTM for spatiotemporal data
        int[] inputShape = [4, 8, 8, 3]; // [timeSteps, height, width, channels]
        int kernelSize = 3;
        int filters = 16;
        var layer = new ConvLSTMLayer<float>(inputShape, kernelSize, filters, 1, 1, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, 4, 8, 8, 3]); // [batch, time, H, W, C]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(5, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void ConvLSTMLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [2, 4, 4, 1];
        var original = new ConvLSTMLayer<float>(inputShape, 3, 8, 1, 1, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([1, 2, 4, 4, 1]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region GatedLinearUnitLayer Tests

    [Fact]
    public void GatedLinearUnitLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputDim = 64;
        int outputDim = 32;
        var layer = new GatedLinearUnitLayer<float>(inputDim, outputDim, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([4, inputDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(4, output.Shape[0]); // batch preserved
        Assert.Equal(outputDim, output.Shape[1]);
    }

    [Fact]
    public void GatedLinearUnitLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new GatedLinearUnitLayer<float>(32, 16, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void GatedLinearUnitLayer_ParameterCount_IsPositive()
    {
        // Arrange
        var layer = new GatedLinearUnitLayer<float>(64, 32, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - GLU has linearWeights (64*32) + gateWeights (64*32) + linearBias (32) + gateBias (32)
        // = 2048 + 2048 + 32 + 32 = 4160 parameters
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
        Assert.Equal(4160, paramCount);
    }

    #endregion

    #region HighwayLayer Tests

    [Fact]
    public void HighwayLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - Highway layers preserve dimensions
        int inputDim = 64;
        var layer = new HighwayLayer<float>(inputDim, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([4, inputDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void HighwayLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new HighwayLayer<float>(32, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void HighwayLayer_ParameterCount_IsPositive()
    {
        // Arrange
        var layer = new HighwayLayer<float>(64, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - Highway has transformWeights (64*64) + transformBias (64) + gateWeights (64*64) + gateBias (64)
        // = 4096 + 64 + 4096 + 64 = 8320 parameters
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
        Assert.Equal(8320, paramCount);
    }

    #endregion

    #region MaxPool3DLayer Tests

    [Fact]
    public void MaxPool3DLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - 3D pooling for volumetric data [channels, depth, height, width]
        int[] inputShape = [3, 8, 8, 8]; // C, D, H, W
        int poolSize = 2;
        var layer = new MaxPool3DLayer<float>(inputShape, poolSize);

        var input = Tensor<float>.CreateRandom([2, 3, 8, 8, 8]); // [batch, C, D, H, W]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(5, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(3, output.Shape[1]); // channels preserved
        Assert.Equal(4, output.Shape[2]); // D halved
        Assert.Equal(4, output.Shape[3]); // H halved
        Assert.Equal(4, output.Shape[4]); // W halved
    }

    [Fact]
    public void MaxPool3DLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange - input shape must be [channels, depth, height, width]
        int[] inputShape = [2, 4, 4, 4];
        var original = new MaxPool3DLayer<float>(inputShape, 2);
        var input = Tensor<float>.CreateRandom([1, 2, 4, 4, 4]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region MultiplyLayer Tests

    [Fact]
    public void MultiplyLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - Element-wise multiplication of multiple inputs
        int[] shape = [8, 16];
        int[][] inputShapes = [shape, shape];
        var layer = new MultiplyLayer<float>(inputShapes, (IActivationFunction<float>?)null);

        var input1 = Tensor<float>.CreateRandom([2, 8, 16]);
        var input2 = Tensor<float>.CreateRandom([2, 8, 16]);

        // Act
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input1.Shape, output.Shape);
    }

    [Fact]
    public void MultiplyLayer_ElementWiseMultiplication_IsCorrect()
    {
        // Arrange
        int[] shape = [2, 2];
        int[][] inputShapes = [shape, shape];
        var layer = new MultiplyLayer<float>(inputShapes, (IActivationFunction<float>?)null);

        var input1 = new Tensor<float>([1, 2, 2]);
        var input2 = new Tensor<float>([1, 2, 2]);
        input1[0] = 2; input1[1] = 3; input1[2] = 4; input1[3] = 5;
        input2[0] = 1; input2[1] = 2; input2[2] = 3; input2[3] = 4;

        // Act
        var output = layer.Forward(input1, input2);

        // Assert - element-wise multiplication
        Assert.Equal(2f, output[0]); // 2 * 1
        Assert.Equal(6f, output[1]); // 3 * 2
        Assert.Equal(12f, output[2]); // 4 * 3
        Assert.Equal(20f, output[3]); // 5 * 4
    }

    [Fact]
    public void MultiplyLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] shape = [4, 8];
        int[][] inputShapes = [shape, shape];
        var original = new MultiplyLayer<float>(inputShapes, (IActivationFunction<float>?)null);
        var input1 = Tensor<float>.CreateRandom([1, 4, 8]);
        var input2 = Tensor<float>.CreateRandom([1, 4, 8]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input1, input2);
        var cloneOutput = clone.Forward(input1, input2);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region MaskingLayer Tests

    [Fact]
    public void MaskingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int[] inputShape = [10, 32]; // sequence, features
        var layer = new MaskingLayer<float>(inputShape, maskValue: 0);

        var input = Tensor<float>.CreateRandom([2, 10, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void MaskingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [5, 16];
        var original = new MaskingLayer<float>(inputShape);
        var input = Tensor<float>.CreateRandom([1, 5, 16]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SplitLayer Tests

    [Fact]
    public void SplitLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - Split tensor along last dimension
        int[] inputShape = [32];
        int numSplits = 4;
        var layer = new SplitLayer<float>(inputShape, numSplits);

        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(new[] { 2, numSplits, 8 }, output.Shape);

        var reconstructed = output.Reshape([2, 32]);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], reconstructed[i]);
        }
    }

    [Fact]
    public void SplitLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [16];
        var original = new SplitLayer<float>(inputShape, 2);
        var input = Tensor<float>.CreateRandom([1, 16]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region ReshapeLayer Tests

    [Fact]
    public void ReshapeLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - Reshape from [8, 4] to [32]
        int[] inputShape = [8, 4];
        int[] outputShape = [32];
        var layer = new ReshapeLayer<float>(inputShape, outputShape);

        var input = Tensor<float>.CreateRandom([2, 8, 4]); // batch + inputShape

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(32, output.Shape[1]); // reshaped
    }

    [Fact]
    public void ReshapeLayer_FlattenToExpand_Works()
    {
        // Arrange - Reshape from [16] to [4, 4]
        int[] inputShape = [16];
        int[] outputShape = [4, 4];
        var layer = new ReshapeLayer<float>(inputShape, outputShape);

        var input = Tensor<float>.CreateRandom([2, 16]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(3, output.Shape.Length); // batch + 2D output
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(4, output.Shape[1]);
        Assert.Equal(4, output.Shape[2]);
    }

    [Fact]
    public void ReshapeLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [8, 4];
        int[] outputShape = [32];
        var original = new ReshapeLayer<float>(inputShape, outputShape);
        var input = Tensor<float>.CreateRandom([1, 8, 4]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region Conv3DLayer Tests

    [Fact]
    public void Conv3DLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - 3D convolution for volumetric data
        int inputChannels = 1;
        int outputChannels = 8;
        int kernelSize = 3;
        int inputDepth = 8, inputHeight = 8, inputWidth = 8;
        var layer = new Conv3DLayer<float>(inputChannels, outputChannels, kernelSize,
            inputDepth, inputHeight, inputWidth, 1, 1, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, inputChannels, inputDepth, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(5, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(outputChannels, output.Shape[1]); // channels
    }

    [Fact]
    public void Conv3DLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputChannels = 1;
        int outputChannels = 4;
        int kernelSize = 3;
        var original = new Conv3DLayer<float>(inputChannels, outputChannels, kernelSize,
            4, 4, 4, 1, 1, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([1, inputChannels, 4, 4, 4]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region GRULayer Tests

    [Fact]
    public void GRULayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - GRU for sequence data
        int inputSize = 16;
        int hiddenSize = 32;
        bool returnSequences = false;
        var layer = new GRULayer<float>(inputSize, hiddenSize, returnSequences, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, 5, inputSize]); // [batch, sequence, features]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // When returnSequences=false, output is [batch, hiddenSize]
        Assert.Equal(2, output.Shape[0]); // batch
    }

    [Fact]
    public void GRULayer_ReturnSequences_ProducesSequenceOutput()
    {
        // Arrange - GRU returning full sequence
        int inputSize = 16;
        int hiddenSize = 32;
        bool returnSequences = true;
        var layer = new GRULayer<float>(inputSize, hiddenSize, returnSequences, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, 5, inputSize]); // [batch, sequence, features]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // When returnSequences=true, output is [batch, sequence, hiddenSize]
        Assert.Equal(2, output.Shape[0]); // batch
    }

    [Fact]
    public void GRULayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 8;
        int hiddenSize = 16;
        var original = new GRULayer<float>(inputSize, hiddenSize, false, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([1, 3, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SelfAttentionLayer Tests

    [Fact]
    public void SelfAttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int sequenceLength = 10;
        int embeddingSize = 64;
        int numHeads = 4;
        var layer = new SelfAttentionLayer<float>(sequenceLength, embeddingSize, numHeads, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, sequenceLength, embeddingSize]); // [batch, sequence, embedding]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void SelfAttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange
        int sequenceLength = 4;
        int embeddingSize = 32;
        int numHeads = 4;
        var layer = new SelfAttentionLayer<float>(sequenceLength, embeddingSize, numHeads, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([4, embeddingSize]); // [batch, embedding]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
    }

    [Fact]
    public void SelfAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int sequenceLength = 5;
        int embeddingSize = 32;
        int numHeads = 2;
        var layer = new SelfAttentionLayer<float>(sequenceLength, embeddingSize, numHeads, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, sequenceLength, embeddingSize]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void SelfAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int sequenceLength = 4;
        int embeddingSize = 32;
        int numHeads = 2;
        var original = new SelfAttentionLayer<float>(sequenceLength, embeddingSize, numHeads, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([1, sequenceLength, embeddingSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void SelfAttentionLayer_ParameterCount_IsPositive()
    {
        // Arrange
        int sequenceLength = 10;
        int embeddingSize = 64;
        int numHeads = 4;
        var layer = new SelfAttentionLayer<float>(sequenceLength, embeddingSize, numHeads, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region MultiHeadAttentionLayer Tests

    [Fact]
    public void MultiHeadAttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int sequenceLength = 10;
        int embeddingSize = 64;
        int numHeads = 8;
        var layer = new MultiHeadAttentionLayer<float>(sequenceLength, embeddingSize, numHeads);

        var input = Tensor<float>.CreateRandom([2, sequenceLength, embeddingSize]); // [batch, sequence, embedding]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void MultiHeadAttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange
        int sequenceLength = 4;
        int embeddingSize = 48;
        int numHeads = 6;
        var layer = new MultiHeadAttentionLayer<float>(sequenceLength, embeddingSize, numHeads);

        var input = Tensor<float>.CreateRandom([4, embeddingSize]); // [batch, embedding]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void MultiHeadAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int sequenceLength = 5;
        int embeddingSize = 32;
        int numHeads = 4;
        var layer = new MultiHeadAttentionLayer<float>(sequenceLength, embeddingSize, numHeads);

        var input = Tensor<float>.CreateRandom([2, sequenceLength, embeddingSize]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void MultiHeadAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int sequenceLength = 4;
        int embeddingSize = 32;
        int numHeads = 4;
        var original = new MultiHeadAttentionLayer<float>(sequenceLength, embeddingSize, numHeads);
        var input = Tensor<float>.CreateRandom([1, sequenceLength, embeddingSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void MultiHeadAttentionLayer_ParameterCount_IsPositive()
    {
        // Arrange
        int sequenceLength = 10;
        int embeddingSize = 64;
        int numHeads = 8;
        var layer = new MultiHeadAttentionLayer<float>(sequenceLength, embeddingSize, numHeads);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region DenseLayer Tests

    [Fact]
    public void DenseLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(4, output.Shape[0]);
        Assert.Equal(outputSize, output.Shape[1]);
    }

    [Fact]
    public void DenseLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var layer = new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

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
    public void DenseLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var original = new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void DenseLayer_ParameterCount_IsCorrect()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - weights (64*32) + biases (32) = 2080
        int expected = inputSize * outputSize + outputSize;
        Assert.Equal(expected, paramCount);
    }

    [Fact]
    public void DenseLayer_WithReLU_ProducesNonNegativeOutput()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var layer = new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());

        var input = Tensor<float>.CreateDefault([2, inputSize], -1.0f);

        // Act
        var output = layer.Forward(input);

        // Assert
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] >= 0, $"Output at index {i} should be non-negative");
        }
    }

    #endregion

    #region LSTMLayer Tests

    [Fact]
    public void LSTMLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 16;
        int hiddenSize = 32;
        int sequenceLength = 5;
        int[] inputShape = [sequenceLength, inputSize];
        var layer = new LSTMLayer<float>(inputSize, hiddenSize, inputShape, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, sequenceLength, inputSize]); // [batch, sequence, features]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void LSTMLayer_ForwardPass_DifferentSequenceLengths_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 16;
        int hiddenSize = 32;
        int sequenceLength = 10;
        int[] inputShape = [sequenceLength, inputSize];
        var layer = new LSTMLayer<float>(inputSize, hiddenSize, inputShape, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, sequenceLength, inputSize]); // [batch, sequence, features]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch
    }

    [Fact]
    public void LSTMLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 8;
        int hiddenSize = 16;
        int sequenceLength = 3;
        int[] inputShape = [sequenceLength, inputSize];
        var layer = new LSTMLayer<float>(inputSize, hiddenSize, inputShape, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, sequenceLength, inputSize]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void LSTMLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 8;
        int hiddenSize = 16;
        int sequenceLength = 3;
        int[] inputShape = [sequenceLength, inputSize];
        var original = new LSTMLayer<float>(inputSize, hiddenSize, inputShape, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([1, sequenceLength, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void LSTMLayer_ParameterCount_IsPositive()
    {
        // Arrange
        int inputSize = 16;
        int hiddenSize = 32;
        int sequenceLength = 5;
        int[] inputShape = [sequenceLength, inputSize];
        var layer = new LSTMLayer<float>(inputSize, hiddenSize, inputShape, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - LSTM has 4 gates, each with weights and biases
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region DropoutLayer Tests

    [Fact]
    public void DropoutLayer_ForwardPass_Training_AppliesDropout()
    {
        // Arrange
        double dropoutRate = 0.5;
        var layer = new DropoutLayer<float>(dropoutRate);
        layer.SetTrainingMode(true);

        var input = Tensor<float>.CreateDefault([4, 64], 1.0f);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);

        // Some values should be zeroed out in training mode
        int zeroCount = output.Data.ToArray().Count(v => v == 0f);
        // With 50% dropout, we expect roughly half to be zero
        Assert.True(zeroCount > 0, "Dropout should zero some values during training");
    }

    [Fact]
    public void DropoutLayer_ForwardPass_Inference_PreservesInput()
    {
        // Arrange
        double dropoutRate = 0.5;
        var layer = new DropoutLayer<float>(dropoutRate);
        layer.SetTrainingMode(false);

        var input = Tensor<float>.CreateDefault([4, 64], 1.0f);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);

        // In inference mode, output should match input (no dropout applied)
        for (int i = 0; i < output.Length; i++)
        {
            Assert.Equal(1.0f, output[i]);
        }
    }

    [Fact]
    public void DropoutLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new DropoutLayer<float>(0.3);
        original.SetTrainingMode(false);
        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region FlattenLayer Tests

    [Fact]
    public void FlattenLayer_ForwardPass_3DInput_FlattensCorrectly()
    {
        // Arrange
        int[] inputShape = [8, 8, 3]; // H, W, C
        var layer = new FlattenLayer<float>(inputShape);

        var input = Tensor<float>.CreateRandom([2, 8, 8, 3]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(8 * 8 * 3, output.Shape[1]); // flattened
    }

    [Fact]
    public void FlattenLayer_ForwardPass_4DInput_FlattensCorrectly()
    {
        // Arrange
        int[] inputShape = [4, 4, 4, 8]; // D, H, W, C
        var layer = new FlattenLayer<float>(inputShape);

        var input = Tensor<float>.CreateRandom([2, 4, 4, 4, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(4 * 4 * 4 * 8, output.Shape[1]); // flattened
    }

    [Fact]
    public void FlattenLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [4, 4, 3];
        var layer = new FlattenLayer<float>(inputShape);

        var input = Tensor<float>.CreateRandom([2, 4, 4, 3]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void FlattenLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [4, 4, 2];
        var original = new FlattenLayer<float>(inputShape);
        var input = Tensor<float>.CreateRandom([1, 4, 4, 2]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region ActivationLayer Tests

    [Fact]
    public void ActivationLayer_ReLU_ProducesNonNegativeOutput()
    {
        // Arrange
        int[] inputShape = [64];
        var layer = new ActivationLayer<float>(inputShape, (IActivationFunction<float>)new ReLUActivation<float>());

        var input = Tensor<float>.CreateRandom([4, 64]);
        // Make some values negative
        for (int i = 0; i < input.Length / 2; i++)
        {
            input[i] = -Math.Abs(input[i]);
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] >= 0, $"ReLU output at {i} should be non-negative");
        }
    }

    [Fact]
    public void ActivationLayer_Sigmoid_ProducesOutputInRange()
    {
        // Arrange
        int[] inputShape = [32];
        var layer = new ActivationLayer<float>(inputShape, (IActivationFunction<float>)new SigmoidActivation<float>());

        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] >= 0 && output[i] <= 1,
                $"Sigmoid output at {i} should be in [0, 1], got {output[i]}");
        }
    }

    [Fact]
    public void ActivationLayer_Tanh_ProducesOutputInRange()
    {
        // Arrange
        int[] inputShape = [32];
        var layer = new ActivationLayer<float>(inputShape, (IActivationFunction<float>)new TanhActivation<float>());

        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] >= -1 && output[i] <= 1,
                $"Tanh output at {i} should be in [-1, 1], got {output[i]}");
        }
    }

    [Fact]
    public void ActivationLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [16];
        var layer = new ActivationLayer<float>(inputShape, (IActivationFunction<float>)new ReLUActivation<float>());

        var input = Tensor<float>.CreateRandom([2, 16]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void ActivationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [16];
        var original = new ActivationLayer<float>(inputShape, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([2, 16]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region InputLayer Tests

    [Fact]
    public void InputLayer_ForwardPass_PreservesInput()
    {
        // Arrange
        int inputSize = 64;
        var layer = new InputLayer<float>(inputSize);

        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
        // InputLayer should pass through without modification
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], output[i]);
        }
    }

    [Fact]
    public void InputLayer_BackwardPass_PreservesGradient()
    {
        // Arrange
        int inputSize = 16;
        var layer = new InputLayer<float>(inputSize);

        var input = Tensor<float>.CreateRandom([4, inputSize]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(upstreamGradient.Shape, gradient.Shape);
    }

    [Fact]
    public void InputLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 784;
        var original = new InputLayer<float>(inputSize);
        var input = Tensor<float>.CreateRandom([1, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region PaddingLayer Tests

    [Fact]
    public void PaddingLayer_ForwardPass_AddsPadding()
    {
        // Arrange
        // Input shape includes batch dimension; padding must match full input dimensions
        int[] inputShape = [2, 8, 8, 3];
        int[] padding = [0, 2, 2, 0]; // No batch pad, pad H and W by 2, no channel pad
        var layer = new PaddingLayer<float>(inputShape, padding, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, 8, 8, 3]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void PaddingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [2, 4, 4, 2];
        int[] padding = [0, 1, 1, 0];
        var layer = new PaddingLayer<float>(inputShape, padding, (IActivationFunction<float>?)null);

        var input = Tensor<float>.CreateRandom([2, 4, 4, 2]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void PaddingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [1, 4, 4, 2];
        int[] padding = [0, 1, 1, 0];
        var original = new PaddingLayer<float>(inputShape, padding, (IActivationFunction<float>?)null);
        var input = Tensor<float>.CreateRandom([1, 4, 4, 2]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region GaussianNoiseLayer Tests

    [Fact]
    public void GaussianNoiseLayer_ForwardPass_Training_AddsNoise()
    {
        // Arrange
        int[] inputShape = [64];
        float stddev = 0.1f;
        var layer = new GaussianNoiseLayer<float>(inputShape, stddev);
        layer.SetTrainingMode(true);

        var input = Tensor<float>.CreateDefault([4, 64], 1.0f);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);

        // In training mode, output should differ from input due to noise
        bool hasDifference = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output[i] - 1.0f) > 0.001f)
            {
                hasDifference = true;
                break;
            }
        }
        Assert.True(hasDifference, "Gaussian noise should modify values during training");
    }

    [Fact]
    public void GaussianNoiseLayer_ForwardPass_Inference_PreservesInput()
    {
        // Arrange
        int[] inputShape = [64];
        float stddev = 0.1f;
        var layer = new GaussianNoiseLayer<float>(inputShape, stddev);
        layer.SetTrainingMode(false);

        var input = Tensor<float>.CreateDefault([4, 64], 1.0f);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);

        // In inference mode, no noise should be added
        for (int i = 0; i < output.Length; i++)
        {
            Assert.Equal(1.0f, output[i]);
        }
    }

    [Fact]
    public void GaussianNoiseLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [32];
        var original = new GaussianNoiseLayer<float>(inputShape, 0.1f);
        original.SetTrainingMode(false);
        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region CrossAttentionLayer Tests

    [Fact]
    public void CrossAttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int queryDim = 64;
        int contextDim = 64;
        int numHeads = 4;
        int sequenceLength = 10;
        var layer = new CrossAttentionLayer<float>(queryDim, contextDim, numHeads, sequenceLength);

        var query = Tensor<float>.CreateRandom([2, sequenceLength, queryDim]);
        var context = Tensor<float>.CreateRandom([2, 15, contextDim]);

        // Act
        var output = layer.Forward(query, context);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(query.Shape, output.Shape); // output shape matches query shape
    }

    [Fact]
    public void CrossAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int queryDim = 32;
        int contextDim = 32;
        int numHeads = 2;
        int sequenceLength = 5;
        var layer = new CrossAttentionLayer<float>(queryDim, contextDim, numHeads, sequenceLength);

        var query = Tensor<float>.CreateRandom([2, sequenceLength, queryDim]);
        var context = Tensor<float>.CreateRandom([2, 8, contextDim]);
        var output = layer.Forward(query, context);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(query.Shape, gradient.Shape);
    }

    [Fact]
    public void CrossAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int queryDim = 32;
        int contextDim = 32;
        int numHeads = 2;
        int sequenceLength = 4;
        var original = new CrossAttentionLayer<float>(queryDim, contextDim, numHeads, sequenceLength);
        var query = Tensor<float>.CreateRandom([1, sequenceLength, queryDim]);
        var context = Tensor<float>.CreateRandom([1, 6, contextDim]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(query, context);
        var cloneOutput = clone.Forward(query, context);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void CrossAttentionLayer_ParameterCount_IsPositive()
    {
        // Arrange
        int queryDim = 64;
        int contextDim = 64;
        int numHeads = 4;
        var layer = new CrossAttentionLayer<float>(queryDim, contextDim, numHeads);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region ConvolutionalLayer Tests

    [Fact]
    public void ConvolutionalLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputDepth = 3;  // e.g., RGB image
        int inputHeight = 28;
        int inputWidth = 28;
        int outputDepth = 16;
        int kernelSize = 3;
        var layer = new ConvolutionalLayer<float>(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize);

        var input = Tensor<float>.CreateRandom([2, inputDepth, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(outputDepth, output.Shape[1]); // output channels
    }

    [Fact]
    public void ConvolutionalLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputDepth = 3;
        int inputHeight = 16;
        int inputWidth = 16;
        int outputDepth = 8;
        int kernelSize = 3;
        var layer = new ConvolutionalLayer<float>(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize);

        var input = Tensor<float>.CreateRandom([2, inputDepth, inputHeight, inputWidth]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void ConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputDepth = 1;
        int inputHeight = 8;
        int inputWidth = 8;
        int outputDepth = 4;
        int kernelSize = 3;
        var original = new ConvolutionalLayer<float>(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize);
        var input = Tensor<float>.CreateRandom([1, inputDepth, inputHeight, inputWidth]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void ConvolutionalLayer_ParameterCount_IsPositive()
    {
        // Arrange
        int inputDepth = 3;
        int inputHeight = 16;
        int inputWidth = 16;
        int outputDepth = 8;
        int kernelSize = 3;
        var layer = new ConvolutionalLayer<float>(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - Conv layer has kernels and biases
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region BatchNormalizationLayer Tests

    [Fact]
    public void BatchNormalizationLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int numFeatures = 64;
        var layer = new BatchNormalizationLayer<float>(numFeatures);

        var input = Tensor<float>.CreateRandom([4, numFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void BatchNormalizationLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int numFeatures = 32;
        var layer = new BatchNormalizationLayer<float>(numFeatures);
        layer.SetTrainingMode(true);

        var input = Tensor<float>.CreateRandom([4, numFeatures]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void BatchNormalizationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int numFeatures = 16;
        var original = new BatchNormalizationLayer<float>(numFeatures);
        var input = Tensor<float>.CreateRandom([2, numFeatures]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void BatchNormalizationLayer_ParameterCount_IsPositive()
    {
        // Arrange
        int numFeatures = 64;
        var layer = new BatchNormalizationLayer<float>(numFeatures);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - BatchNorm has gamma, beta, running mean, running var
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region MaxPoolingLayer Tests

    [Fact]
    public void MaxPoolingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        // inputShape = [C, H, W] format (channels, height, width)
        int[] inputShape = [3, 28, 28];
        int poolSize = 2;
        int stride = 2;
        var layer = new MaxPoolingLayer<float>(inputShape, poolSize, stride);

        // Input tensor = [B, C, H, W] format
        var input = Tensor<float>.CreateRandom([4, 3, 28, 28]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // batch preserved
        Assert.Equal(3, output.Shape[1]); // channels preserved
        Assert.Equal(14, output.Shape[2]); // 28/2 = 14 (height)
        Assert.Equal(14, output.Shape[3]); // 28/2 = 14 (width)
    }

    [Fact]
    public void MaxPoolingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        // inputShape = [C, H, W] format
        int[] inputShape = [2, 16, 16];
        int poolSize = 2;
        int stride = 2;
        var layer = new MaxPoolingLayer<float>(inputShape, poolSize, stride);

        // Input tensor = [B, C, H, W] format
        var input = Tensor<float>.CreateRandom([2, 2, 16, 16]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void MaxPoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        // inputShape = [C, H, W] format
        int[] inputShape = [1, 8, 8];
        int poolSize = 2;
        int stride = 2;
        var original = new MaxPoolingLayer<float>(inputShape, poolSize, stride);
        // Input tensor = [B, C, H, W] format
        var input = Tensor<float>.CreateRandom([1, 1, 8, 8]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region AveragePoolingLayer Tests

    [Fact]
    public void AveragePoolingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        // inputShape = [C, H, W] format
        int[] inputShape = [3, 16, 16];
        int poolSize = 2;
        int stride = 2;
        var layer = new AveragePoolingLayer<float>(inputShape, poolSize, stride);

        // Input tensor = [B, C, H, W] format
        var input = Tensor<float>.CreateRandom([4, 3, 16, 16]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // batch preserved
        Assert.Equal(3, output.Shape[1]); // channels preserved
        Assert.Equal(8, output.Shape[2]); // 16/2 = 8 (height)
        Assert.Equal(8, output.Shape[3]); // 16/2 = 8 (width)
    }

    [Fact]
    public void AveragePoolingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        // inputShape = [C, H, W] format
        int[] inputShape = [2, 8, 8];
        int poolSize = 2;
        int stride = 2;
        var layer = new AveragePoolingLayer<float>(inputShape, poolSize, stride);

        // Input tensor = [B, C, H, W] format
        var input = Tensor<float>.CreateRandom([2, 2, 8, 8]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void AveragePoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        // inputShape = [C, H, W] format
        int[] inputShape = [1, 4, 4];
        int poolSize = 2;
        int stride = 2;
        var original = new AveragePoolingLayer<float>(inputShape, poolSize, stride);
        // Input tensor = [B, C, H, W] format
        var input = Tensor<float>.CreateRandom([1, 1, 4, 4]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region EmbeddingLayer Tests

    [Fact]
    public void EmbeddingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int vocabSize = 1000;
        int embeddingDim = 64;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);

        // Create input with token indices (values 0 to vocabSize-1)
        var input = new Tensor<float>([4, 10]); // batch=4, sequence_length=10
        var rand = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = rand.Next(0, vocabSize);
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // batch
        Assert.Equal(10, output.Shape[1]); // sequence length
        Assert.Equal(embeddingDim, output.Shape[2]); // embedding dimension
    }

    [Fact]
    public void EmbeddingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int vocabSize = 500;
        int embeddingDim = 32;
        var original = new EmbeddingLayer<float>(vocabSize, embeddingDim);
        var input = new Tensor<float>([2, 5]);
        var rand = RandomHelper.CreateSeededRandom(123);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = rand.Next(0, vocabSize);
        }

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void EmbeddingLayer_ParameterCount_IsPositive()
    {
        // Arrange
        int vocabSize = 1000;
        int embeddingDim = 64;
        var layer = new EmbeddingLayer<float>(vocabSize, embeddingDim);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - Embedding has vocabSize * embeddingDim parameters
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
        Assert.Equal(vocabSize * embeddingDim, paramCount);
    }

    #endregion

    #region LayerNormalizationLayer Tests

    [Fact]
    public void LayerNormalizationLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int featureSize = 64;
        var layer = new LayerNormalizationLayer<float>(featureSize);

        var input = Tensor<float>.CreateRandom([4, featureSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void LayerNormalizationLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int featureSize = 32;
        var layer = new LayerNormalizationLayer<float>(featureSize);

        var input = Tensor<float>.CreateRandom([4, featureSize]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void LayerNormalizationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int featureSize = 16;
        var original = new LayerNormalizationLayer<float>(featureSize);
        var input = Tensor<float>.CreateRandom([2, featureSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    [Fact]
    public void LayerNormalizationLayer_ParameterCount_IsPositive()
    {
        // Arrange
        int featureSize = 64;
        var layer = new LayerNormalizationLayer<float>(featureSize);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - LayerNorm has gamma and beta (2 * featureSize)
        Assert.True(paramCount > 0, $"Expected positive parameter count but got {paramCount}");
    }

    #endregion

    #region GlobalPoolingLayer Tests

    [Fact]
    public void GlobalPoolingLayer_MaxPooling_ProducesValidOutput()
    {
        // Arrange
        // GlobalPoolingLayer uses NHWC format: inputShape = [batch, height, width, channels]
        int[] inputShape = [4, 16, 16, 3];
        var layer = new GlobalPoolingLayer<float>(inputShape, PoolingType.Max);

        // Input tensor = [batch, height, width, channels]
        var input = Tensor<float>.CreateRandom([4, 16, 16, 3]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // batch preserved
        // Output is [batch, 1, 1, channels] -> spatial dimensions are pooled
    }

    [Fact]
    public void GlobalPoolingLayer_AveragePooling_ProducesValidOutput()
    {
        // Arrange
        // GlobalPoolingLayer uses NHWC format: inputShape = [batch, height, width, channels]
        int[] inputShape = [2, 16, 16, 3];
        var layer = new GlobalPoolingLayer<float>(inputShape, PoolingType.Average);
        var input = Tensor<float>.CreateRandom([2, 16, 16, 3]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void GlobalPoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        // GlobalPoolingLayer uses NHWC format: inputShape = [batch, height, width, channels]
        int[] inputShape = [1, 8, 8, 2];
        var original = new GlobalPoolingLayer<float>(inputShape, PoolingType.Max);
        var input = Tensor<float>.CreateRandom([1, 8, 8, 2]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region GroupNormalizationLayer Tests

    [Fact]
    public void GroupNormalizationLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int numGroups = 4;
        int numChannels = 16;
        var layer = new GroupNormalizationLayer<float>(numGroups, numChannels);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, 16, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void GroupNormalizationLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int numGroups = 2;
        int numChannels = 8;
        var layer = new GroupNormalizationLayer<float>(numGroups, numChannels);

        var input = Tensor<float>.CreateRandom([2, 8, 4, 4]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void GroupNormalizationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int numGroups = 2;
        int numChannels = 8;
        var original = new GroupNormalizationLayer<float>(numGroups, numChannels);
        var input = Tensor<float>.CreateRandom([2, 8, 4, 4]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region InstanceNormalizationLayer Tests

    [Fact]
    public void InstanceNormalizationLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int numChannels = 16;
        var layer = new InstanceNormalizationLayer<float>(numChannels);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, 16, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void InstanceNormalizationLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int numChannels = 8;
        var layer = new InstanceNormalizationLayer<float>(numChannels);

        var input = Tensor<float>.CreateRandom([2, 8, 4, 4]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void InstanceNormalizationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int numChannels = 8;
        var original = new InstanceNormalizationLayer<float>(numChannels);
        var input = Tensor<float>.CreateRandom([2, 8, 4, 4]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region PositionalEncodingLayer Tests

    [Fact]
    public void PositionalEncodingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int maxSequenceLength = 32;
        int embeddingSize = 64;
        var layer = new PositionalEncodingLayer<float>(maxSequenceLength, embeddingSize);

        // Input: [batch, sequence, embedding]
        var input = Tensor<float>.CreateRandom([2, maxSequenceLength, embeddingSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void PositionalEncodingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int maxSequenceLength = 16;
        int embeddingSize = 32;
        var layer = new PositionalEncodingLayer<float>(maxSequenceLength, embeddingSize);

        var input = Tensor<float>.CreateRandom([2, maxSequenceLength, embeddingSize]);
        var output = layer.Forward(input);
        var upstreamGradient = Tensor<float>.CreateRandom(output.Shape);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
    }

    [Fact]
    public void PositionalEncodingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int maxSequenceLength = 16;
        int embeddingSize = 32;
        var original = new PositionalEncodingLayer<float>(maxSequenceLength, embeddingSize);
        var input = Tensor<float>.CreateRandom([2, maxSequenceLength, embeddingSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region DepthwiseSeparableConvolutionalLayer Tests

    [Fact]
    public void DepthwiseSeparableConvLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputChannels = 3;
        int outputChannels = 16;
        int height = 32;
        int width = 32;
        int kernelSize = 3;
        // Constructor: (inputDepth, outputDepth, kernelSize, inputHeight, inputWidth, stride, padding, activation)
        var layer = new DepthwiseSeparableConvolutionalLayer<float>(
            inputChannels, outputChannels, kernelSize, height, width, 1, 0, (IActivationFunction<float>)new ReLUActivation<float>());

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inputChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(outputChannels, output.Shape[1]); // output channels
    }

    [Fact]
    public void DepthwiseSeparableConvLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputChannels = 3;
        int outputChannels = 8;
        int height = 16;
        int width = 16;
        int kernelSize = 3;
        // Constructor: (inputDepth, outputDepth, kernelSize, inputHeight, inputWidth, stride, padding, activation)
        var original = new DepthwiseSeparableConvolutionalLayer<float>(
            inputChannels, outputChannels, kernelSize, height, width, 1, 0, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([1, inputChannels, height, width]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region DilatedConvolutionalLayer Tests

    [Fact]
    public void DilatedConvolutionalLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputChannels = 3;
        int outputChannels = 16;
        int height = 32;
        int width = 32;
        int kernelSize = 3;
        int dilation = 2;
        // Constructor: (inputDepth, outputDepth, kernelSize, inputHeight, inputWidth, dilation, stride, padding, activation)
        var layer = new DilatedConvolutionalLayer<float>(
            inputChannels, outputChannels, kernelSize, height, width, dilation, 1, 0, (IActivationFunction<float>)new ReLUActivation<float>());

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inputChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(outputChannels, output.Shape[1]); // output channels
    }

    [Fact]
    public void DilatedConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputChannels = 3;
        int outputChannels = 8;
        int height = 16;
        int width = 16;
        int kernelSize = 3;
        int dilation = 2;
        // Constructor: (inputDepth, outputDepth, kernelSize, inputHeight, inputWidth, dilation, stride, padding, activation)
        var original = new DilatedConvolutionalLayer<float>(
            inputChannels, outputChannels, kernelSize, height, width, dilation, 1, 0, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([1, inputChannels, height, width]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SeparableConvolutionalLayer Tests

    [Fact]
    public void SeparableConvolutionalLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int batchSize = 2;
        int inputChannels = 3;
        int outputChannels = 16;
        int height = 32;
        int width = 32;
        int kernelSize = 3;
        // SeparableConvolutionalLayer uses NHWC format: inputShape = [batch, height, width, channels]
        int[] inputShape = [batchSize, height, width, inputChannels];
        var layer = new SeparableConvolutionalLayer<float>(
            inputShape, outputChannels, kernelSize, 1, 0, (IActivationFunction<float>)new IdentityActivation<float>());

        // Input tensor matches inputShape: [batch, height, width, channels]
        var input = Tensor<float>.CreateRandom(inputShape);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(batchSize, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void SeparableConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int batchSize = 1;
        int inputChannels = 3;
        int outputChannels = 8;
        int height = 16;
        int width = 16;
        int kernelSize = 3;
        // SeparableConvolutionalLayer uses NHWC format: inputShape = [batch, height, width, channels]
        int[] inputShape = [batchSize, height, width, inputChannels];
        var original = new SeparableConvolutionalLayer<float>(
            inputShape, outputChannels, kernelSize, 1, 0, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = Tensor<float>.CreateRandom(inputShape);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region AdaptiveAveragePoolingLayer Tests

    [Fact]
    public void AdaptiveAveragePoolingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputChannels = 3;
        int inputHeight = 32;
        int inputWidth = 32;
        int outputHeight = 4;
        int outputWidth = 4;
        var layer = new AdaptiveAveragePoolingLayer<float>(
            inputChannels, inputHeight, inputWidth, outputHeight, outputWidth);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inputChannels, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(inputChannels, output.Shape[1]); // channels preserved
    }

    [Fact]
    public void AdaptiveAveragePoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputChannels = 3;
        int inputHeight = 16;
        int inputWidth = 16;
        var original = new AdaptiveAveragePoolingLayer<float>(
            inputChannels, inputHeight, inputWidth, 2, 2);
        var input = Tensor<float>.CreateRandom([1, inputChannels, inputHeight, inputWidth]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region TimeEmbeddingLayer Tests

    [Fact]
    public void TimeEmbeddingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int embeddingDim = 64;
        int outputDim = 128;
        var layer = new TimeEmbeddingLayer<float>(embeddingDim, outputDim);

        // Input: timestep values
        var input = Tensor<float>.CreateRandom([4, 1]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void TimeEmbeddingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int embeddingDim = 32;
        int outputDim = 64;
        var original = new TimeEmbeddingLayer<float>(embeddingDim, outputDim);
        var input = Tensor<float>.CreateRandom([2, 1]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region MeanLayer Tests

    [Fact]
    public void MeanLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int[] inputShape = [4, 8, 16];
        int axis = 1; // Mean over axis 1
        var layer = new MeanLayer<float>(inputShape, axis);

        var input = Tensor<float>.CreateRandom([2, 4, 8, 16]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void MeanLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [4, 8];
        int axis = 0;
        var original = new MeanLayer<float>(inputShape, axis);
        var input = Tensor<float>.CreateRandom([2, 4, 8]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region LambdaLayer Tests

    [Fact]
    public void LambdaLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int[] inputShape = [10];
        int[] outputShape = [10];
        // Simple identity-like function for testing
        Func<Tensor<float>, Tensor<float>> forwardFunction = input => input;
        var layer = new LambdaLayer<float>(inputShape, outputShape, forwardFunction, null,
            (IActivationFunction<float>)new IdentityActivation<float>());

        var input = Tensor<float>.CreateRandom([4, 10]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void LambdaLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [8];
        int[] outputShape = [8];
        Func<Tensor<float>, Tensor<float>> forwardFunction = input => input;
        var original = new LambdaLayer<float>(inputShape, outputShape, forwardFunction, null,
            (IActivationFunction<float>)new IdentityActivation<float>());
        var input = Tensor<float>.CreateRandom([2, 8]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region LocallyConnectedLayer Tests

    [Fact]
    public void LocallyConnectedLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputHeight = 16;
        int inputWidth = 16;
        int inputChannels = 3;
        int outputChannels = 8;
        int kernelSize = 3;
        int stride = 1;
        var layer = new LocallyConnectedLayer<float>(
            inputHeight, inputWidth, inputChannels, outputChannels, kernelSize, stride,
            (IActivationFunction<float>)new ReLUActivation<float>());

        // LocallyConnectedLayer expects NHWC format: [batch, height, width, channels]
        var input = Tensor<float>.CreateRandom([2, inputHeight, inputWidth, inputChannels]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void LocallyConnectedLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputHeight = 8;
        int inputWidth = 8;
        int inputChannels = 2;
        int outputChannels = 4;
        int kernelSize = 3;
        int stride = 1;
        var original = new LocallyConnectedLayer<float>(
            inputHeight, inputWidth, inputChannels, outputChannels, kernelSize, stride,
            (IActivationFunction<float>)new ReLUActivation<float>());
        // LocallyConnectedLayer expects NHWC format: [batch, height, width, channels]
        var input = Tensor<float>.CreateRandom([1, inputHeight, inputWidth, inputChannels]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SpectralNormalizationLayer Tests

    [Fact]
    public void SpectralNormalizationLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        var innerLayer = new DenseLayer<float>(64, 32);
        var layer = new SpectralNormalizationLayer<float>(innerLayer, powerIterations: 1);

        var input = Tensor<float>.CreateRandom([4, 64]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // batch preserved
        Assert.Equal(32, output.Shape[1]); // output features
    }

    [Fact]
    public void SpectralNormalizationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var innerLayer = new DenseLayer<float>(32, 16);
        var original = new SpectralNormalizationLayer<float>(innerLayer, powerIterations: 2);
        var input = Tensor<float>.CreateRandom([2, 32]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region CroppingLayer Tests

    [Fact]
    public void CroppingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        // CroppingLayer constructor: (inputShape, cropTop[], cropBottom[], cropLeft[], cropRight[], activation)
        // Each crop array must have the same length as inputShape
        int[] inputShape = [32, 32, 3]; // height, width, channels
        int[] cropTop = [2, 2, 0];      // crop 2 from top of height, 2 from left of width, 0 from channels
        int[] cropBottom = [2, 2, 0];   // crop 2 from bottom of height, 2 from right of width, 0 from channels
        int[] cropLeft = [0, 0, 0];     // additional left cropping per dimension
        int[] cropRight = [0, 0, 0];    // additional right cropping per dimension
        var layer = new CroppingLayer<float>(inputShape, cropTop, cropBottom, cropLeft, cropRight,
            (IActivationFunction<float>)new IdentityActivation<float>());

        // Input: matches inputShape
        var input = Tensor<float>.CreateRandom([2, 32, 32, 3]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void CroppingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        // Each crop array must have the same length as inputShape
        int[] inputShape = [16, 16, 3];
        int[] cropTop = [1, 1, 0];
        int[] cropBottom = [1, 1, 0];
        int[] cropLeft = [0, 0, 0];
        int[] cropRight = [0, 0, 0];
        var original = new CroppingLayer<float>(inputShape, cropTop, cropBottom, cropLeft, cropRight,
            (IActivationFunction<float>)new IdentityActivation<float>());
        var input = Tensor<float>.CreateRandom([1, 16, 16, 3]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SubpixelConvolutionalLayer Tests

    [Fact]
    public void SubpixelConvolutionalLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        // Constructor: (inputDepth, outputDepth, upscaleFactor, kernelSize, inputHeight, inputWidth, activation)
        int inputDepth = 12;
        int outputDepth = 3;
        int upscaleFactor = 2;
        int kernelSize = 3;
        int height = 8;
        int width = 8;
        var layer = new SubpixelConvolutionalLayer<float>(
            inputDepth, outputDepth, upscaleFactor, kernelSize, height, width,
            (IActivationFunction<float>)new ReLUActivation<float>());

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inputDepth, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void SubpixelConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputDepth = 4;
        int outputDepth = 1;
        int upscaleFactor = 2;
        int kernelSize = 3;
        var original = new SubpixelConvolutionalLayer<float>(
            inputDepth, outputDepth, upscaleFactor, kernelSize, 8, 8,
            (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([1, inputDepth, 8, 8]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region PoolingLayer Tests

    [Fact]
    public void PoolingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputDepth = 3;
        int inputHeight = 16;
        int inputWidth = 16;
        int poolSize = 2;
        int stride = 2;
        var layer = new PoolingLayer<float>(inputDepth, inputHeight, inputWidth, poolSize, stride, PoolingType.Max);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inputDepth, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(inputDepth, output.Shape[1]); // channels preserved
        Assert.Equal(8, output.Shape[2]); // height halved
        Assert.Equal(8, output.Shape[3]); // width halved
    }

    [Fact]
    public void PoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputDepth = 3;
        int inputHeight = 8;
        int inputWidth = 8;
        int poolSize = 2;
        int stride = 2;
        var original = new PoolingLayer<float>(inputDepth, inputHeight, inputWidth, poolSize, stride, PoolingType.Average);
        var input = Tensor<float>.CreateRandom([1, inputDepth, inputHeight, inputWidth]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region RecurrentLayer Tests

    [Fact]
    public void RecurrentLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 10;
        int hiddenSize = 20;
        var layer = new RecurrentLayer<float>(inputSize, hiddenSize,
            (IActivationFunction<float>)new TanhActivation<float>());

        // Input: [batch, sequenceLength, inputSize]
        var input = Tensor<float>.CreateRandom([2, 5, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void RecurrentLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 8;
        int hiddenSize = 16;
        var original = new RecurrentLayer<float>(inputSize, hiddenSize,
            (IActivationFunction<float>)new TanhActivation<float>());
        var input = Tensor<float>.CreateRandom([1, 4, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region FullyConnectedLayer Tests

    [Fact]
    public void FullyConnectedLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new FullyConnectedLayer<float>(inputSize, outputSize,
            (IActivationFunction<float>)new ReLUActivation<float>());

        // Input: [batch, inputSize]
        var input = Tensor<float>.CreateRandom([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void FullyConnectedLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var original = new FullyConnectedLayer<float>(inputSize, outputSize,
            (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region BidirectionalLayer Tests

    [Fact]
    public void BidirectionalLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 10;
        int hiddenSize = 20;
        int sequenceLength = 5;
        int[] inputShape = [sequenceLength, inputSize];
        var innerLayer = new LSTMLayer<float>(inputSize, hiddenSize, inputShape,
            (IActivationFunction<float>)new TanhActivation<float>());
        var layer = new BidirectionalLayer<float>(innerLayer, mergeMode: true,
            activationFunction: (IActivationFunction<float>)new IdentityActivation<float>());

        // Input: [batch, sequenceLength, inputSize]
        var input = Tensor<float>.CreateRandom([2, sequenceLength, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void BidirectionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 8;
        int hiddenSize = 16;
        int sequenceLength = 4;
        var innerLayer = new GRULayer<float>(inputSize, hiddenSize, returnSequences: false,
            activation: (IActivationFunction<float>)new TanhActivation<float>());
        var original = new BidirectionalLayer<float>(innerLayer, mergeMode: false,
            activationFunction: (IActivationFunction<float>)new IdentityActivation<float>());
        var input = Tensor<float>.CreateRandom([1, sequenceLength, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region TimeDistributedLayer Tests

    [Fact]
    public void TimeDistributedLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var innerLayer = new DenseLayer<float>(inputSize, outputSize);
        var layer = new TimeDistributedLayer<float>(innerLayer,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());

        // Input: [batch, sequenceLength, inputSize]
        var input = Tensor<float>.CreateRandom([2, 5, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void TimeDistributedLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var innerLayer = new DenseLayer<float>(inputSize, outputSize);
        var original = new TimeDistributedLayer<float>(innerLayer,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([1, 4, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SequenceLastLayer Tests

    [Fact]
    public void SequenceLastLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int featureSize = 32;
        var layer = new SequenceLastLayer<float>(featureSize);

        // Input: [batch, sequenceLength, featureSize]
        var input = Tensor<float>.CreateRandom([2, 5, featureSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SequenceLastLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int featureSize = 16;
        var original = new SequenceLastLayer<float>(featureSize);
        var input = Tensor<float>.CreateRandom([1, 4, featureSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region Upsample3DLayer Tests

    [Fact]
    public void Upsample3DLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        // inputShape: [channels, depth, height, width]
        int[] inputShape = [3, 4, 8, 8];
        int scaleFactor = 2;
        var layer = new Upsample3DLayer<float>(inputShape, scaleFactor);

        // Input: [batch, channels, depth, height, width]
        var input = Tensor<float>.CreateRandom([2, 3, 4, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape[0]); // batch preserved
    }

    [Fact]
    public void Upsample3DLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [2, 2, 4, 4];
        int scaleFactor = 2;
        var original = new Upsample3DLayer<float>(inputShape, scaleFactor);
        var input = Tensor<float>.CreateRandom([1, 2, 2, 4, 4]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region GraphConvolutionalLayer Tests

    [Fact]
    public void GraphConvolutionalLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int numNodes = 10;
        int inputFeatures = 16;
        int outputFeatures = 8;
        var layer = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures,
            (IActivationFunction<float>)new ReLUActivation<float>());

        // Create adjacency matrix (all ones for full connectivity)
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        layer.SetAdjacencyMatrix(adjacencyMatrix);

        // Node features: [numNodes, inputFeatures]
        var nodeFeatures = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void GraphConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 4;
        var original = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures,
            (IActivationFunction<float>)new IdentityActivation<float>());

        // Create adjacency matrix and set it
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        original.SetAdjacencyMatrix(adjacencyMatrix);

        var input = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var clone = original.Clone();
        // Clone needs adjacency matrix set too
        if (clone is GraphConvolutionalLayer<float> cloneGcn)
        {
            cloneGcn.SetAdjacencyMatrix(adjacencyMatrix);
        }
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region GraphAttentionLayer Tests

    [Fact]
    public void GraphAttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int numNodes = 10;
        int inputFeatures = 16;
        int outputFeatures = 8;
        int numHeads = 2;
        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads,
            activationFunction: (IActivationFunction<float>)new LeakyReLUActivation<float>());

        // Create adjacency matrix (all ones for full connectivity)
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        layer.SetAdjacencyMatrix(adjacencyMatrix);

        // Node features: [numNodes, inputFeatures]
        var nodeFeatures = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void GraphAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 4;
        int numHeads = 1;
        var original = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads);

        // Create adjacency matrix and set it
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        original.SetAdjacencyMatrix(adjacencyMatrix);

        var input = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var clone = original.Clone();
        // Clone needs adjacency matrix set too
        if (clone is GraphAttentionLayer<float> cloneGat)
        {
            cloneGat.SetAdjacencyMatrix(adjacencyMatrix);
        }
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region AttentionLayer Tests

    [Fact]
    public void AttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        var layer = new AttentionLayer<float>(inputSize, attentionSize,
            (IActivationFunction<float>)new SoftmaxActivation<float>());

        // Input: [batch, sequenceLength, inputSize]
        var input = Tensor<float>.CreateRandom([2, 8, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void AttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 16;
        int attentionSize = 8;
        var original = new AttentionLayer<float>(inputSize, attentionSize,
            (IActivationFunction<float>)new SoftmaxActivation<float>());
        var input = Tensor<float>.CreateRandom([1, 4, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region DecoderLayer Tests

    [Fact]
    public void DecoderLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        int feedForwardSize = 64;
        var layer = new DecoderLayer<float>(inputSize, attentionSize, feedForwardSize,
            (IActivationFunction<float>)new ReLUActivation<float>());

        // Input: [batch, sequenceLength, inputSize]
        var input = Tensor<float>.CreateRandom([2, 8, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void DecoderLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 16;
        int attentionSize = 8;
        int feedForwardSize = 32;
        var original = new DecoderLayer<float>(inputSize, attentionSize, feedForwardSize,
            (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([1, 4, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region RBMLayer Tests

    [Fact]
    public void RBMLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int visibleUnits = 64;
        int hiddenUnits = 32;
        var layer = new RBMLayer<float>(visibleUnits, hiddenUnits,
            (IActivationFunction<float>)new SigmoidActivation<float>());

        // Input: [batch, visibleUnits]
        var input = Tensor<float>.CreateRandom([4, visibleUnits]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void RBMLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int visibleUnits = 32;
        int hiddenUnits = 16;
        var original = new RBMLayer<float>(visibleUnits, hiddenUnits,
            (IActivationFunction<float>)new SigmoidActivation<float>());
        var input = Tensor<float>.CreateRandom([2, visibleUnits]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region ReservoirLayer Tests

    [Fact]
    public void ReservoirLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 16;
        int reservoirSize = 64;
        var layer = new ReservoirLayer<float>(inputSize, reservoirSize);

        // Input: [batch, inputSize]
        var input = Tensor<float>.CreateRandom([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void ReservoirLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 8;
        int reservoirSize = 32;
        var original = new ReservoirLayer<float>(inputSize, reservoirSize);
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SparseLinearLayer Tests

    [Fact]
    public void SparseLinearLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputFeatures = 64;
        int outputFeatures = 32;
        var layer = new SparseLinearLayer<float>(inputFeatures, outputFeatures, sparsity: 0.5);

        // Input: [batch, inputFeatures]
        var input = Tensor<float>.CreateRandom([4, inputFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SparseLinearLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputFeatures = 32;
        int outputFeatures = 16;
        var original = new SparseLinearLayer<float>(inputFeatures, outputFeatures, sparsity: 0.5);
        var input = Tensor<float>.CreateRandom([2, inputFeatures]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SpatialTransformerLayer Tests

    [Fact]
    public void SpatialTransformerLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputHeight = 28;
        int inputWidth = 28;
        int outputHeight = 28;
        int outputWidth = 28;
        var layer = new SpatialTransformerLayer<float>(inputHeight, inputWidth, outputHeight, outputWidth,
            (IActivationFunction<float>)new TanhActivation<float>());

        // Input: [batch, height, width]
        var input = Tensor<float>.CreateRandom([2, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SpatialTransformerLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputHeight = 16;
        int inputWidth = 16;
        int outputHeight = 16;
        int outputWidth = 16;
        var original = new SpatialTransformerLayer<float>(inputHeight, inputWidth, outputHeight, outputWidth,
            (IActivationFunction<float>)new TanhActivation<float>());
        var input = Tensor<float>.CreateRandom([1, inputHeight, inputWidth]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region PatchEmbeddingLayer Tests

    [Fact]
    public void PatchEmbeddingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int imageHeight = 28;
        int imageWidth = 28;
        int channels = 3;
        int patchSize = 7;
        int embeddingDim = 64;
        var layer = new PatchEmbeddingLayer<float>(imageHeight, imageWidth, channels, patchSize, embeddingDim);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, channels, imageHeight, imageWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void PatchEmbeddingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int imageHeight = 16;
        int imageWidth = 16;
        int channels = 1;
        int patchSize = 4;
        int embeddingDim = 32;
        var original = new PatchEmbeddingLayer<float>(imageHeight, imageWidth, channels, patchSize, embeddingDim);
        var input = Tensor<float>.CreateRandom([1, channels, imageHeight, imageWidth]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region TransitionLayer Tests

    [Fact]
    public void TransitionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputChannels = 64;
        int inputHeight = 16;
        int inputWidth = 16;
        double compressionFactor = 0.5;
        var layer = new TransitionLayer<float>(inputChannels, inputHeight, inputWidth, compressionFactor);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inputChannels, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void TransitionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputChannels = 32;
        int inputHeight = 8;
        int inputWidth = 8;
        double compressionFactor = 0.5;
        var original = new TransitionLayer<float>(inputChannels, inputHeight, inputWidth, compressionFactor);
        var input = Tensor<float>.CreateRandom([1, inputChannels, inputHeight, inputWidth]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region BasicBlock Tests

    [Fact]
    public void BasicBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inChannels = 64;
        int outChannels = 64;
        int height = 56;
        int width = 56;
        var layer = new BasicBlock<float>(inChannels, outChannels, stride: 1, inputHeight: height, inputWidth: width);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void BasicBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inChannels = 32;
        int outChannels = 32;
        int height = 28;
        int width = 28;
        var original = new BasicBlock<float>(inChannels, outChannels, stride: 1, inputHeight: height, inputWidth: width);
        var input = Tensor<float>.CreateRandom([1, inChannels, height, width]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region BottleneckBlock Tests

    [Fact]
    public void BottleneckBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inChannels = 64;
        int baseChannels = 64;
        int height = 56;
        int width = 56;
        var layer = new BottleneckBlock<float>(inChannels, baseChannels, stride: 1, inputHeight: height, inputWidth: width);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void BottleneckBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inChannels = 32;
        int baseChannels = 32;
        int height = 28;
        int width = 28;
        var original = new BottleneckBlock<float>(inChannels, baseChannels, stride: 1, inputHeight: height, inputWidth: width);
        var input = Tensor<float>.CreateRandom([1, inChannels, height, width]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region CapsuleLayer Tests

    [Fact]
    public void CapsuleLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputCapsules = 32;
        int inputDimension = 8;
        int numCapsules = 10;
        int capsuleDimension = 16;
        int numRoutingIterations = 3;
        var layer = new CapsuleLayer<float>(inputCapsules, inputDimension, numCapsules, capsuleDimension, numRoutingIterations);

        // Input: [batch, inputCapsules, inputDimension]
        var input = Tensor<float>.CreateRandom([2, inputCapsules, inputDimension]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void CapsuleLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputCapsules = 16;
        int inputDimension = 4;
        int numCapsules = 5;
        int capsuleDimension = 8;
        int numRoutingIterations = 2;
        var original = new CapsuleLayer<float>(inputCapsules, inputDimension, numCapsules, capsuleDimension, numRoutingIterations);
        var input = Tensor<float>.CreateRandom([1, inputCapsules, inputDimension]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region DenseBlock Tests

    [Fact]
    public void DenseBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputChannels = 64;
        int numLayers = 4;
        int growthRate = 32;
        int height = 28;
        int width = 28;
        var layer = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inputChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void DenseBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputChannels = 32;
        int numLayers = 2;
        int growthRate = 16;
        int height = 14;
        int width = 14;
        var original = new DenseBlock<float>(inputChannels, numLayers, growthRate, height, width);
        var input = Tensor<float>.CreateRandom([1, inputChannels, height, width]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region GraphIsomorphismLayer Tests

    [Fact]
    public void GraphIsomorphismLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int numNodes = 10;
        int inputFeatures = 16;
        int outputFeatures = 32;
        var layer = new GraphIsomorphismLayer<float>(inputFeatures, outputFeatures,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());

        // Set adjacency matrix before forward pass
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        layer.SetAdjacencyMatrix(adjacencyMatrix);

        // Input: [numNodes, inputFeatures]
        var input = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void GraphIsomorphismLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;
        var original = new GraphIsomorphismLayer<float>(inputFeatures, outputFeatures,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        original.SetAdjacencyMatrix(adjacencyMatrix);
        var input = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var clone = (GraphIsomorphismLayer<float>)original.Clone();
        clone.SetAdjacencyMatrix(adjacencyMatrix);
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region GraphSAGELayer Tests

    [Fact]
    public void GraphSAGELayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int numNodes = 10;
        int inputFeatures = 16;
        int outputFeatures = 32;
        var layer = new GraphSAGELayer<float>(inputFeatures, outputFeatures,
            aggregatorType: SAGEAggregatorType.Mean,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());

        // Set adjacency matrix before forward pass
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        layer.SetAdjacencyMatrix(adjacencyMatrix);

        // Input: [numNodes, inputFeatures]
        var input = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void GraphSAGELayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;
        var original = new GraphSAGELayer<float>(inputFeatures, outputFeatures,
            aggregatorType: SAGEAggregatorType.Mean,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        original.SetAdjacencyMatrix(adjacencyMatrix);
        var input = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var clone = (GraphSAGELayer<float>)original.Clone();
        clone.SetAdjacencyMatrix(adjacencyMatrix);
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region InvertedResidualBlock Tests

    [Fact]
    public void InvertedResidualBlock_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inChannels = 32;
        int outChannels = 64;
        int height = 28;
        int width = 28;
        var layer = new InvertedResidualBlock<float>(inChannels, outChannels, height, width,
            expansionRatio: 6, stride: 1);

        // Input: [batch, channels, height, width]
        var input = Tensor<float>.CreateRandom([2, inChannels, height, width]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void InvertedResidualBlock_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inChannels = 16;
        int outChannels = 32;
        int height = 14;
        int width = 14;
        var original = new InvertedResidualBlock<float>(inChannels, outChannels, height, width,
            expansionRatio: 4, stride: 1);
        var input = Tensor<float>.CreateRandom([1, inChannels, height, width]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region MessagePassingLayer Tests

    [Fact]
    public void MessagePassingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int numNodes = 10;
        int inputFeatures = 16;
        int outputFeatures = 32;
        var layer = new MessagePassingLayer<float>(inputFeatures, outputFeatures,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());

        // Set adjacency matrix before forward pass
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        layer.SetAdjacencyMatrix(adjacencyMatrix);

        // Input: [numNodes, inputFeatures]
        var input = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void MessagePassingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;
        var original = new MessagePassingLayer<float>(inputFeatures, outputFeatures,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());
        var adjacencyMatrix = Tensor<float>.CreateDefault([numNodes, numNodes], 1.0f);
        original.SetAdjacencyMatrix(adjacencyMatrix);
        var input = Tensor<float>.CreateRandom([numNodes, inputFeatures]);

        // Act
        var clone = (MessagePassingLayer<float>)original.Clone();
        clone.SetAdjacencyMatrix(adjacencyMatrix);
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region SpikingLayer Tests

    [Fact]
    public void SpikingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var layer = new SpikingLayer<float>(inputSize, outputSize,
            neuronType: SpikingNeuronType.LeakyIntegrateAndFire);

        // Input: [batch, inputSize]
        var input = Tensor<float>.CreateRandom([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SpikingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var original = new SpikingLayer<float>(inputSize, outputSize,
            neuronType: SpikingNeuronType.LeakyIntegrateAndFire);
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region QuantumLayer Tests

    [Fact]
    public void QuantumLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 4;
        int outputSize = 4;
        int numQubits = 2;
        var layer = new QuantumLayer<float>(inputSize, outputSize, numQubits);

        // Input: [batch, inputSize]
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void QuantumLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 4;
        int outputSize = 4;
        int numQubits = 2;
        var original = new QuantumLayer<float>(inputSize, outputSize, numQubits);
        var input = Tensor<float>.CreateRandom([1, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region AnomalyDetectorLayer Tests

    [Fact]
    public void AnomalyDetectorLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 16;
        double threshold = 2.0;
        var layer = new AnomalyDetectorLayer<float>(inputSize, threshold);

        // Input: [batch, inputSize]
        var input = Tensor<float>.CreateRandom([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void AnomalyDetectorLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 8;
        double threshold = 1.5;
        var original = new AnomalyDetectorLayer<float>(inputSize, threshold);
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region RBFLayer Tests

    [Fact]
    public void RBFLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var rbf = new GaussianRBF<float>(1.0f);
        var layer = new RBFLayer<float>(inputSize, outputSize, rbf);

        // Input: [batch, inputSize]
        var input = Tensor<float>.CreateRandom([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void RBFLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 8;
        int outputSize = 4;
        var rbf = new GaussianRBF<float>(1.0f);
        var original = new RBFLayer<float>(inputSize, outputSize, rbf);
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region ExpertLayer Tests

    [Fact]
    public void ExpertLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 16;
        int hiddenSize = 32;
        int outputSize = 8;
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(inputSize, hiddenSize, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(hiddenSize, outputSize, (IActivationFunction<float>)new IdentityActivation<float>())
        };
        var layer = new ExpertLayer<float>(layers, [inputSize], [outputSize]);

        // Input: [batch, inputSize]
        var input = Tensor<float>.CreateRandom([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void ExpertLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 8;
        int outputSize = 4;
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>())
        };
        var original = new ExpertLayer<float>(layers, [inputSize], [outputSize]);
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var clone = original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
    }

    #endregion

    #region HyperbolicLinearLayer Tests

    [Fact]
    public void HyperbolicLinearLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange
        int inputFeatures = 6;
        int outputFeatures = 4;
        var layer = new HyperbolicLinearLayer<float>(inputFeatures, outputFeatures);
        var input = Tensor<float>.CreateRandom([3, inputFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal([3, outputFeatures], output.Shape);
    }

    [Fact]
    public void HyperbolicLinearLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange
        int inputFeatures = 5;
        int outputFeatures = 3;
        var layer = new HyperbolicLinearLayer<float>(inputFeatures, outputFeatures);
        var input = Tensor<float>.CreateRandom([2, 4, inputFeatures]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, 4, outputFeatures], output.Shape);
    }

    [Fact]
    public void HyperbolicLinearLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputFeatures = 4;
        int outputFeatures = 3;
        var layer = new HyperbolicLinearLayer<float>(inputFeatures, outputFeatures);
        var input = Tensor<float>.CreateRandom([2, inputFeatures]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = Tensor<float>.CreateRandom(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    #endregion

    #region OctonionLinearLayer Tests

    [Fact]
    public void OctonionLinearLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange
        int inputFeatures = 3;
        int outputFeatures = 2;
        var layer = new OctonionLinearLayer<float>(inputFeatures, outputFeatures);
        var input = Tensor<float>.CreateRandom([4, inputFeatures * 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([4, outputFeatures * 8], output.Shape);
    }

    [Fact]
    public void OctonionLinearLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange
        int inputFeatures = 2;
        int outputFeatures = 3;
        var layer = new OctonionLinearLayer<float>(inputFeatures, outputFeatures);
        var input = Tensor<float>.CreateRandom([2, 3, inputFeatures * 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, 3, outputFeatures * 8], output.Shape);
    }

    [Fact]
    public void OctonionLinearLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputFeatures = 2;
        int outputFeatures = 2;
        var layer = new OctonionLinearLayer<float>(inputFeatures, outputFeatures);
        var input = Tensor<float>.CreateRandom([3, inputFeatures * 8]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = Tensor<float>.CreateRandom(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    #endregion

    #region ReadoutLayer Tests

    [Fact]
    public void ReadoutLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 8;
        int outputSize = 5;
        var layer = new ReadoutLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([3, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([3, outputSize], output.Shape);
    }

    [Fact]
    public void ReadoutLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 6;
        int outputSize = 4;
        var layer = new ReadoutLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([2, 3, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, 3, outputSize], output.Shape);
    }

    [Fact]
    public void ReadoutLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 5;
        int outputSize = 3;
        var layer = new ReadoutLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = Tensor<float>.CreateRandom(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    #endregion

    #region MeasurementLayer Tests

    [Fact]
    public void MeasurementLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange
        int size = 8;
        var layer = new MeasurementLayer<float>(size);
        var input = Tensor<float>.CreateRandom([3, size]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void MeasurementLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange
        int size = 6;
        var layer = new MeasurementLayer<float>(size);
        var input = Tensor<float>.CreateRandom([2, 4, size]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void MeasurementLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int size = 5;
        var layer = new MeasurementLayer<float>(size);
        var input = Tensor<float>.CreateRandom([2, size]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = Tensor<float>.CreateRandom(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    #endregion

    #region MixtureOfExpertsLayer Tests

    [Fact]
    public void MixtureOfExpertsLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 6;
        int outputSize = 4;
        int numExperts = 2;
        var experts = new List<ILayer<float>>
        {
            new DenseLayer<float>(inputSize, outputSize),
            new DenseLayer<float>(inputSize, outputSize)
        };
        var router = new DenseLayer<float>(inputSize, numExperts);
        var layer = new MixtureOfExpertsLayer<float>(experts, router, [inputSize], [outputSize]);
        var input = Tensor<float>.CreateRandom([2, 3, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, 3, outputSize], output.Shape);
    }

    [Fact]
    public void MixtureOfExpertsLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 4;
        int outputSize = 3;
        int numExperts = 3;
        var experts = new List<ILayer<float>>
        {
            new DenseLayer<float>(inputSize, outputSize),
            new DenseLayer<float>(inputSize, outputSize),
            new DenseLayer<float>(inputSize, outputSize)
        };
        var router = new DenseLayer<float>(inputSize, numExperts);
        var layer = new MixtureOfExpertsLayer<float>(experts, router, [inputSize], [outputSize]);
        var input = Tensor<float>.CreateRandom([2, inputSize]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = Tensor<float>.CreateRandom(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void MixtureOfExpertsLayer_TopKRouting_ProducesValidOutput()
    {
        // Arrange
        int inputSize = 5;
        int outputSize = 2;
        int numExperts = 3;
        var experts = new List<ILayer<float>>
        {
            new DenseLayer<float>(inputSize, outputSize),
            new DenseLayer<float>(inputSize, outputSize),
            new DenseLayer<float>(inputSize, outputSize)
        };
        var router = new DenseLayer<float>(inputSize, numExperts);
        var layer = new MixtureOfExpertsLayer<float>(experts, router, [inputSize], [outputSize], topK: 1);
        var input = Tensor<float>.CreateRandom([3, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([3, outputSize], output.Shape);
    }

    #endregion
}
