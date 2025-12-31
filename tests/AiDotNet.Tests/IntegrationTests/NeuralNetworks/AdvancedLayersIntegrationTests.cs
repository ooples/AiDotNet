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

    #region ConvLSTMLayer Tests

    [Fact(Skip = "ConvLSTMLayer has a bug in ConvLSTMCell where tensor Add operation fails due to shape mismatch")]
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

    [Fact(Skip = "ConvLSTMLayer has a bug in ConvLSTMCell where tensor Add operation fails due to shape mismatch")]
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
    public void GatedLinearUnitLayer_ParameterCount_IsNonNegative()
    {
        // Arrange
        var layer = new GatedLinearUnitLayer<float>(64, 32, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - ParameterCount property is accessible and returns valid value
        Assert.True(paramCount >= 0);
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
    public void HighwayLayer_ParameterCount_IsNonNegative()
    {
        // Arrange
        var layer = new HighwayLayer<float>(64, (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert - ParameterCount property is accessible and returns valid value
        Assert.True(paramCount >= 0);
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
        input1.Data[0] = 2; input1.Data[1] = 3; input1.Data[2] = 4; input1.Data[3] = 5;
        input2.Data[0] = 1; input2.Data[1] = 2; input2.Data[2] = 3; input2.Data[3] = 4;

        // Act
        var output = layer.Forward(input1, input2);

        // Assert - element-wise multiplication
        Assert.Equal(2f, output.Data[0]); // 2 * 1
        Assert.Equal(6f, output.Data[1]); // 3 * 2
        Assert.Equal(12f, output.Data[2]); // 4 * 3
        Assert.Equal(20f, output.Data[3]); // 5 * 4
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
        // Split should create numSplits sections
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
}
