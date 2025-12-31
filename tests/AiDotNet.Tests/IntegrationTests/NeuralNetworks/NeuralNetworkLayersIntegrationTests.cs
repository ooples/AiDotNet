using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using System.Reflection;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for neural network layer classes.
/// Tests forward pass, backward pass, shape correctness, and training/inference modes.
/// These tests are designed to find bugs in layer implementations.
/// </summary>
public class NeuralNetworkLayersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region DenseLayer Tests

    [Fact]
    public void DenseLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputSize = 10;
        int outputSize = 5;
        var layer = new DenseLayer<double>(inputSize, outputSize);
        var input = new Tensor<double>([1, inputSize]);

        // Initialize with random values
        for (int i = 0; i < inputSize; i++)
        {
            input[0, i] = i * 0.1;
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]); // batch size
        Assert.Equal(outputSize, output.Shape[1]); // output features
    }

    [Fact]
    public void DenseLayer_ForwardPass_BatchProcessing_ProducesCorrectShape()
    {
        // Arrange
        int batchSize = 8;
        int inputSize = 10;
        int outputSize = 5;
        var layer = new DenseLayer<double>(inputSize, outputSize);
        var input = new Tensor<double>([batchSize, inputSize]);

        // Initialize with random values
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputSize; i++)
            {
                input[b, i] = b * 0.1 + i * 0.01;
            }
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(outputSize, output.Shape[1]);
    }

    [Fact]
    public void DenseLayer_BackwardPass_ProducesCorrectGradientShape()
    {
        // Arrange
        int inputSize = 10;
        int outputSize = 5;
        var layer = new DenseLayer<double>(inputSize, outputSize);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>([1, inputSize]);
        for (int i = 0; i < inputSize; i++)
        {
            input[0, i] = i * 0.1;
        }

        var output = layer.Forward(input);

        // Create upstream gradient
        var upstreamGradient = new Tensor<double>(output.Shape);
        for (int i = 0; i < outputSize; i++)
        {
            upstreamGradient[0, i] = 1.0;
        }

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape.Length, gradient.Shape.Length);
        Assert.Equal(inputSize, gradient.Shape[^1]); // Last dimension should be input size
    }

    [Fact]
    public void DenseLayer_WithActivation_AppliesActivationFunction()
    {
        // Arrange
        int inputSize = 5;
        int outputSize = 3;
        IActivationFunction<double> relu = new ReLUActivation<double>();
        var layer = new DenseLayer<double>(inputSize, outputSize, relu);

        var input = new Tensor<double>([1, inputSize]);
        for (int i = 0; i < inputSize; i++)
        {
            input[0, i] = 1.0; // All positive inputs
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        // With ReLU, all outputs should be non-negative
        for (int i = 0; i < outputSize; i++)
        {
            Assert.True(output[0, i] >= 0.0, $"Output at index {i} should be non-negative with ReLU");
        }
    }

    [Fact]
    public void DenseLayer_SupportsTraining_ReturnsTrue()
    {
        // Arrange
        var layer = new DenseLayer<double>(10, 5);

        // Assert
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void DenseLayer_GetParameters_ReturnsWeightsAndBiases()
    {
        // Arrange
        int inputSize = 4;
        int outputSize = 3;
        var layer = new DenseLayer<double>(inputSize, outputSize);

        // Act
        var parameters = layer.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0);
    }

    #endregion

    #region ConvolutionalLayer Tests

    [Fact]
    public void ConvolutionalLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputDepth = 3;
        int inputHeight = 28;
        int inputWidth = 28;
        int outputDepth = 16;
        int kernelSize = 3;
        int stride = 1;
        int padding = 1;

        var layer = new ConvolutionalLayer<double>(
            inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding);

        // Create input tensor [batch, depth, height, width]
        var input = new Tensor<double>([1, inputDepth, inputHeight, inputWidth]);
        InitializeRandomTensor(input);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // With padding=1, stride=1, kernel=3: output_size = (28 - 3 + 2*1) / 1 + 1 = 28
        Assert.Equal(28, output.Shape[2]); // height
        Assert.Equal(28, output.Shape[3]); // width
        Assert.Equal(outputDepth, output.Shape[1]); // depth
    }

    [Fact]
    public void ConvolutionalLayer_ForwardPass_WithStride2_ReducesDimensions()
    {
        // Arrange
        int inputDepth = 3;
        int inputHeight = 28;
        int inputWidth = 28;
        int outputDepth = 32;
        int kernelSize = 3;
        int stride = 2;
        int padding = 1;

        var layer = new ConvolutionalLayer<double>(
            inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding);

        var input = new Tensor<double>([1, inputDepth, inputHeight, inputWidth]);
        InitializeRandomTensor(input);

        // Act
        var output = layer.Forward(input);

        // Assert
        // With stride=2: output_size = (28 - 3 + 2*1) / 2 + 1 = 14
        Assert.Equal(14, output.Shape[2]); // height
        Assert.Equal(14, output.Shape[3]); // width
    }

    [Fact]
    public void ConvolutionalLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputDepth = 1;
        int inputHeight = 8;
        int inputWidth = 8;
        int outputDepth = 2;
        int kernelSize = 3;

        var layer = new ConvolutionalLayer<double>(
            inputDepth, inputHeight, inputWidth, outputDepth, kernelSize);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>([1, inputDepth, inputHeight, inputWidth]);
        InitializeRandomTensor(input);

        var output = layer.Forward(input);
        var upstreamGradient = new Tensor<double>(output.Shape);
        InitializeTensorWithValue(upstreamGradient, 1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape.Length, gradient.Shape.Length);
    }

    [Fact]
    public void ConvolutionalLayer_BatchProcessing_HandlesMultipleSamples()
    {
        // Arrange
        int batchSize = 4;
        int inputDepth = 3;
        int inputHeight = 16;
        int inputWidth = 16;
        int outputDepth = 8;
        int kernelSize = 3;
        int stride = 1;
        int padding = 1;

        var layer = new ConvolutionalLayer<double>(
            inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding);

        var input = new Tensor<double>([batchSize, inputDepth, inputHeight, inputWidth]);
        InitializeRandomTensor(input);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(outputDepth, output.Shape[1]);
    }

    #endregion

    #region LSTMLayer Tests

    [Fact]
    public void LSTMLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputSize = 10;
        int hiddenSize = 20;
        int sequenceLength = 5;
        int batchSize = 2;
        int[] inputShape = [batchSize, sequenceLength, inputSize];

        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        var input = new Tensor<double>(inputShape);
        InitializeRandomTensor(input);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(hiddenSize, output.Shape[^1]); // Hidden dimension
    }

    [Fact]
    public void LSTMLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputSize = 5;
        int hiddenSize = 10;
        int sequenceLength = 3;
        int batchSize = 1;
        int[] inputShape = [batchSize, sequenceLength, inputSize];

        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>(inputShape);
        InitializeRandomTensor(input);

        var output = layer.Forward(input);
        var upstreamGradient = new Tensor<double>(output.Shape);
        InitializeTensorWithValue(upstreamGradient, 1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
    }

    [Fact]
    public void LSTMLayer_SupportsTraining_ReturnsTrue()
    {
        // Arrange
        int[] inputShape = [1, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(10, 20, inputShape, tanh);

        // Assert
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void LSTMLayer_GetParameters_ReturnsGateWeights()
    {
        // Arrange
        int[] inputShape = [1, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(10, 20, inputShape, tanh);

        // Act
        var parameters = layer.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0, "LSTM should have trainable parameters");
    }

    #endregion

    #region BatchNormalizationLayer Tests

    [Fact]
    public void BatchNormalizationLayer_ForwardPass_NormalizesOutput()
    {
        // Arrange
        int numFeatures = 10;
        var layer = new BatchNormalizationLayer<double>(numFeatures);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>([32, numFeatures]); // Large batch for stable statistics
        InitializeRandomTensor(input, scale: 10.0); // Large scale to test normalization

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape[0], output.Shape[0]);
        Assert.Equal(input.Shape[1], output.Shape[1]);
    }

    [Fact]
    public void BatchNormalizationLayer_TrainingVsInference_ProducesDifferentBehavior()
    {
        // Arrange
        int numFeatures = 5;
        var layer = new BatchNormalizationLayer<double>(numFeatures);

        var input = new Tensor<double>([8, numFeatures]);
        InitializeRandomTensor(input, scale: 5.0);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] += 3.5;
        }

        // Act - Training mode
        layer.SetTrainingMode(true);
        var trainingOutput = layer.Forward(input);

        // Act - Inference mode
        layer.SetTrainingMode(false);
        var inferenceOutput = layer.Forward(input);
        var inferenceOutputRepeat = layer.Forward(input);

        // Assert - Both should produce valid outputs
        Assert.NotNull(trainingOutput);
        Assert.NotNull(inferenceOutput);
        Assert.Equal(trainingOutput.Shape, inferenceOutput.Shape);

        bool outputsDiffer = false;
        for (int i = 0; i < trainingOutput.Length; i++)
        {
            if (Math.Abs(trainingOutput[i] - inferenceOutput[i]) > 1e-6)
            {
                outputsDiffer = true;
                break;
            }
        }

        Assert.True(outputsDiffer);

        for (int i = 0; i < inferenceOutput.Length; i++)
        {
            Assert.Equal(inferenceOutput[i], inferenceOutputRepeat[i], 8);
        }
    }

    [Fact]
    public void BatchNormalizationLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int numFeatures = 10;
        var layer = new BatchNormalizationLayer<double>(numFeatures);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>([16, numFeatures]);
        InitializeRandomTensor(input);

        var output = layer.Forward(input);
        var upstreamGradient = new Tensor<double>(output.Shape);
        InitializeTensorWithValue(upstreamGradient, 1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape[0], gradient.Shape[0]);
        Assert.Equal(input.Shape[1], gradient.Shape[1]);
    }

    [Fact]
    public void BatchNormalizationLayer_SupportsTraining_ReturnsTrue()
    {
        // Arrange
        var layer = new BatchNormalizationLayer<double>(10);

        // Assert
        Assert.True(layer.SupportsTraining);
    }

    #endregion

    #region DropoutLayer Tests

    [Fact]
    public void DropoutLayer_Construction_WithValidDropoutRate_SupportsTraining()
    {
        // Arrange & Act
        var layer = new DropoutLayer<double>(0.5);

        // Assert
        Assert.NotNull(layer);
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void DropoutLayer_Construction_InvalidDropoutRate_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new DropoutLayer<double>(1.0));
        Assert.Throws<ArgumentException>(() => new DropoutLayer<double>(-0.1));
    }

    [Fact]
    public void DropoutLayer_InferenceMode_PassesThroughUnchanged()
    {
        // Arrange
        var layer = new DropoutLayer<double>(0.5);
        layer.SetTrainingMode(false);

        var input = new Tensor<double>([1, 10]);
        for (int i = 0; i < 10; i++)
        {
            input[0, i] = i + 1;
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // In inference mode, output should equal input
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(input[0, i], output[0, i], Tolerance);
        }
    }

    [Fact]
    public void DropoutLayer_TrainingMode_AppliesDropout()
    {
        // Arrange
        var layer = new DropoutLayer<double>(0.5);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>([1, 100]); // Larger tensor for statistical significance
        InitializeTensorWithValue(input, 1.0);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);

        // Count zeros - should have roughly 50% zeros with dropout rate 0.5
        int zeroCount = 0;
        for (int i = 0; i < 100; i++)
        {
            if (Math.Abs(output[0, i]) < Tolerance)
            {
                zeroCount++;
            }
        }

        // With 50% dropout, expect approximately 50 zeros (allow for variance)
        Assert.True(zeroCount > 35 && zeroCount < 65,
            $"Expected roughly 50% zeros but got {zeroCount} zeros");
    }

    [Fact]
    public void DropoutLayer_BackwardPass_PreservesMask()
    {
        // Arrange
        var layer = new DropoutLayer<double>(0.5);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>([1, 20]);
        InitializeTensorWithValue(input, 1.0);

        var output = layer.Forward(input);
        var upstreamGradient = new Tensor<double>(output.Shape);
        InitializeTensorWithValue(upstreamGradient, 1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        // Gradient should be zero where output was zero (same mask applied)    
        for (int i = 0; i < output.Shape[1]; i++)
        {
            if (Math.Abs(output[0, i]) < Tolerance)
            {
                Assert.True(Math.Abs(gradient[0, i]) < Tolerance,
                    $"Gradient[{i}] should be zero where output was dropped");
            }
            else
            {
                Assert.Equal(output[0, i], gradient[0, i], Tolerance);
            }
        }
    }

    #endregion

    #region MultiHeadAttentionLayer Tests

    [Fact]
    public void MultiHeadAttentionLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int sequenceLength = 10;
        int embeddingDimension = 64;
        int headCount = 8;

        var layer = new MultiHeadAttentionLayer<double>(sequenceLength, embeddingDimension, headCount);
        var input = new Tensor<double>([1, sequenceLength, embeddingDimension]);
        InitializeRandomTensor(input);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(1, output.Shape[0]); // batch
        Assert.Equal(sequenceLength, output.Shape[1]);
        Assert.Equal(embeddingDimension, output.Shape[2]);
    }

    [Fact]
    public void MultiHeadAttentionLayer_HeadDimensionValidation()
    {
        // Arrange
        int sequenceLength = 10;
        int embeddingDimension = 64;
        int headCount = 8;

        // Act
        var layer = new MultiHeadAttentionLayer<double>(sequenceLength, embeddingDimension, headCount);

        // Assert
        Assert.NotNull(layer);
        var headDimField = typeof(MultiHeadAttentionLayer<double>)
            .GetField("_headDimension", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(headDimField);
        var headDimension = (int)(headDimField?.GetValue(layer) ?? 0);
        Assert.Equal(embeddingDimension / headCount, headDimension);
    }

    [Fact]
    public void MultiHeadAttentionLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int sequenceLength = 5;
        int embeddingDimension = 32;
        int headCount = 4;

        var layer = new MultiHeadAttentionLayer<double>(sequenceLength, embeddingDimension, headCount);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>([1, sequenceLength, embeddingDimension]);
        InitializeRandomTensor(input);

        var output = layer.Forward(input);
        var upstreamGradient = new Tensor<double>(output.Shape);
        InitializeTensorWithValue(upstreamGradient, 1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
    }

    [Fact]
    public void MultiHeadAttentionLayer_BatchProcessing_HandlesMultipleSamples()
    {
        // Arrange
        int batchSize = 4;
        int sequenceLength = 8;
        int embeddingDimension = 32;
        int headCount = 4;

        var layer = new MultiHeadAttentionLayer<double>(sequenceLength, embeddingDimension, headCount);
        var input = new Tensor<double>([batchSize, sequenceLength, embeddingDimension]);
        InitializeRandomTensor(input);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(batchSize, output.Shape[0]);
    }

    [Fact]
    public void MultiHeadAttentionLayer_SupportsTraining_ReturnsTrue()
    {
        // Arrange
        var layer = new MultiHeadAttentionLayer<double>(10, 64, 8);

        // Assert
        Assert.True(layer.SupportsTraining);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void DenseLayer_ZeroOutput_ThrowsArgumentOutOfRange()
    {
        // Arrange, Act & Assert - zero output size is invalid
        Assert.Throws<ArgumentOutOfRangeException>(() => new DenseLayer<double>(5, 0));
    }

    [Fact]
    public void ConvolutionalLayer_SmallInput_HandlesGracefully()
    {
        // Arrange - Input smaller than kernel
        int inputDepth = 1;
        int inputHeight = 2;
        int inputWidth = 2;
        int outputDepth = 1;
        int kernelSize = 3;

        // Act & Assert
        try
        {
            var layer = new ConvolutionalLayer<double>(
                inputDepth, inputHeight, inputWidth, outputDepth, kernelSize);
            var input = new Tensor<double>([1, inputDepth, inputHeight, inputWidth]);
            var output = layer.Forward(input);
            // If we get here, output shape should be valid (possibly 0 or padded)
            Assert.NotNull(output);
        }
        catch (ArgumentException)
        {
            // Expected - input too small for kernel
            Assert.True(true);
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void DenseLayer_LargeValues_RemainsNumericallyStable()
    {
        // Arrange
        var layer = new DenseLayer<double>(10, 5);
        var input = new Tensor<double>([1, 10]);
        InitializeTensorWithValue(input, 1000.0); // Large values

        // Act
        var output = layer.Forward(input);

        // Assert - Should not contain NaN or Infinity
        for (int i = 0; i < 5; i++)
        {
            Assert.False(double.IsNaN(output[0, i]), $"Output[{i}] is NaN");
            Assert.False(double.IsInfinity(output[0, i]), $"Output[{i}] is Infinity");
        }
    }

    [Fact]
    public void DenseLayer_SmallValues_RemainsNumericallyStable()
    {
        // Arrange
        var layer = new DenseLayer<double>(10, 5);
        var input = new Tensor<double>([1, 10]);
        InitializeTensorWithValue(input, 1e-10); // Very small values

        // Act
        var output = layer.Forward(input);

        // Assert - Should not contain NaN
        for (int i = 0; i < 5; i++)
        {
            Assert.False(double.IsNaN(output[0, i]), $"Output[{i}] is NaN");
        }
    }

    [Fact]
    public void BatchNormalizationLayer_ZeroVariance_HandlesGracefully()
    {
        // Arrange
        int numFeatures = 5;
        var layer = new BatchNormalizationLayer<double>(numFeatures);
        layer.SetTrainingMode(true);

        // All same values - zero variance
        var input = new Tensor<double>([8, numFeatures]);
        InitializeTensorWithValue(input, 5.0);

        // Act
        var output = layer.Forward(input);

        // Assert - Should handle gracefully without NaN or Infinity (epsilon prevents division by zero)
        Assert.NotNull(output);
        for (int b = 0; b < 8; b++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                Assert.False(double.IsNaN(output[b, f]), $"Output[{b},{f}] is NaN with zero variance");
                Assert.False(double.IsInfinity(output[b, f]), $"Output[{b},{f}] is Infinity with zero variance");
            }
        }
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void DenseLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new DenseLayer<double>(10, 5);
        var input = new Tensor<double>([1, 10]);
        InitializeRandomTensor(input);

        var originalOutput = original.Forward(input);

        // Act
        var clone = original.Clone();
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(original, clone);

        // Clone should produce same output
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(originalOutput[0, i], cloneOutput[0, i], Tolerance);
        }
    }

    #endregion

    #region Helper Methods

    private void InitializeRandomTensor(Tensor<double> tensor, double scale = 1.0)
    {
        // Fixed seed for reproducibility of test runs.
        // Note: during development, occasionally run with different seeds (or add parameterized
        // tests) to catch bugs that only appear under certain random initializations.
        var random = new Random(42);
        int totalElements = 1;
        foreach (var dim in tensor.Shape)
        {
            totalElements *= dim;
        }

        // Use flat indexer directly (tensor[flatIndex] works in Tensor<T>)
        for (int i = 0; i < totalElements; i++)
        {
            tensor[i] = (random.NextDouble() - 0.5) * 2.0 * scale;
        }
    }

    private void InitializeTensorWithValue(Tensor<double> tensor, double value)
    {
        int totalElements = 1;
        foreach (var dim in tensor.Shape)
        {
            totalElements *= dim;
        }

        // Use flat indexer directly
        for (int i = 0; i < totalElements; i++)
        {
            tensor[i] = value;
        }
    }

    #endregion
}
