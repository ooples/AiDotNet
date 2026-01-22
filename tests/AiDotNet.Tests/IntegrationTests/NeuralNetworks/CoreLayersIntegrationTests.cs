using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for core neural network layers.
/// Tests DenseLayer, ConvolutionalLayer, MaxPoolingLayer, AveragePoolingLayer,
/// DropoutLayer, FlattenLayer, ReshapeLayer, and ActivationLayer.
/// </summary>
public class CoreLayersIntegrationTests
{
    private const float Tolerance = 1e-5f;

    #region Helper Methods

    /// <summary>
    /// Creates random input tensor with specified shape.
    /// </summary>
    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var random = new Random(seed);
        var length = 1;
        foreach (var dim in shape) length *= dim;
        var flatData = new float[length];
        for (int i = 0; i < flatData.Length; i++)
        {
            flatData[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return new Tensor<float>(flatData, shape);
    }

    /// <summary>
    /// Creates 2D input tensor [batch, features].
    /// </summary>
    private static Tensor<float> Create2DInput(int batchSize, int features, int seed = 42)
    {
        return CreateRandomTensor([batchSize, features], seed);
    }

    /// <summary>
    /// Creates 4D input tensor [batch, channels, height, width].
    /// </summary>
    private static Tensor<float> Create4DInput(int batchSize, int channels, int height, int width, int seed = 42)
    {
        return CreateRandomTensor([batchSize, channels, height, width], seed);
    }

    #endregion

    #region DenseLayer Tests

    [Fact]
    public void DenseLayer_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new DenseLayer<float>(inputSize, outputSize);
        var input = Create2DInput(4, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(4, output.Shape[0]); // batch size
        Assert.Equal(outputSize, output.Shape[1]); // output features
    }

    [Fact]
    public void DenseLayer_Forward_1DInput_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var layer = new DenseLayer<float>(inputSize, outputSize);
        var input = CreateRandomTensor([inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Single(output.Shape);
        Assert.Equal(outputSize, output.Shape[0]);
    }

    [Fact]
    public void DenseLayer_Forward_3DInput_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var layer = new DenseLayer<float>(inputSize, outputSize);
        var input = CreateRandomTensor([2, 5, inputSize]); // [batch, sequence, features]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(3, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(5, output.Shape[1]); // sequence
        Assert.Equal(outputSize, output.Shape[2]); // output features
    }

    [Fact]
    public void DenseLayer_Backward_ProducesGradientWithCorrectShape()
    {
        // Arrange
        int inputSize = 32;
        int outputSize = 16;
        var layer = new DenseLayer<float>(inputSize, outputSize);
        var input = Create2DInput(4, inputSize);
        var outputGrad = CreateRandomTensor([4, outputSize], 123);

        // Act
        var output = layer.Forward(input);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void DenseLayer_UpdateParameters_ModifiesWeights()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var layer = new DenseLayer<float>(inputSize, outputSize);
        var input = Create2DInput(2, inputSize);
        var outputGrad = CreateRandomTensor([2, outputSize], 123);

        // Get initial parameters
        var initialParams = layer.GetParameters();
        var initialParamsArray = new float[initialParams.Length];
        for (int i = 0; i < initialParams.Length; i++)
            initialParamsArray[i] = initialParams[i];

        // Act
        layer.Forward(input);
        layer.Backward(outputGrad);
        layer.UpdateParameters(0.01f);

        // Get updated parameters
        var updatedParams = layer.GetParameters();

        // Assert - parameters should have changed
        bool paramsChanged = false;
        for (int i = 0; i < updatedParams.Length; i++)
        {
            if (Math.Abs(updatedParams[i] - initialParamsArray[i]) > Tolerance)
            {
                paramsChanged = true;
                break;
            }
        }
        Assert.True(paramsChanged, "Parameters should change after update");
    }

    [Fact]
    public void DenseLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var original = new DenseLayer<float>(inputSize, outputSize);
        var input = Create2DInput(2, inputSize);

        // Act
        var clone = (DenseLayer<float>)original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert - outputs should be identical
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    [Fact]
    public void DenseLayer_SetAndGetWeights_WorksCorrectly()
    {
        // Arrange
        int inputSize = 8;
        int outputSize = 4;
        var layer = new DenseLayer<float>(inputSize, outputSize);
        var newWeights = CreateRandomTensor([inputSize, outputSize], 99);

        // Act - Use SetParameter from IWeightLoadable instead of protected SetWeights
        layer.SetParameter("weight", newWeights);
        var retrievedWeights = layer.GetWeights();

        // Assert
        Assert.Equal(newWeights.Shape, retrievedWeights.Shape);
        for (int i = 0; i < newWeights.Length; i++)
        {
            Assert.Equal(newWeights[i], retrievedWeights[i], Tolerance);
        }
    }

    [Fact]
    public void DenseLayer_ParameterCount_ReturnsCorrectValue()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new DenseLayer<float>(inputSize, outputSize);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        // weights: inputSize * outputSize + biases: outputSize
        int expected = inputSize * outputSize + outputSize;
        Assert.Equal(expected, paramCount);
    }

    [Fact]
    public void DenseLayer_WithDifferentActivations_ProducesDifferentOutputs()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var reluLayer = new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());
        var sigmoidLayer = new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new SigmoidActivation<float>());
        var input = Create2DInput(2, inputSize);

        // Copy weights from relu to sigmoid for fair comparison
        sigmoidLayer.SetParameters(reluLayer.GetParameters());

        // Act
        var reluOutput = reluLayer.Forward(input);
        var sigmoidOutput = sigmoidLayer.Forward(input);

        // Assert - outputs should be different due to different activations
        bool outputsDiffer = false;
        for (int i = 0; i < reluOutput.Length; i++)
        {
            if (Math.Abs(reluOutput[i] - sigmoidOutput[i]) > Tolerance)
            {
                outputsDiffer = true;
                break;
            }
        }
        Assert.True(outputsDiffer, "Different activations should produce different outputs");
    }

    [Fact]
    public void DenseLayer_L2Regularization_ComputesAuxiliaryLoss()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var layer = new DenseLayer<float>(inputSize, outputSize)
        {
            UseAuxiliaryLoss = true,
            Regularization = RegularizationType.L2,
            L2Strength = 0.01f
        };

        // Act
        var loss = layer.ComputeAuxiliaryLoss();

        // Assert - loss should be positive (sum of squared weights)
        Assert.True(loss > 0, "L2 regularization loss should be positive");
    }

    #endregion

    #region ConvolutionalLayer Tests

    [Fact]
    public void ConvolutionalLayer_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        int inputChannels = 3;
        int outputChannels = 16;
        int kernelSize = 3;
        int stride = 1;
        int padding = 1;
        int inputHeight = 28;
        int inputWidth = 28;

        var layer = new ConvolutionalLayer<float>(
            inputChannels, inputHeight, inputWidth,
            outputChannels, kernelSize, stride, padding);
        var input = Create4DInput(2, inputChannels, inputHeight, inputWidth);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(4, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(outputChannels, output.Shape[1]); // channels
        // With padding=1, stride=1, kernel=3: outputSize = inputSize
        Assert.Equal(inputHeight, output.Shape[2]); // height
        Assert.Equal(inputWidth, output.Shape[3]); // width
    }

    [Fact]
    public void ConvolutionalLayer_Backward_ProducesGradientWithCorrectShape()
    {
        // Arrange
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        int inputChannels = 3;
        int outputChannels = 8;
        int kernelSize = 3;
        int stride = 1;
        int padding = 1;
        int inputHeight = 16;
        int inputWidth = 16;

        var layer = new ConvolutionalLayer<float>(
            inputChannels, inputHeight, inputWidth,
            outputChannels, kernelSize, stride, padding);
        var input = Create4DInput(2, inputChannels, inputHeight, inputWidth);

        // Act
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape, 123);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void ConvolutionalLayer_WithStride2_ReducesSpatialDimensions()
    {
        // Arrange
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        int inputChannels = 1;
        int outputChannels = 4;
        int kernelSize = 3;
        int stride = 2;
        int padding = 1;
        int inputHeight = 16;
        int inputWidth = 16;

        var layer = new ConvolutionalLayer<float>(
            inputChannels, inputHeight, inputWidth,
            outputChannels, kernelSize, stride, padding);
        var input = Create4DInput(1, inputChannels, inputHeight, inputWidth);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(4, output.Rank);
        Assert.Equal(outputChannels, output.Shape[1]);
        // With stride=2: outputSize = (inputSize + 2*padding - kernelSize) / stride + 1 = (16+2-3)/2+1 = 8
        Assert.Equal(8, output.Shape[2]);
        Assert.Equal(8, output.Shape[3]);
    }

    [Fact]
    public void ConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        var layer = new ConvolutionalLayer<float>(1, 8, 8, 4, 3, 1, 1);
        var input = Create4DInput(1, 1, 8, 8);

        // Act
        var clone = (ConvolutionalLayer<float>)layer.Clone();
        var originalOutput = layer.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    [Fact]
    public void ConvolutionalLayer_ParameterCount_ReturnsCorrectValue()
    {
        // Arrange
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        int inputChannels = 3;
        int outputChannels = 16;
        int kernelSize = 3;
        var layer = new ConvolutionalLayer<float>(inputChannels, 28, 28, outputChannels, kernelSize, 1, 1);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        // kernels: outputChannels * inputChannels * kernelSize * kernelSize + biases: outputChannels
        int expected = outputChannels * inputChannels * kernelSize * kernelSize + outputChannels;
        Assert.Equal(expected, paramCount);
    }

    #endregion

    #region MaxPoolingLayer Tests

    [Fact]
    public void MaxPoolingLayer_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        int channels = 16;
        int inputHeight = 16;
        int inputWidth = 16;
        int poolSize = 2;
        int stride = 2;

        var layer = new MaxPoolingLayer<float>([channels, inputHeight, inputWidth], poolSize, stride);
        var input = Create4DInput(2, channels, inputHeight, inputWidth);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(4, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(channels, output.Shape[1]); // channels unchanged
        Assert.Equal(inputHeight / poolSize, output.Shape[2]); // height halved
        Assert.Equal(inputWidth / poolSize, output.Shape[3]); // width halved
    }

    [Fact]
    public void MaxPoolingLayer_Forward_PreservesMaxValues()
    {
        // Arrange
        int channels = 1;
        int size = 4;
        var layer = new MaxPoolingLayer<float>([channels, size, size], 2, 2);

        // Create input with known values
        var input = new Tensor<float>([1, 1, 4, 4]);
        // Set up a 4x4 grid where max in each 2x2 is in different positions
        input[0, 0, 0, 0] = 1f; input[0, 0, 0, 1] = 2f; input[0, 0, 0, 2] = 3f; input[0, 0, 0, 3] = 8f;
        input[0, 0, 1, 0] = 4f; input[0, 0, 1, 1] = 3f; input[0, 0, 1, 2] = 7f; input[0, 0, 1, 3] = 6f;
        input[0, 0, 2, 0] = 5f; input[0, 0, 2, 1] = 6f; input[0, 0, 2, 2] = 1f; input[0, 0, 2, 3] = 2f;
        input[0, 0, 3, 0] = 9f; input[0, 0, 3, 1] = 8f; input[0, 0, 3, 2] = 4f; input[0, 0, 3, 3] = 5f;

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Shape[2]);
        Assert.Equal(2, output.Shape[3]);
        Assert.Equal(4f, output[0, 0, 0, 0]); // max of [1,2,4,3]
        Assert.Equal(8f, output[0, 0, 0, 1]); // max of [3,8,7,6]
        Assert.Equal(9f, output[0, 0, 1, 0]); // max of [5,6,9,8]
        Assert.Equal(5f, output[0, 0, 1, 1]); // max of [1,2,4,5]
    }

    [Fact]
    public void MaxPoolingLayer_Backward_ProducesGradientWithCorrectShape()
    {
        // Arrange
        int channels = 8;
        int inputHeight = 16;
        int inputWidth = 16;
        var layer = new MaxPoolingLayer<float>([channels, inputHeight, inputWidth], 2, 2);
        var input = Create4DInput(2, channels, inputHeight, inputWidth);

        // Act
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape, 123);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void MaxPoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var layer = new MaxPoolingLayer<float>([4, 8, 8], 2, 2);
        var input = Create4DInput(1, 4, 8, 8);

        // Act
        var clone = (MaxPoolingLayer<float>)layer.Clone();
        var originalOutput = layer.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region AveragePoolingLayer Tests

    [Fact]
    public void AveragePoolingLayer_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        int channels = 16;
        int inputHeight = 16;
        int inputWidth = 16;
        int poolSize = 2;
        int stride = 2;

        var layer = new AveragePoolingLayer<float>([channels, inputHeight, inputWidth], poolSize, stride);
        var input = Create4DInput(2, channels, inputHeight, inputWidth);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(4, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(channels, output.Shape[1]); // channels unchanged
        Assert.Equal(inputHeight / poolSize, output.Shape[2]); // height halved
        Assert.Equal(inputWidth / poolSize, output.Shape[3]); // width halved
    }

    [Fact]
    public void AveragePoolingLayer_Forward_ComputesCorrectAverages()
    {
        // Arrange
        int channels = 1;
        int size = 4;
        var layer = new AveragePoolingLayer<float>([channels, size, size], 2, 2);

        // Create input with known values
        var input = new Tensor<float>([1, 1, 4, 4]);
        // Set up a 4x4 grid
        input[0, 0, 0, 0] = 1f; input[0, 0, 0, 1] = 2f; input[0, 0, 0, 2] = 3f; input[0, 0, 0, 3] = 4f;
        input[0, 0, 1, 0] = 5f; input[0, 0, 1, 1] = 6f; input[0, 0, 1, 2] = 7f; input[0, 0, 1, 3] = 8f;
        input[0, 0, 2, 0] = 9f; input[0, 0, 2, 1] = 10f; input[0, 0, 2, 2] = 11f; input[0, 0, 2, 3] = 12f;
        input[0, 0, 3, 0] = 13f; input[0, 0, 3, 1] = 14f; input[0, 0, 3, 2] = 15f; input[0, 0, 3, 3] = 16f;

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Shape[2]);
        Assert.Equal(2, output.Shape[3]);
        Assert.Equal((1f + 2f + 5f + 6f) / 4f, output[0, 0, 0, 0], Tolerance); // avg = 3.5
        Assert.Equal((3f + 4f + 7f + 8f) / 4f, output[0, 0, 0, 1], Tolerance); // avg = 5.5
        Assert.Equal((9f + 10f + 13f + 14f) / 4f, output[0, 0, 1, 0], Tolerance); // avg = 11.5
        Assert.Equal((11f + 12f + 15f + 16f) / 4f, output[0, 0, 1, 1], Tolerance); // avg = 13.5
    }

    [Fact]
    public void AveragePoolingLayer_Backward_ProducesGradientWithCorrectShape()
    {
        // Arrange
        int channels = 8;
        int inputHeight = 16;
        int inputWidth = 16;
        var layer = new AveragePoolingLayer<float>([channels, inputHeight, inputWidth], 2, 2);
        var input = Create4DInput(2, channels, inputHeight, inputWidth);

        // Act
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape, 123);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    #endregion

    #region DropoutLayer Tests

    [Fact]
    public void DropoutLayer_Forward_TrainingMode_AppliesDropout()
    {
        // Arrange
        int[] shape = [8, 16];
        float dropoutRate = 0.5f;
        var layer = new DropoutLayer<float>(dropoutRate);
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor(shape);

        // Act
        var output = layer.Forward(input);

        // Assert - some values should be zero due to dropout
        int zeroCount = 0;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output[i]) < Tolerance)
                zeroCount++;
        }
        // With 50% dropout, expect roughly 50% zeros
        Assert.True(zeroCount > output.Length * 0.2, "Dropout should zero some values");
        Assert.True(zeroCount < output.Length * 0.8, "Dropout should not zero all values");
    }

    [Fact]
    public void DropoutLayer_Forward_InferenceMode_PassesThroughUnchanged()
    {
        // Arrange
        int[] shape = [8, 16];
        float dropoutRate = 0.5f;
        var layer = new DropoutLayer<float>(dropoutRate);
        layer.SetTrainingMode(false);
        var input = CreateRandomTensor(shape);

        // Act
        var output = layer.Forward(input);

        // Assert - output should equal input in inference mode
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], output[i], Tolerance);
        }
    }

    [Fact]
    public void DropoutLayer_Backward_ProducesGradientWithCorrectShape()
    {
        // Arrange
        int[] shape = [4, 32];
        var layer = new DropoutLayer<float>(0.3);
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor(shape);

        // Act
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(shape, 123);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void DropoutLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] shape = [4, 16];
        var original = new DropoutLayer<float>(0.3);
        original.SetTrainingMode(false);
        var input = CreateRandomTensor(shape);

        // Act
        var clone = (DropoutLayer<float>)original.Clone();
        clone.SetTrainingMode(false);
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert - in inference mode, outputs should match
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region FlattenLayer Tests

    [Fact]
    public void FlattenLayer_Forward_Flattens3DTo1D()
    {
        // Arrange
        int[] inputShape = [8, 8, 3];
        var layer = new FlattenLayer<float>(inputShape);
        var input = CreateRandomTensor([2, 8, 8, 3]); // [batch, height, width, channels]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(8 * 8 * 3, output.Shape[1]); // flattened
    }

    [Fact]
    public void FlattenLayer_Forward_PreservesData()
    {
        // Arrange
        int[] inputShape = [4, 4, 2];
        var layer = new FlattenLayer<float>(inputShape);
        var input = CreateRandomTensor([1, 4, 4, 2]);

        // Act
        var output = layer.Forward(input);

        // Assert - all data should be preserved
        Assert.Equal(input.Length, output.Length);
        float sum1 = 0, sum2 = 0;
        for (int i = 0; i < input.Length; i++)
        {
            sum1 += input[i];
            sum2 += output[i];
        }
        Assert.Equal(sum1, sum2, Tolerance);
    }

    [Fact]
    public void FlattenLayer_Backward_RestoresOriginalShape()
    {
        // Arrange
        int[] inputShape = [8, 8, 3];
        var layer = new FlattenLayer<float>(inputShape);
        var input = CreateRandomTensor([2, 8, 8, 3]);

        // Act
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape, 123);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void FlattenLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [4, 4, 2];
        var original = new FlattenLayer<float>(inputShape);
        var input = CreateRandomTensor([1, 4, 4, 2]);

        // Act
        var clone = (FlattenLayer<float>)original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region ReshapeLayer Tests

    [Fact]
    public void ReshapeLayer_Forward_ReshapesCorrectly()
    {
        // Arrange
        int[] inputShape = [32];
        int[] outputShape = [4, 8];
        var layer = new ReshapeLayer<float>(inputShape, outputShape);
        var input = CreateRandomTensor([2, 32]); // [batch, features]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(3, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(4, output.Shape[1]);
        Assert.Equal(8, output.Shape[2]);
    }

    [Fact]
    public void ReshapeLayer_Forward_PreservesData()
    {
        // Arrange
        int[] inputShape = [24];
        int[] outputShape = [4, 6];
        var layer = new ReshapeLayer<float>(inputShape, outputShape);
        var input = CreateRandomTensor([1, 24]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Length, output.Length);
        float sum1 = 0, sum2 = 0;
        for (int i = 0; i < input.Length; i++)
        {
            sum1 += input[i];
            sum2 += output[i];
        }
        Assert.Equal(sum1, sum2, Tolerance);
    }

    [Fact]
    public void ReshapeLayer_Backward_RestoresOriginalShape()
    {
        // Arrange
        int[] inputShape = [32];
        int[] outputShape = [4, 8];
        var layer = new ReshapeLayer<float>(inputShape, outputShape);
        var input = CreateRandomTensor([2, 32]);

        // Act
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape, 123);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void ReshapeLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [16];
        int[] outputShape = [4, 4];
        var original = new ReshapeLayer<float>(inputShape, outputShape);
        var input = CreateRandomTensor([1, 16]);

        // Act
        var clone = (ReshapeLayer<float>)original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region ActivationLayer Tests

    [Fact]
    public void ActivationLayer_ReLU_Forward_ZerosNegativeValues()
    {
        // Arrange
        int[] shape = [8, 16];
        var layer = new ActivationLayer<float>(shape, (IActivationFunction<float>)new ReLUActivation<float>());

        // Create input with positive and negative values
        var input = new Tensor<float>([2, 8, 16]);
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(random.NextDouble() * 2 - 1); // [-1, 1]
        }

        // Act
        var output = layer.Forward(input);

        // Assert - negative values should be zeroed
        for (int i = 0; i < output.Length; i++)
        {
            if (input[i] < 0)
            {
                Assert.Equal(0f, output[i], Tolerance);
            }
            else
            {
                Assert.Equal(input[i], output[i], Tolerance);
            }
        }
    }

    [Fact]
    public void ActivationLayer_Sigmoid_Forward_OutputsInRange01()
    {
        // Arrange
        int[] shape = [8, 16];
        var layer = new ActivationLayer<float>(shape, (IActivationFunction<float>)new SigmoidActivation<float>());
        var input = CreateRandomTensor([2, 8, 16]);

        // Act
        var output = layer.Forward(input);

        // Assert - all values should be in (0, 1)
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] > 0f && output[i] < 1f,
                $"Sigmoid output {output[i]} should be in (0, 1)");
        }
    }

    [Fact]
    public void ActivationLayer_Tanh_Forward_OutputsInRangeMinus1To1()
    {
        // Arrange
        int[] shape = [8, 16];
        var layer = new ActivationLayer<float>(shape, (IActivationFunction<float>)new TanhActivation<float>());
        var input = CreateRandomTensor([2, 8, 16]);

        // Act
        var output = layer.Forward(input);

        // Assert - all values should be in (-1, 1)
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] > -1f && output[i] < 1f,
                $"Tanh output {output[i]} should be in (-1, 1)");
        }
    }

    [Fact]
    public void ActivationLayer_Backward_ProducesGradientWithCorrectShape()
    {
        // Arrange
        int[] shape = [8, 16];
        var layer = new ActivationLayer<float>(shape, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = CreateRandomTensor([2, 8, 16]);

        // Act
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape, 123);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void ActivationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] shape = [4, 8];
        var original = new ActivationLayer<float>(shape, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = CreateRandomTensor([1, 4, 8]);

        // Act
        var clone = (ActivationLayer<float>)original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region Layer Integration Tests

    [Fact]
    public void DenseLayerChain_ForwardAndBackward_WorksCorrectly()
    {
        // Arrange - a simple network: Dense(64) -> Dense(32) -> Dense(10)
        var layer1 = new DenseLayer<float>(64, 32);
        var layer2 = new DenseLayer<float>(32, 16);
        var layer3 = new DenseLayer<float>(16, 10);

        var input = Create2DInput(4, 64);
        var targetGrad = CreateRandomTensor([4, 10], 123);

        // Act - Forward
        var out1 = layer1.Forward(input);
        var out2 = layer2.Forward(out1);
        var out3 = layer3.Forward(out2);

        // Backward
        var grad3 = layer3.Backward(targetGrad);
        var grad2 = layer2.Backward(grad3);
        var grad1 = layer1.Backward(grad2);

        // Assert
        Assert.Equal(input.Shape, grad1.Shape);
        Assert.Equal(out1.Shape, grad2.Shape);
        Assert.Equal(out2.Shape, grad3.Shape);
    }

    [Fact]
    public void ConvFlattenDenseChain_ForwardAndBackward_WorksCorrectly()
    {
        // Arrange - Conv -> Flatten -> Dense
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        var convLayer = new ConvolutionalLayer<float>(1, 8, 8, 4, 3, 1, 1);
        var flattenLayer = new FlattenLayer<float>([4, 8, 8]);
        var denseLayer = new DenseLayer<float>(4 * 8 * 8, 10);

        var input = Create4DInput(2, 1, 8, 8);
        var targetGrad = CreateRandomTensor([2, 10], 123);

        // Act - Forward
        var convOut = convLayer.Forward(input);
        var flatOut = flattenLayer.Forward(convOut);
        var denseOut = denseLayer.Forward(flatOut);

        // Backward
        var denseGrad = denseLayer.Backward(targetGrad);
        var flatGrad = flattenLayer.Backward(denseGrad);
        var convGrad = convLayer.Backward(flatGrad);

        // Assert
        Assert.Equal([2, 10], denseOut.Shape);
        Assert.Equal(input.Shape, convGrad.Shape);
    }

    [Fact]
    public void ConvPoolChain_ForwardAndBackward_WorksCorrectly()
    {
        // Arrange - Conv -> MaxPool -> Conv -> MaxPool
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        var conv1 = new ConvolutionalLayer<float>(1, 16, 16, 8, 3, 1, 1);
        var pool1 = new MaxPoolingLayer<float>([8, 16, 16], 2, 2);
        var conv2 = new ConvolutionalLayer<float>(8, 8, 8, 16, 3, 1, 1);
        var pool2 = new MaxPoolingLayer<float>([16, 8, 8], 2, 2);

        var input = Create4DInput(2, 1, 16, 16);
        var targetGrad = CreateRandomTensor([2, 16, 4, 4], 123);

        // Act - Forward
        var c1Out = conv1.Forward(input);
        var p1Out = pool1.Forward(c1Out);
        var c2Out = conv2.Forward(p1Out);
        var p2Out = pool2.Forward(c2Out);

        // Backward
        var p2Grad = pool2.Backward(targetGrad);
        var c2Grad = conv2.Backward(p2Grad);
        var p1Grad = pool1.Backward(c2Grad);
        var c1Grad = conv1.Backward(p1Grad);

        // Assert
        Assert.Equal([2, 16, 4, 4], p2Out.Shape);
        Assert.Equal(input.Shape, c1Grad.Shape);
    }

    [Fact]
    public void DenseWithDropout_TrainingVsInference_BehavesDifferently()
    {
        // Arrange
        var dense = new DenseLayer<float>(32, 16);
        var dropout = new DropoutLayer<float>(0.5);
        var input = Create2DInput(4, 32);

        // Act - Training mode
        dropout.SetTrainingMode(true);
        var denseOut = dense.Forward(input);
        var trainOutput = dropout.Forward(denseOut);

        // Inference mode
        dropout.SetTrainingMode(false);
        var inferenceOutput = dropout.Forward(denseOut);

        // Assert - training output should have some zeros, inference should not
        int trainZeros = 0;
        int inferenceZeros = 0;
        for (int i = 0; i < trainOutput.Length; i++)
        {
            if (Math.Abs(trainOutput[i]) < Tolerance) trainZeros++;
            if (Math.Abs(inferenceOutput[i]) < Tolerance) inferenceZeros++;
        }
        Assert.True(trainZeros > 0, "Training should have some dropouts");
        Assert.True(trainZeros > inferenceZeros, "Training should have more zeros than inference");
    }

    #endregion
}
