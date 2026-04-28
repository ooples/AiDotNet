using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new DenseLayer<float>(outputSize);
        var input = Create2DInput(4, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(4, output.Shape[0]); // batch size
        Assert.Equal(outputSize, output.Shape[1]); // output features
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_Forward_1DInput_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var layer = new DenseLayer<float>(outputSize);
        var input = CreateRandomTensor([inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Single(output.Shape.ToArray());
        Assert.Equal(outputSize, output.Shape[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_Forward_3DInput_ProducesCorrectOutputShape()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var layer = new DenseLayer<float>(outputSize);
        var input = CreateRandomTensor([2, 5, inputSize]); // [batch, sequence, features]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(3, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(5, output.Shape[1]); // sequence
        Assert.Equal(outputSize, output.Shape[2]); // output features
    }



    [Fact(Timeout = 120000)]
    public async Task DenseLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var original = new DenseLayer<float>(outputSize);
        var input = Create2DInput(2, inputSize);

        // Act
        var clone = (DenseLayer<float>)original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert - outputs should be identical
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_SetAndGetWeights_WorksCorrectly()
    {
        // Arrange
        int inputSize = 8;
        int outputSize = 4;
        var layer = new DenseLayer<float>(outputSize);
        var newWeights = CreateRandomTensor([inputSize, outputSize], 99);

        // Act - Use SetParameter from IWeightLoadable instead of protected SetWeights
        layer.SetParameter("weight", newWeights);
        var retrievedWeights = layer.GetWeights();

        // Assert
        Assert.Equal(newWeights.Shape.ToArray(), retrievedWeights.Shape.ToArray());
        for (int i = 0; i < newWeights.Length; i++)
        {
            Assert.Equal(newWeights[i], retrievedWeights[i], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_ParameterCount_ReturnsCorrectValue()
    {
        // Arrange
        int inputSize = 64;
        int outputSize = 32;
        var layer = new DenseLayer<float>(outputSize);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        // weights: inputSize * outputSize + biases: outputSize
        int expected = inputSize * outputSize + outputSize;
        Assert.Equal(expected, paramCount);
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_WithDifferentActivations_ProducesDifferentOutputs()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var reluLayer = new DenseLayer<float>(outputSize, (IActivationFunction<float>)new ReLUActivation<float>());
        var sigmoidLayer = new DenseLayer<float>(outputSize, (IActivationFunction<float>)new SigmoidActivation<float>());
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

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_L2Regularization_ComputesAuxiliaryLoss()
    {
        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        var layer = new DenseLayer<float>(outputSize)
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

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_Forward_ProducesCorrectOutputShape()
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


    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_WithStride2_ReducesSpatialDimensions()
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

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        var layer = new ConvolutionalLayer<float>(4, 3, 1, 1);
        var input = Create4DInput(1, 1, 8, 8);

        // Act
        var clone = (ConvolutionalLayer<float>)layer.Clone();
        var originalOutput = layer.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ConvolutionalLayer_ParameterCount_ReturnsCorrectValue()
    {
        // Arrange
        // Constructor: (inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        int inputChannels = 3;
        int outputChannels = 16;
        int kernelSize = 3;
        var layer = new ConvolutionalLayer<float>(outputChannels, kernelSize, 1, 1);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        // kernels: outputChannels * inputChannels * kernelSize * kernelSize + biases: outputChannels
        int expected = outputChannels * inputChannels * kernelSize * kernelSize + outputChannels;
        Assert.Equal(expected, paramCount);
    }

    #endregion

    #region MaxPoolingLayer Tests

    [Fact(Timeout = 120000)]
    public async Task MaxPoolingLayer_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        int channels = 16;
        int inputHeight = 16;
        int inputWidth = 16;
        int poolSize = 2;
        int stride = 2;

        var layer = new MaxPoolingLayer<float>(poolSize, stride);
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

    [Fact(Timeout = 120000)]
    public async Task MaxPoolingLayer_Forward_PreservesMaxValues()
    {
        // Arrange
        int channels = 1;
        int size = 4;
        var layer = new MaxPoolingLayer<float>(2, 2);

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


    [Fact(Timeout = 120000)]
    public async Task MaxPoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var layer = new MaxPoolingLayer<float>(2, 2);
        var input = Create4DInput(1, 4, 8, 8);

        // Act
        var clone = (MaxPoolingLayer<float>)layer.Clone();
        var originalOutput = layer.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region AveragePoolingLayer Tests

    [Fact(Timeout = 120000)]
    public async Task AveragePoolingLayer_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        int channels = 16;
        int inputHeight = 16;
        int inputWidth = 16;
        int poolSize = 2;
        int stride = 2;

        var layer = new AveragePoolingLayer<float>(poolSize, stride);
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

    [Fact(Timeout = 120000)]
    public async Task AveragePoolingLayer_Forward_ComputesCorrectAverages()
    {
        // Arrange
        int channels = 1;
        int size = 4;
        var layer = new AveragePoolingLayer<float>(2, 2);

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


    #endregion

    #region DropoutLayer Tests

    [Fact(Timeout = 120000)]
    public async Task DropoutLayer_Forward_TrainingMode_AppliesDropout()
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

    [Fact(Timeout = 120000)]
    public async Task DropoutLayer_Forward_InferenceMode_PassesThroughUnchanged()
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


    [Fact(Timeout = 120000)]
    public async Task DropoutLayer_Clone_CreatesIndependentCopy()
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
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region FlattenLayer Tests

    [Fact(Timeout = 120000)]
    public async Task FlattenLayer_Forward_Flattens3DTo1D()
    {
        // Arrange
        int[] inputShape = [8, 8, 3];
        var layer = new FlattenLayer<float>();
        var input = CreateRandomTensor([2, 8, 8, 3]); // [batch, height, width, channels]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(8 * 8 * 3, output.Shape[1]); // flattened
    }

    [Fact(Timeout = 120000)]
    public async Task FlattenLayer_Forward_PreservesData()
    {
        // Arrange
        int[] inputShape = [4, 4, 2];
        var layer = new FlattenLayer<float>();
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


    [Fact(Timeout = 120000)]
    public async Task FlattenLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [4, 4, 2];
        var original = new FlattenLayer<float>();
        var input = CreateRandomTensor([1, 4, 4, 2]);

        // Act
        var clone = (FlattenLayer<float>)original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region ReshapeLayer Tests

    [Fact(Timeout = 120000)]
    public async Task ReshapeLayer_Forward_ReshapesCorrectly()
    {
        // Arrange
        int[] inputShape = [32];
        int[] outputShape = [4, 8];
        var layer = new ReshapeLayer<float>(outputShape);
        var input = CreateRandomTensor([2, 32]); // [batch, features]

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(3, output.Rank);
        Assert.Equal(2, output.Shape[0]); // batch
        Assert.Equal(4, output.Shape[1]);
        Assert.Equal(8, output.Shape[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task ReshapeLayer_Forward_PreservesData()
    {
        // Arrange
        int[] inputShape = [24];
        int[] outputShape = [4, 6];
        var layer = new ReshapeLayer<float>(outputShape);
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


    [Fact(Timeout = 120000)]
    public async Task ReshapeLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [16];
        int[] outputShape = [4, 4];
        var original = new ReshapeLayer<float>(outputShape);
        var input = CreateRandomTensor([1, 16]);

        // Act
        var clone = (ReshapeLayer<float>)original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region ActivationLayer Tests

    [Fact(Timeout = 120000)]
    public async Task ActivationLayer_ReLU_Forward_ZerosNegativeValues()
    {
        // Arrange
        int[] shape = [8, 16];
        var layer = new ActivationLayer<float>((IActivationFunction<float>)new ReLUActivation<float>());

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

    [Fact(Timeout = 120000)]
    public async Task ActivationLayer_Sigmoid_Forward_OutputsInRange01()
    {
        // Arrange
        int[] shape = [8, 16];
        var layer = new ActivationLayer<float>((IActivationFunction<float>)new SigmoidActivation<float>());
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

    [Fact(Timeout = 120000)]
    public async Task ActivationLayer_Tanh_Forward_OutputsInRangeMinus1To1()
    {
        // Arrange
        int[] shape = [8, 16];
        var layer = new ActivationLayer<float>((IActivationFunction<float>)new TanhActivation<float>());
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


    [Fact(Timeout = 120000)]
    public async Task ActivationLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] shape = [4, 8];
        var original = new ActivationLayer<float>((IActivationFunction<float>)new ReLUActivation<float>());
        var input = CreateRandomTensor([1, 4, 8]);

        // Act
        var clone = (ActivationLayer<float>)original.Clone();
        var originalOutput = original.Forward(input);
        var cloneOutput = clone.Forward(input);

        // Assert
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region DenseLayer Training vs Inference Mode Tests (PR change coverage)

    /// <summary>
    /// After the PR change, DenseLayer.Forward in training mode always uses
    /// FusedLinear(None) + ApplyActivation separately (even for fused-supported activations).
    /// In inference mode it uses the fused path.  Both paths must produce identical values.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DenseLayer_TrainingVsInferenceMode_ProduceIdenticalOutputValues()
    {
        // Arrange
        int inputSize = 8;
        int outputSize = 4;
        // ReLU is a fused-supported activation — exercises the changed branch
        var layer = new DenseLayer<float>(outputSize, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Create2DInput(2, inputSize, seed: 77);

        // Act
        layer.SetTrainingMode(false);
        var inferenceOutput = layer.Forward(input);

        layer.SetTrainingMode(true);
        var trainingOutput = layer.Forward(input);

        // Assert — correctness: both paths compute the same linear + activation
        Assert.Equal(inferenceOutput.Shape.ToArray(), trainingOutput.Shape.ToArray());
        for (int i = 0; i < inferenceOutput.Length; i++)
        {
            Assert.Equal(inferenceOutput[i], trainingOutput[i], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_InferenceModeWithFusedActivation_ProducesFiniteOutput()
    {
        // Inference mode takes the new fused path — verify output is well-formed
        var layer = new DenseLayer<float>(8, (IActivationFunction<float>)new ReLUActivation<float>());
        layer.SetTrainingMode(false);
        var input = Create2DInput(4, 16, seed: 10);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Rank);
        Assert.Equal(4, output.Shape[0]);
        Assert.Equal(8, output.Shape[1]);
        foreach (var v in output)
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), "Inference output must be finite.");
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_TrainingModeWithFusedActivation_ProducesFiniteOutput()
    {
        // Training mode now always goes through else-branch — verify output shape and finiteness
        var layer = new DenseLayer<float>(8, (IActivationFunction<float>)new ReLUActivation<float>());
        layer.SetTrainingMode(true);
        var input = Create2DInput(4, 16, seed: 20);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Rank);
        Assert.Equal(4, output.Shape[0]);
        Assert.Equal(8, output.Shape[1]);
        foreach (var v in output)
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), "Training output must be finite.");
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_TrainingMode_WithIdentityActivation_ProducesSameOutputAsInference()
    {
        // IdentityActivation has no fused type, so both training and inference go through else-branch.
        // The new Activate(Tensor) override returns the same reference — verify consistent results.
        var layer = new DenseLayer<float>(4, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = Create2DInput(2, 8, seed: 55);

        layer.SetTrainingMode(false);
        var inferenceOutput = layer.Forward(input);

        layer.SetTrainingMode(true);
        var trainingOutput = layer.Forward(input);

        for (int i = 0; i < inferenceOutput.Length; i++)
            Assert.Equal(inferenceOutput[i], trainingOutput[i], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_ReLUActivation_TrainingMode_OutputsAreNonNegative()
    {
        // Regression: after the PR change the else-branch applies ApplyActivation(preActivation).
        // ReLU should still zero out negatives.
        var layer = new DenseLayer<float>(4, (IActivationFunction<float>)new ReLUActivation<float>());
        layer.SetTrainingMode(true);

        // Use a fixed input where some pre-activation outputs will be negative
        var input = new Tensor<float>(new float[] { -3f, -2f, -1f, 0f, 1f, 2f, 3f, 4f }, [1, 8]);
        var output = layer.Forward(input);

        foreach (var v in output)
            Assert.True(v >= 0f, $"ReLU output must be non-negative; got {v}");
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_SwitchingModesDuringMultiplePasses_ProducesConsistentResults()
    {
        // Stress-test the training/inference switch: alternating modes should never corrupt output.
        var layer = new DenseLayer<float>(4, (IActivationFunction<float>)new ReLUActivation<float>());
        var input = Create2DInput(2, 8, seed: 99);

        layer.SetTrainingMode(false);
        var refOutput = layer.Forward(input);

        for (int pass = 0; pass < 5; pass++)
        {
            layer.SetTrainingMode(pass % 2 == 0);
            var output = layer.Forward(input);
            for (int i = 0; i < refOutput.Length; i++)
                Assert.Equal(refOutput[i], output[i], Tolerance);
        }
    }

    #endregion

    #region Layer Integration Tests




    [Fact(Timeout = 120000)]
    public async Task DenseWithDropout_TrainingVsInference_BehavesDifferently()
    {
        // Arrange
        var dense = new DenseLayer<float>(16);
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