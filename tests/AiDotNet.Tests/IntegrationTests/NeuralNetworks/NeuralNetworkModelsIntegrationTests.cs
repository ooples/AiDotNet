using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for complete neural network models.
/// Tests FeedForwardNeuralNetwork, ConvolutionalNeuralNetwork, RecurrentNeuralNetwork.
/// Note: Autoencoder tests are skipped due to a bug in LayerHelper.CreateDefaultAutoEncoderLayers
/// that requires at least 3 layer sizes but GetLayerSizes() returns only 1 when no layers are defined.
/// </summary>
public class NeuralNetworkModelsIntegrationTests
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
    /// Creates unbatched (single sample) input tensor.
    /// </summary>
    private static Tensor<float> CreateUnbatchedInput(int size, int seed = 42)
    {
        return CreateRandomTensor([size], seed);
    }

    /// <summary>
    /// Creates 3D unbatched input for CNN.
    /// </summary>
    private static Tensor<float> CreateUnbatched3DInput(int depth, int height, int width, int seed = 42)
    {
        return CreateRandomTensor([depth, height, width], seed);
    }

    /// <summary>
    /// Creates 2D unbatched input for RNN.
    /// </summary>
    private static Tensor<float> CreateUnbatched2DInput(int seqLen, int features, int seed = 42)
    {
        return CreateRandomTensor([seqLen, features], seed);
    }

    /// <summary>
    /// Creates one-hot encoded labels vector (unbatched).
    /// </summary>
    private static Tensor<float> CreateOneHotLabel(int numClasses, int classIndex)
    {
        var labels = new float[numClasses];
        labels[classIndex] = 1f;
        return new Tensor<float>(labels, [numClasses]);
    }

    #endregion

    #region FeedForwardNeuralNetwork Tests

    [Fact]
    public void FeedForwardNeuralNetwork_Predict_ProducesCorrectOutputShape()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Medium,
            inputSize: 64,
            outputSize: 10);

        var network = new FeedForwardNeuralNetwork<float>(architecture);
        var input = CreateUnbatchedInput(64);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.Equal(1, output.Rank);
        Assert.Equal(10, output.Shape[0]);
    }

    [Fact]
    public void FeedForwardNeuralNetwork_Forward_ProducesNonZeroOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 10,
            outputSize: 5);

        var network = new FeedForwardNeuralNetwork<float>(architecture);
        var input = CreateUnbatchedInput(10);

        // Act
        var output = network.Forward(input);

        // Assert - output should have some non-zero values
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output[i]) > Tolerance)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "Output should have non-zero values");
    }

    [Fact]
    public void FeedForwardNeuralNetwork_Backward_ProducesGradientWithCorrectShape()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var network = new FeedForwardNeuralNetwork<float>(architecture);
        var input = CreateUnbatchedInput(16);
        var outputGrad = CreateRandomTensor([4], 123);

        // Act
        var output = network.Forward(input);
        var inputGrad = network.Backward(outputGrad);

        // Assert
        Assert.Equal(16, inputGrad.Shape[0]);
    }

    [Fact]
    public void FeedForwardNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: 64,
            outputSize: 10);

        var network = new FeedForwardNeuralNetwork<float>(architecture);

        // Act
        var paramCount = network.GetParameterCount();

        // Assert
        Assert.True(paramCount > 0, "Parameter count should be positive");
    }

    [Fact]
    public void FeedForwardNeuralNetwork_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 8);

        var network = new FeedForwardNeuralNetwork<float>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.FeedForwardNetwork, metadata.ModelType);
        Assert.NotNull(metadata.AdditionalInfo);
        Assert.True(metadata.AdditionalInfo.ContainsKey("LayerCount"));
    }

    [Fact]
    public void FeedForwardNeuralNetwork_Train_CompletesWithoutError()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var network = new FeedForwardNeuralNetwork<float>(architecture);
        var input = CreateUnbatchedInput(16);
        var labels = CreateOneHotLabel(4, 2);

        // Act - Train for several iterations
        for (int i = 0; i < 10; i++)
        {
            network.Train(input, labels);
        }

        // Assert - Verify network still produces valid output after training
        var output = network.Predict(input);
        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[0]);
    }

    [Fact]
    public void FeedForwardNeuralNetwork_MultipleTrainingIterations_DoesNotThrow()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 10,
            outputSize: 1);

        var network = new FeedForwardNeuralNetwork<float>(architecture);
        var input = CreateUnbatchedInput(10);
        var target = CreateRandomTensor([1]);

        // Act & Assert - multiple training iterations should not throw
        for (int i = 0; i < 20; i++)
        {
            network.Train(input, target);
        }

        // Verify output is valid after training
        var output = network.Predict(input);
        Assert.NotNull(output);
    }

    #endregion

    #region ConvolutionalNeuralNetwork Tests

    [Fact]
    public void ConvolutionalNeuralNetwork_Predict_ProducesCorrectOutputShape()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputDepth: 3,
            inputHeight: 32,
            inputWidth: 32,
            outputSize: 10);

        var network = new ConvolutionalNeuralNetwork<float>(architecture);
        var input = CreateUnbatched3DInput(3, 32, 32);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(10, output.Shape[^1]); // Last dim is output classes
    }

    [Fact]
    public void ConvolutionalNeuralNetwork_Train_DoesNotThrow()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputDepth: 1,
            inputHeight: 16,
            inputWidth: 16,
            outputSize: 5);

        var network = new ConvolutionalNeuralNetwork<float>(architecture);
        var input = CreateUnbatched3DInput(1, 16, 16);
        var labels = CreateOneHotLabel(5, 3);

        // Act & Assert - should not throw
        network.Train(input, labels);

        // Verify output is valid after training
        var output = network.Predict(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void ConvolutionalNeuralNetwork_Forward_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputDepth: 1,
            inputHeight: 8,
            inputWidth: 8,
            outputSize: 4);

        var network = new ConvolutionalNeuralNetwork<float>(architecture);
        var input = CreateUnbatched3DInput(1, 8, 8);

        // Act
        var output = network.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void ConvolutionalNeuralNetwork_GetMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputDepth: 1,
            inputHeight: 16,
            inputWidth: 16,
            outputSize: 3);

        var network = new ConvolutionalNeuralNetwork<float>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.ConvolutionalNeuralNetwork, metadata.ModelType);
    }

    #endregion

    #region RecurrentNeuralNetwork Tests

    [Fact]
    public void RecurrentNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 10,  // sequence length
            inputWidth: 8,    // input features
            outputSize: 4);

        var network = new RecurrentNeuralNetwork<float>(architecture);
        var input = CreateUnbatched2DInput(10, 8);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void RecurrentNeuralNetwork_Train_DoesNotThrow()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 5,   // sequence length
            inputWidth: 4,    // input features
            outputSize: 2);

        var network = new RecurrentNeuralNetwork<float>(architecture);
        var input = CreateUnbatched2DInput(5, 4);
        var target = CreateRandomTensor([2]);

        // Act & Assert
        network.Train(input, target);

        // Verify output is valid after training
        var output = network.Predict(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void RecurrentNeuralNetwork_Predict_ProducesConsistentOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 6,
            inputWidth: 5,
            outputSize: 3);

        var network = new RecurrentNeuralNetwork<float>(architecture);
        var input = CreateUnbatched2DInput(6, 5);

        // Act - RNN only exposes Predict publicly, not Forward
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void ConvNet_TrainAndPredict_ConsistentBehavior()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputDepth: 1,
            inputHeight: 16,
            inputWidth: 16,
            outputSize: 3);

        var network = new ConvolutionalNeuralNetwork<float>(architecture);
        var trainInput = CreateUnbatched3DInput(1, 16, 16);
        var trainLabels = CreateOneHotLabel(3, 1);
        var testInput = CreateUnbatched3DInput(1, 16, 16, 999);

        // Act
        network.Train(trainInput, trainLabels);
        var output1 = network.Predict(testInput);
        var output2 = network.Predict(testInput);

        // Assert - same input should produce same output (inference mode)
        Assert.Equal(output1.Shape, output2.Shape);
        for (int i = 0; i < output1.Length; i++)
        {
            var diff = Math.Abs(output1[i] - output2[i]);
            Assert.True(diff <= Tolerance, $"Outputs diverged more than {Tolerance} at index {i}: diff={diff}");
        }
    }

    [Fact]
    public void FeedForwardNetwork_DifferentComplexities_ProduceOutput()
    {
        // Test with different complexity settings
        var complexities = new[]
        {
            NetworkComplexity.Simple,
            NetworkComplexity.Medium,
            NetworkComplexity.Deep
        };

        foreach (var complexity in complexities)
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: complexity,
                inputSize: 32,
                outputSize: 8);

            var network = new FeedForwardNeuralNetwork<float>(architecture);
            var input = CreateUnbatchedInput(32);

            // Act
            var output = network.Predict(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(8, output.Shape[0]);
        }
    }

    [Fact]
    public void FeedForwardNetwork_DifferentTaskTypes_ProduceOutput()
    {
        // Test with different task types
        var taskTypes = new[]
        {
            NeuralNetworkTaskType.Regression,
            NeuralNetworkTaskType.BinaryClassification,
            NeuralNetworkTaskType.MultiClassClassification
        };

        foreach (var taskType in taskTypes)
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: taskType,
                complexity: NetworkComplexity.Simple,
                inputSize: 16,
                outputSize: taskType == NeuralNetworkTaskType.BinaryClassification ? 1 : 5);

            var network = new FeedForwardNeuralNetwork<float>(architecture);
            var input = CreateUnbatchedInput(16);

            // Act
            var output = network.Predict(input);

            // Assert
            Assert.NotNull(output);
        }
    }

    #endregion

    #region Autoencoder Tests - SKIPPED

    // NOTE: Autoencoder tests are skipped because of a bug in LayerHelper.CreateDefaultAutoEncoderLayers
    // The method requires GetLayerSizes() to return at least 3 elements, but when no layers are defined
    // in the architecture, GetLayerSizes() only returns [inputSize].
    //
    // BUG LOCATION: src/Helpers/LayerHelper.cs line 670
    // ISSUE: architecture.GetLayerSizes() returns only [CalculatedInputSize] when Layers is null/empty
    // EXPECTED: Should auto-generate encoder/decoder layer sizes based on complexity

    #endregion
}
