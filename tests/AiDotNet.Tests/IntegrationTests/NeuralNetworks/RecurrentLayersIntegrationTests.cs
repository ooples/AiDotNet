using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for recurrent neural network layers (LSTM, GRU, Bidirectional, RecurrentLayer).
/// Tests forward pass, backward pass, shape correctness, sequence handling, and training/inference modes.
/// </summary>
public class RecurrentLayersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region GRULayer Tests

    [Fact]
    public void GRULayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(inputSize, hiddenSize, returnSequences: false, activation: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(hiddenSize, output.Shape[1]);
    }

    [Fact]
    public void GRULayer_ForwardPass_ReturnSequences_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(inputSize, hiddenSize, returnSequences: true, activation: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(timeSteps, output.Shape[1]);
        Assert.Equal(hiddenSize, output.Shape[2]);
    }

    [Fact]
    public void GRULayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(inputSize, hiddenSize, returnSequences: false, activation: tanh);
        layer.SetTrainingMode(true);

        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);
        var output = layer.Forward(input);

        var upstreamGradient = new Tensor<double>(output.Shape);
        upstreamGradient.Fill(1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(batchSize, gradient.Shape[0]);
        Assert.Equal(timeSteps, gradient.Shape[1]);
        Assert.Equal(inputSize, gradient.Shape[2]);
        AssertNoNaNOrInf(gradient);
    }

    [Fact]
    public void GRULayer_BackwardPass_ReturnSequences_ProducesValidGradients()
    {
        // Arrange
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(inputSize, hiddenSize, returnSequences: true, activation: tanh);
        layer.SetTrainingMode(true);

        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);
        var output = layer.Forward(input);

        var upstreamGradient = new Tensor<double>(output.Shape);
        upstreamGradient.Fill(1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
        AssertNoNaNOrInf(gradient);
    }

    [Fact]
    public void GRULayer_SupportsTraining_ReturnsTrue()
    {
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(10, 8, false, tanh);
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void GRULayer_GetParameters_ReturnsParameters()
    {
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(10, 8, false, tanh);
        var parameters = layer.GetParameters();
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0, "Parameters should not be empty");
    }

    [Fact]
    public void GRULayer_LongSequence_RemainsNumericallyStable()
    {
        // Arrange
        int batchSize = 2, timeSteps = 50, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(inputSize, hiddenSize, returnSequences: true, activation: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
        Assert.True(output.Max().maxVal < 100.0, "Values exploded in long sequence");
    }

    #endregion

    #region LSTMLayer Tests

    [Fact]
    public void LSTMLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(hiddenSize, output.Shape[^1]);
    }

    [Fact]
    public void LSTMLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        layer.SetTrainingMode(true);

        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);
        var output = layer.Forward(input);

        var upstreamGradient = new Tensor<double>(output.Shape);
        upstreamGradient.Fill(1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        AssertNoNaNOrInf(gradient);
    }

    [Fact]
    public void LSTMLayer_GetParameters_ReturnsParameters()
    {
        int[] inputShape = [1, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(10, 8, inputShape, tanh);
        var parameters = layer.GetParameters();
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0, "Parameters should not be empty");
    }

    [Fact]
    public void LSTMLayer_LongSequence_RemainsNumericallyStable()
    {
        // Arrange
        int batchSize = 2, timeSteps = 50, inputSize = 10, hiddenSize = 8;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
        Assert.True(output.Max().maxVal < 100.0, "Values exploded in long sequence");
    }

    [Fact]
    public void LSTMLayer_SupportsTraining_ReturnsTrue()
    {
        int[] inputShape = [1, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(10, 8, inputShape, tanh);
        Assert.True(layer.SupportsTraining);
    }

    #endregion

    #region BidirectionalLayer Tests

    [Fact]
    public void BidirectionalLayer_WithGRU_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var innerLayer = new GRULayer<double>(inputSize, hiddenSize, returnSequences: true, activation: tanh);
        var layer = new BidirectionalLayer<double>(innerLayer, mergeMode: true, activationFunction: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // With mergeMode=true, outputs are added together, so last dim = hiddenSize (not hiddenSize * 2)
        Assert.Equal(hiddenSize, output.Shape[^1]);
    }

    [Fact]
    public void BidirectionalLayer_WithLSTM_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var innerLayer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        var layer = new BidirectionalLayer<double>(innerLayer, mergeMode: true, activationFunction: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // With mergeMode=true, outputs are added together, so last dim = hiddenSize (not hiddenSize * 2)
        Assert.Equal(hiddenSize, output.Shape[^1]);
    }

    [Fact]
    public void BidirectionalLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var innerLayer = new GRULayer<double>(inputSize, hiddenSize, returnSequences: true, activation: tanh);
        var layer = new BidirectionalLayer<double>(innerLayer, mergeMode: true, activationFunction: tanh);
        layer.SetTrainingMode(true);

        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);
        var output = layer.Forward(input);

        var upstreamGradient = new Tensor<double>(output.Shape);
        upstreamGradient.Fill(1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(input.Shape, gradient.Shape);
        AssertNoNaNOrInf(gradient);
    }

    #endregion

    #region RecurrentLayer Base Tests

    [Fact]
    public void RecurrentLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new RecurrentLayer<double>(inputSize, hiddenSize, tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(hiddenSize, output.Shape[^1]);
    }

    [Fact]
    public void RecurrentLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new RecurrentLayer<double>(inputSize, hiddenSize, tanh);
        layer.SetTrainingMode(true);

        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);
        var output = layer.Forward(input);

        var upstreamGradient = new Tensor<double>(output.Shape);
        upstreamGradient.Fill(1.0);

        // Act
        var gradient = layer.Backward(upstreamGradient);

        // Assert
        Assert.NotNull(gradient);
        AssertNoNaNOrInf(gradient);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void GRULayer_ZeroInput_HandlesGracefully()
    {
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(inputSize, hiddenSize, false, tanh);
        var input = new Tensor<double>(inputShape);
        input.Fill(0.0);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    [Fact]
    public void LSTMLayer_ZeroInput_HandlesGracefully()
    {
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        var input = new Tensor<double>(inputShape);
        input.Fill(0.0);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    [Fact]
    public void GRULayer_LargeInputValues_RemainsStable()
    {
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(inputSize, hiddenSize, false, tanh);
        var input = new Tensor<double>(inputShape);
        input.Fill(10.0);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    [Fact]
    public void LSTMLayer_LargeInputValues_RemainsStable()
    {
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        var input = new Tensor<double>(inputShape);
        input.Fill(10.0);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    [Fact]
    public void GRULayer_BatchSizeOne_WorksCorrectly()
    {
        int batchSize = 1, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>(inputSize, hiddenSize, returnSequences: true, activation: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        Assert.Equal(batchSize, output.Shape[0]);
    }

    [Fact]
    public void LSTMLayer_BatchSizeOne_WorksCorrectly()
    {
        int batchSize = 1, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>(inputSize, hiddenSize, inputShape, tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        Assert.Equal(batchSize, output.Shape[0]);
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void GRULayer_Clone_CreatesIndependentCopy()
    {
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var original = new GRULayer<double>(10, 8, false, tanh);
        var input = Tensor<double>.CreateRandom(2, 5, 10);

        var originalOutput = original.Forward(input);
        var clone = original.Clone();
        var cloneOutput = clone.Forward(input);

        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    [Fact]
    public void LSTMLayer_Clone_CreatesIndependentCopy()
    {
        int[] inputShape = [2, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var original = new LSTMLayer<double>(10, 8, inputShape, tanh);
        var input = Tensor<double>.CreateRandom(2, 5, 10);

        var originalOutput = original.Forward(input);
        var clone = original.Clone();
        var cloneOutput = clone.Forward(input);

        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape, cloneOutput.Shape);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    #endregion

    #region Helper Methods

    private static void AssertNoNaNOrInf(Tensor<double> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.False(double.IsNaN(tensor[i]), $"Tensor contains NaN at index {i}");
            Assert.False(double.IsInfinity(tensor[i]), $"Tensor contains Infinity at index {i}");
        }
    }

    #endregion
}
