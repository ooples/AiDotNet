using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for recurrent neural network layers (LSTM, GRU, Bidirectional, RecurrentLayer).
/// Tests forward pass, backward pass, shape correctness, sequence handling, and training/inference modes.
/// </summary>
public class RecurrentLayersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region GRULayer Tests

    [Fact(Timeout = 120000)]
    public async Task GRULayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>( hiddenSize, returnSequences: false, activation: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(hiddenSize, output.Shape[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task GRULayer_ForwardPass_ReturnSequences_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>( hiddenSize, returnSequences: true, activation: tanh);
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



    [Fact(Timeout = 120000)]
    public async Task GRULayer_SupportsTraining_ReturnsTrue()
    {
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>( 8, false, tanh);
        Assert.True(layer.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task GRULayer_GetParameters_ReturnsParameters()
    {
        int[] inputShape = [1, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>( 8, false, tanh);
        // GRULayer is lazy-only (eager ctors removed in #1212): its own ParameterCount
        // doc says "call ResolveFromShape first". Resolve the input width so weights
        // allocate and GetParameters returns the real parameter vector before Forward.
        layer.ResolveFromShape(inputShape);
        var parameters = layer.GetParameters();
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0, "Parameters should not be empty");
    }

    [Fact(Timeout = 120000)]
    public async Task GRULayer_LongSequence_RemainsNumericallyStable()
    {
        // Arrange
        int batchSize = 2, timeSteps = 50, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>( hiddenSize, returnSequences: true, activation: tanh);
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

    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>( hiddenSize, tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(hiddenSize, output.Shape[^1]);
    }


    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_GetParameters_ReturnsParameters()
    {
        int[] inputShape = [1, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>( 8, tanh);
        // LSTMLayer is lazy-only (eager ctors removed in #1212): input feature
        // width is unknown from hiddenSize alone, so weights stay [0,0] until the
        // shape is resolved. ResolveFromShape is the documented bridge that lets
        // GetParameters / ParameterCount work on a freshly-constructed layer before
        // any Forward — exactly what a parent network does during ResolveLazyLayerShapes.
        layer.ResolveFromShape(inputShape);
        var parameters = layer.GetParameters();
        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0, "Parameters should not be empty");
    }

    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_LongSequence_RemainsNumericallyStable()
    {
        // Arrange
        int batchSize = 2, timeSteps = 50, inputSize = 10, hiddenSize = 8;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>( hiddenSize, tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
        Assert.True(output.Max().maxVal < 100.0, "Values exploded in long sequence");
    }

    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_FusedInferencePath_MatchesDecomposedTrainingPath()
    {
        await Task.Yield();
        // Coverage for the inference-only fused fast path in LSTMLayer.Forward
        // (routes the whole sequence through CpuEngine.LstmSequenceForward). It is
        // gated on typeof(T) == float && !IsTrainingMode && CpuEngine && !GraphMode,
        // so the double-typed tests above never exercise it. This asserts the fused
        // path (inference mode) produces the SAME output as the decomposed per-step
        // loop (training mode) within float epsilon, on whatever AiDotNet.Tensors
        // version is pinned — guards both the weight-packing wiring and the fused
        // kernel's contract across Tensors bumps.
        int batchSize = 4, timeSteps = 8, inputSize = 6, hiddenSize = 5;
        var layer = new LSTMLayer<float>(hiddenSize);
        var input = Tensor<float>.CreateRandom(batchSize, timeSteps, inputSize);

        // Materialize weights via a training-mode forward, then capture the
        // decomposed-path output (training mode keeps the per-step loop).
        layer.SetTrainingMode(true);
        var slow = layer.Forward(input);

        // Inference mode engages the fused path.
        layer.SetTrainingMode(false);
        var fast = layer.Forward(input);

        Assert.Equal(slow.Length, fast.Length);
        double maxAbsDiff = 0.0;
        for (int i = 0; i < slow.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)slow[i] - fast[i]));
        Assert.True(maxAbsDiff < 1e-4,
            $"Fused LSTM inference output diverged from the decomposed path: maxAbsDiff={maxAbsDiff:E3}.");
    }

    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_SupportsTraining_ReturnsTrue()
    {
        int[] inputShape = [1, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>( 8, tanh);
        Assert.True(layer.SupportsTraining);
    }

    #endregion

    #region BidirectionalLayer Tests

    [Fact(Timeout = 120000)]
    public async Task BidirectionalLayer_WithGRU_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var innerLayer = new GRULayer<double>( hiddenSize, returnSequences: true, activation: tanh);
        var layer = new BidirectionalLayer<double>(innerLayer, mergeMode: true, activationFunction: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // With mergeMode=true, outputs are added together, so last dim = hiddenSize (not hiddenSize * 2)
        Assert.Equal(hiddenSize, output.Shape[^1]);
    }

    [Fact(Timeout = 120000)]
    public async Task BidirectionalLayer_WithLSTM_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var innerLayer = new LSTMLayer<double>( hiddenSize, tanh);
        var layer = new BidirectionalLayer<double>(innerLayer, mergeMode: true, activationFunction: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        // With mergeMode=true, outputs are added together, so last dim = hiddenSize (not hiddenSize * 2)
        Assert.Equal(hiddenSize, output.Shape[^1]);
    }


    #endregion

    #region RecurrentLayer Base Tests

    [Fact(Timeout = 120000)]
    public async Task RecurrentLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int batchSize = 2, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new RecurrentLayer<double>( hiddenSize, tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(hiddenSize, output.Shape[^1]);
    }


    #endregion

    #region Edge Cases

    [Fact(Timeout = 120000)]
    public async Task GRULayer_ZeroInput_HandlesGracefully()
    {
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>( hiddenSize, false, tanh);
        var input = new Tensor<double>(inputShape);
        input.Fill(0.0);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_ZeroInput_HandlesGracefully()
    {
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>( hiddenSize, tanh);
        var input = new Tensor<double>(inputShape);
        input.Fill(0.0);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    [Fact(Timeout = 120000)]
    public async Task GRULayer_LargeInputValues_RemainsStable()
    {
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>( hiddenSize, false, tanh);
        var input = new Tensor<double>(inputShape);
        input.Fill(10.0);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_LargeInputValues_RemainsStable()
    {
        int batchSize = 2, timeSteps = 3, inputSize = 5, hiddenSize = 4;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>( hiddenSize, tanh);
        var input = new Tensor<double>(inputShape);
        input.Fill(10.0);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        AssertNoNaNOrInf(output);
    }

    [Fact(Timeout = 120000)]
    public async Task GRULayer_BatchSizeOne_WorksCorrectly()
    {
        int batchSize = 1, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new GRULayer<double>( hiddenSize, returnSequences: true, activation: tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        Assert.Equal(batchSize, output.Shape[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_BatchSizeOne_WorksCorrectly()
    {
        int batchSize = 1, timeSteps = 5, inputSize = 10, hiddenSize = 8;
        int[] inputShape = [batchSize, timeSteps, inputSize];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var layer = new LSTMLayer<double>( hiddenSize, tanh);
        var input = Tensor<double>.CreateRandom(batchSize, timeSteps, inputSize);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        Assert.Equal(batchSize, output.Shape[0]);
    }

    #endregion

    #region Clone Tests

    [Fact(Timeout = 120000)]
    public async Task GRULayer_Clone_CreatesIndependentCopy()
    {
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var original = new GRULayer<double>( 8, false, tanh);
        var input = Tensor<double>.CreateRandom(2, 5, 10);

        var originalOutput = original.Forward(input);
        var clone = original.Clone();
        var cloneOutput = clone.Forward(input);

        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.Equal(originalOutput[i], cloneOutput[i], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task LSTMLayer_Clone_CreatesIndependentCopy()
    {
        int[] inputShape = [2, 5, 10];
        IActivationFunction<double> tanh = new TanhActivation<double>();
        var original = new LSTMLayer<double>( 8, tanh);
        var input = Tensor<double>.CreateRandom(2, 5, 10);

        var originalOutput = original.Forward(input);
        var clone = original.Clone();
        var cloneOutput = clone.Forward(input);

        Assert.NotSame(original, clone);
        Assert.Equal(originalOutput.Shape.ToArray(), cloneOutput.Shape.ToArray());
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
