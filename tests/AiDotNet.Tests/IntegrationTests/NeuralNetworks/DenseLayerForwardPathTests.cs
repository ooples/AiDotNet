using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for the <see cref="DenseLayer{T}.Forward"/> path change introduced in this PR:
/// fused activation is now only used during inference (IsTrainingMode == false).
/// During training the layer performs a single FusedLinear(none) call followed by a separate
/// activation step so the gradient tape records exactly one entry per forward pass.
/// </summary>
public class DenseLayerForwardPathTests
{
    // ──────────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────────

    private static DenseLayer<double> CreateLayer(int inputSize, int outputSize)
        => new DenseLayer<double>(inputSize, outputSize, new ReLUActivation<double>());

    private static DenseLayer<double> CreateIdentityLayer(int inputSize, int outputSize)
        => new DenseLayer<double>(inputSize, outputSize, new IdentityActivation<double>());

    private static Tensor<double> CreateInput(int inputSize, double fillValue = 1.0)
    {
        var t = new Tensor<double>([1, inputSize]);
        for (int i = 0; i < inputSize; i++)
            t[0, i] = fillValue;
        return t;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Shape contracts: both modes must produce the same output shape
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Forward_InferenceMode_WithReLU_ProducesCorrectOutputShape()
    {
        // Arrange
        var layer = CreateLayer(4, 8);
        layer.SetTrainingMode(false); // inference path: fused activation

        var input = CreateInput(4);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(8, output.Shape[1]);
    }

    [Fact]
    public void Forward_TrainingMode_WithReLU_ProducesCorrectOutputShape()
    {
        // Arrange
        var layer = CreateLayer(4, 8);
        layer.SetTrainingMode(true); // training path: separate pre-activation + activation

        var input = CreateInput(4);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(8, output.Shape[1]);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Value contracts: both paths must produce the same numeric results for
    // the same weights.  We use a zero-input so the output equals the biases
    // after activation, which is deterministic.
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Forward_InferenceAndTraining_ProduceSameOutputValues_WithZeroInput()
    {
        // Arrange — two layers with identical parameters
        var layerA = CreateLayer(4, 4);
        var layerB = CreateLayer(4, 4);
        // Copy parameters from A to B so they're identical
        layerB.SetParameters(layerA.GetParameters());

        layerA.SetTrainingMode(false); // inference
        layerB.SetTrainingMode(true);  // training

        var inputA = CreateInput(4, 0.0);
        var inputB = CreateInput(4, 0.0);

        // Act
        var outputA = layerA.Forward(inputA);
        var outputB = layerB.Forward(inputB);

        // Assert — same numeric result regardless of path
        Assert.Equal(outputA.Length, outputB.Length);
        for (int i = 0; i < outputA.Length; i++)
        {
            Assert.Equal(outputA.GetFlat(i), outputB.GetFlat(i), 12,
                $"Element {i} differs between inference and training forward paths.");
        }
    }

    [Fact]
    public void Forward_InferenceAndTraining_ProduceSameOutputValues_WithNonZeroInput()
    {
        // Arrange
        var layerA = CreateLayer(3, 5);
        var layerB = CreateLayer(3, 5);
        layerB.SetParameters(layerA.GetParameters());

        layerA.SetTrainingMode(false);
        layerB.SetTrainingMode(true);

        var inputA = CreateInput(3, 0.5);
        var inputB = CreateInput(3, 0.5);

        // Act
        var outputA = layerA.Forward(inputA);
        var outputB = layerB.Forward(inputB);

        // Assert
        Assert.Equal(outputA.Length, outputB.Length);
        for (int i = 0; i < outputA.Length; i++)
        {
            Assert.Equal(outputA.GetFlat(i), outputB.GetFlat(i), 10,
                $"Element {i} differs between inference and training forward paths.");
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Identity activation: GetFusedActivationType() returns None for
    // IdentityActivation, so both modes always take the else (separate) branch.
    // The newly added Activate(Tensor<T>) override makes the activation step a
    // zero-cost pass-through.  Values must still be correct in both modes.
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Forward_IdentityActivation_InferenceMode_ProducesCorrectShape()
    {
        // Arrange
        var layer = CreateIdentityLayer(4, 6);
        layer.SetTrainingMode(false);
        var input = CreateInput(4, 1.0);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(6, output.Shape[1]);
    }

    [Fact]
    public void Forward_IdentityActivation_TrainingMode_ProducesCorrectShape()
    {
        // Arrange
        var layer = CreateIdentityLayer(4, 6);
        layer.SetTrainingMode(true);
        var input = CreateInput(4, 1.0);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(6, output.Shape[1]);
    }

    [Fact]
    public void Forward_IdentityActivation_BothModes_ProduceSameOutputValues()
    {
        // Arrange — Identity activation: no non-linearity; outputs are purely linear
        var layerA = CreateIdentityLayer(3, 4);
        var layerB = CreateIdentityLayer(3, 4);
        layerB.SetParameters(layerA.GetParameters());

        layerA.SetTrainingMode(false);
        layerB.SetTrainingMode(true);

        var inputA = CreateInput(3, 2.0);
        var inputB = CreateInput(3, 2.0);

        // Act
        var outputA = layerA.Forward(inputA);
        var outputB = layerB.Forward(inputB);

        // Assert
        Assert.Equal(outputA.Length, outputB.Length);
        for (int i = 0; i < outputA.Length; i++)
        {
            Assert.Equal(outputA.GetFlat(i), outputB.GetFlat(i), 12,
                $"Identity-activation layer element {i} differs between modes.");
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Multiple consecutive forward passes in training mode must produce
    // consistent results (regression test: calling FusedLinear twice in the
    // old code could corrupt tape entries via RemoveLastNTapeEntries).
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Forward_TrainingMode_MultipleConsecutiveCalls_ProduceConsistentResults()
    {
        // Arrange
        var layer = CreateLayer(4, 4);
        layer.SetTrainingMode(true);

        var input = CreateInput(4, 1.0);

        // Act — three consecutive forward passes with the same input
        var output1 = layer.Forward(input);
        var output2 = layer.Forward(input);
        var output3 = layer.Forward(input);

        // Assert — each call should produce the same values
        for (int i = 0; i < output1.Length; i++)
        {
            Assert.Equal(output1.GetFlat(i), output2.GetFlat(i), 12,
                $"Element {i}: outputs differ between 1st and 2nd training forward pass.");
            Assert.Equal(output1.GetFlat(i), output3.GetFlat(i), 12,
                $"Element {i}: outputs differ between 1st and 3rd training forward pass.");
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Switching modes between calls: going from training → inference → training
    // must all produce the correct shape and values each time.
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Forward_ModeSwitching_ProducesCorrectShapeAndValues()
    {
        // Arrange
        var layer = CreateLayer(4, 4);
        var input = CreateInput(4, 0.5);

        // Act & Assert — train → infer → train
        layer.SetTrainingMode(true);
        var t1 = layer.Forward(input);
        Assert.Equal(4, t1.Shape[1]);

        layer.SetTrainingMode(false);
        var t2 = layer.Forward(input);
        Assert.Equal(4, t2.Shape[1]);

        layer.SetTrainingMode(true);
        var t3 = layer.Forward(input);
        Assert.Equal(4, t3.Shape[1]);

        // Values must be the same across mode switches (same weights, same input)
        for (int i = 0; i < t1.Length; i++)
        {
            Assert.Equal(t1.GetFlat(i), t2.GetFlat(i), 10,
                $"Element {i}: training vs inference value mismatch after mode switch.");
            Assert.Equal(t1.GetFlat(i), t3.GetFlat(i), 10,
                $"Element {i}: training values differ across mode-switch round-trip.");
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // ReLU in inference mode: negative pre-activations must be clamped to zero
    // (ensures the fused path actually applies the activation).
    // ──────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Forward_InferenceMode_ReLU_ClampsNegativeOutputsToZero()
    {
        // Arrange — a 1-input, 1-output layer lets us control the sign easily.
        // Set weights to a large negative value so the pre-activation is negative.
        var layer = new DenseLayer<double>(1, 1, new ReLUActivation<double>());

        // Override with known negative weight and bias
        var w = layer.GetWeights();
        w.SetFlat(0, -100.0); // weight = -100
        var b = layer.GetBiases();
        b.SetFlat(0, 0.0);    // bias = 0

        layer.SetTrainingMode(false); // use fused path

        var input = new Tensor<double>([1, 1]);
        input[0, 0] = 1.0; // pre-activation = -100 * 1 + 0 = -100

        // Act
        var output = layer.Forward(input);

        // Assert — ReLU clamps to 0
        Assert.Equal(0.0, output.GetFlat(0), 12);
    }

    [Fact]
    public void Forward_TrainingMode_ReLU_ClampsNegativeOutputsToZero()
    {
        // Arrange — same setup but in training mode (separate path)
        var layer = new DenseLayer<double>(1, 1, new ReLUActivation<double>());

        var w = layer.GetWeights();
        w.SetFlat(0, -100.0);
        var b = layer.GetBiases();
        b.SetFlat(0, 0.0);

        layer.SetTrainingMode(true);

        var input = new Tensor<double>([1, 1]);
        input[0, 0] = 1.0;

        // Act
        var output = layer.Forward(input);

        // Assert — training path must also clamp to 0
        Assert.Equal(0.0, output.GetFlat(0), 12);
    }
}