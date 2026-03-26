using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for layers that require two input tensors in their Forward pass
/// (e.g., CrossAttention, TransformerDecoder, MemoryRead/Write).
/// Tests mathematical invariants: finite output, determinism, gradient flow, and parameter consistency.
///
/// Subclasses override CreateLayer(), and optionally PrimaryInputShape/SecondaryInputShape.
/// The Forward method is called via the layer's Forward(Tensor, Tensor) overload.
/// </summary>
public abstract class DualInputLayerTestBase
{
    /// <summary>
    /// Factory method — create a fresh instance of the dual-input layer under test.
    /// The returned layer must have a Forward(Tensor, Tensor) method.
    /// </summary>
    protected abstract ILayer<double> CreateLayer();

    /// <summary>
    /// Shape of the primary input tensor (first argument to Forward).
    /// Default: [1, 4] — single sample, 4 features.
    /// </summary>
    protected virtual int[] PrimaryInputShape => [1, 4];

    /// <summary>
    /// Shape of the secondary input tensor (second argument to Forward).
    /// Default: same as PrimaryInputShape.
    /// </summary>
    protected virtual int[] SecondaryInputShape => PrimaryInputShape;

    /// <summary>Whether the layer has trainable parameters. Default: true.</summary>
    protected virtual bool ExpectsTrainableParameters => true;

    /// <summary>Whether Backward produces non-zero weight gradients. Default: true.</summary>
    protected virtual bool ExpectsNonZeroGradients => true;

    /// <summary>Whether constant primary inputs should produce different outputs.
    /// False for attention-based memory layers where constant keys produce uniform attention.</summary>
    protected virtual bool ExpectsDifferentOutputForConstantInputs => true;

    // =========================================================================
    // Helpers
    // =========================================================================

    protected static Tensor<double> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    /// <summary>
    /// Calls the dual-input Forward method via reflection, since ILayer only declares
    /// Forward(Tensor). Layers with dual-input define an additional Forward(Tensor, Tensor).
    /// </summary>
    protected Tensor<double> ForwardDual(ILayer<double> layer, Tensor<double> primary, Tensor<double> secondary)
    {
        var method = layer.GetType().GetMethod("Forward",
            new[] { typeof(Tensor<double>), typeof(Tensor<double>) });

        if (method is null)
        {
            // Fall back to standard Forward with just the primary input
            return layer.Forward(primary);
        }

        var result = method.Invoke(layer, new object[] { primary, secondary });
        if (result is Tensor<double> tensor)
            return tensor;

        throw new InvalidOperationException(
            $"Forward(Tensor, Tensor) on {layer.GetType().Name} returned {result?.GetType().Name ?? "null"} instead of Tensor<double>.");
    }

    // =========================================================================
    // INVARIANT 1: Forward produces finite, non-empty output
    // =========================================================================

    [Fact]
    public void Forward_ShouldProduceFiniteOutput()
    {
        var layer = CreateLayer();
        var primary = CreateRandomTensor(PrimaryInputShape);
        var secondary = CreateRandomTensor(SecondaryInputShape, seed: 77);

        var output = ForwardDual(layer, primary, secondary);

        Assert.True(output.Length > 0, "Layer output should not be empty.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN — numerical instability in Forward.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity — overflow in Forward.");
        }
    }

    // =========================================================================
    // INVARIANT 2: Forward is deterministic
    // =========================================================================

    [Fact]
    public void Forward_ShouldBeDeterministic()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(false);
        var primary = CreateRandomTensor(PrimaryInputShape);
        var secondary = CreateRandomTensor(SecondaryInputShape, seed: 77);

        var out1 = ForwardDual(layer, primary, secondary);
        layer.ResetState();
        var out2 = ForwardDual(layer, primary, secondary);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
    }

    // =========================================================================
    // INVARIANT 3: Different primary inputs produce different outputs
    // =========================================================================

    [Fact]
    public void Forward_DifferentPrimaryInputs_ShouldProduceDifferentOutputs()
    {
        if (!ExpectsDifferentOutputForConstantInputs) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(false);
        var secondary = CreateRandomTensor(SecondaryInputShape, seed: 77);

        var input1 = CreateRandomTensor(PrimaryInputShape, seed: 1);
        var input2 = CreateRandomTensor(PrimaryInputShape, seed: 2);

        layer.ResetState();
        var output1 = ForwardDual(layer, input1, secondary);
        layer.ResetState();
        var output2 = ForwardDual(layer, input2, secondary);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Layer produces identical output for different primary inputs.");
    }

    // =========================================================================
    // INVARIANT 4: Backward produces finite gradient
    // =========================================================================

    [Fact]
    public void Backward_ShouldProduceFiniteGradient()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(true);
        var primary = CreateRandomTensor(PrimaryInputShape);
        var secondary = CreateRandomTensor(SecondaryInputShape, seed: 77);

        var output = ForwardDual(layer, primary, secondary);
        var outputGrad = CreateRandomTensor(output.Shape.ToArray(), seed: 99);

        var inputGrad = layer.Backward(outputGrad);

        Assert.True(inputGrad.Length > 0, "Input gradient should not be empty.");
        for (int i = 0; i < inputGrad.Length; i++)
        {
            Assert.False(double.IsNaN(inputGrad[i]),
                $"InputGradient[{i}] is NaN — broken backward pass.");
            Assert.False(double.IsInfinity(inputGrad[i]),
                $"InputGradient[{i}] is Infinity — gradient explosion.");
        }
    }

    // =========================================================================
    // INVARIANT 5: Parameter count consistency
    // =========================================================================

    [Fact]
    public void Parameters_CountShouldMatchVector()
    {
        var layer = CreateLayer();
        int count = layer.ParameterCount;
        var parameters = layer.GetParameters();

        Assert.True(count >= 0, "ParameterCount should be non-negative.");
        Assert.Equal(count, parameters.Length);

        if (ExpectsTrainableParameters)
            Assert.True(count > 0, "Layer should have trainable parameters but ParameterCount is 0.");
    }

    // =========================================================================
    // INVARIANT 6: SetParameters → GetParameters roundtrip
    // =========================================================================

    [Fact]
    public void Parameters_SetGet_Roundtrip()
    {
        var layer = CreateLayer();
        if (layer.ParameterCount == 0) return;

        var original = layer.GetParameters();
        var modified = new Vector<double>(original.Length);
        for (int i = 0; i < original.Length; i++)
            modified[i] = original[i] + 0.001;

        layer.SetParameters(modified);
        var retrieved = layer.GetParameters();

        Assert.Equal(modified.Length, retrieved.Length);
        for (int i = 0; i < modified.Length; i++)
            Assert.Equal(modified[i], retrieved[i], 1e-15);
    }

    // =========================================================================
    // INVARIANT 7: Non-zero weight gradients after backward
    // =========================================================================

    [Fact]
    public void Backward_ShouldProduceNonZeroWeightGradients()
    {
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(true);
        layer.ClearGradients();

        var primary = CreateRandomTensor(PrimaryInputShape);
        var secondary = CreateRandomTensor(SecondaryInputShape, seed: 77);
        var output = ForwardDual(layer, primary, secondary);
        var outputGrad = CreateRandomTensor(output.Shape.ToArray(), seed: 99);

        layer.Backward(outputGrad);
        var gradients = layer.GetParameterGradients();

        Assert.True(gradients.Length > 0, "Trainable layer should have gradients after Backward.");

        bool anyNonZero = false;
        for (int i = 0; i < gradients.Length; i++)
        {
            if (Math.Abs(gradients[i]) > 1e-15)
            {
                anyNonZero = true;
                break;
            }
        }
        Assert.True(anyNonZero, "All parameter gradients are zero after Backward.");
    }

    // =========================================================================
    // INVARIANT 8: ResetState doesn't break the layer
    // =========================================================================

    [Fact]
    public void ResetState_ShouldNotBreakForward()
    {
        var layer = CreateLayer();
        var primary = CreateRandomTensor(PrimaryInputShape);
        var secondary = CreateRandomTensor(SecondaryInputShape, seed: 77);

        ForwardDual(layer, primary, secondary);
        layer.ResetState();

        var output = ForwardDual(layer, primary, secondary);
        Assert.True(output.Length > 0, "Output should not be empty after ResetState.");
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN after ResetState + Forward.");
    }
}
