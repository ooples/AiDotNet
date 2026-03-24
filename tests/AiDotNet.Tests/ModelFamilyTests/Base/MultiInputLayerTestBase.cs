using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for layers that require multiple input tensors via Forward(params Tensor[]).
/// Used by merge layers: AddLayer, ConcatenateLayer, MultiplyLayer.
/// Tests forward output, backward gradient flow, parameter consistency, and serialization.
/// </summary>
public abstract class MultiInputLayerTestBase
{
    protected abstract ILayer<double> CreateLayer();

    /// <summary>Shape of each input tensor. Default: [1, 4].</summary>
    protected virtual int[] InputShape => [1, 4];

    /// <summary>Number of inputs to pass to Forward. Default: 2.</summary>
    protected virtual int NumInputs => 2;

    /// <summary>Whether the layer has trainable parameters. Default: false (merge layers typically don't).</summary>
    protected virtual bool ExpectsTrainableParameters => false;

    protected static Tensor<double> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    private Tensor<double>[] CreateInputs(int baseSeed = 42)
    {
        var inputs = new Tensor<double>[NumInputs];
        for (int i = 0; i < NumInputs; i++)
            inputs[i] = CreateRandomTensor(InputShape, baseSeed + i * 100);
        return inputs;
    }

    /// <summary>
    /// Calls the multi-input Forward(params Tensor[]) method on the layer.
    /// The ILayer interface only exposes Forward(Tensor), so we use the
    /// LayerBase.Forward(params Tensor[]) overload via reflection.
    /// </summary>
    protected Tensor<double> ForwardMulti(ILayer<double> layer, Tensor<double>[] inputs)
    {
        var method = layer.GetType().GetMethod("Forward",
            new[] { typeof(Tensor<double>[]) });

        if (method is not null)
        {
            var result = method.Invoke(layer, new object[] { inputs });
            if (result is Tensor<double> tensor)
                return tensor;
        }

        // Fallback: try calling Forward with first input only
        return layer.Forward(inputs[0]);
    }

    // =========================================================================
    // INVARIANT 1: Forward produces finite output
    // =========================================================================

    [Fact]
    public void Forward_ShouldProduceFiniteOutput()
    {
        var layer = CreateLayer();
        var inputs = CreateInputs();

        var output = ForwardMulti(layer, inputs);

        Assert.True(output.Length > 0, "Output should not be empty.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] is Infinity.");
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
        var inputs = CreateInputs();

        var out1 = ForwardMulti(layer, inputs);
        layer.ResetState();
        var out2 = ForwardMulti(layer, inputs);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
    }

    // =========================================================================
    // INVARIANT 3: Different inputs produce different outputs
    // =========================================================================

    [Fact]
    public void Forward_DifferentInputs_ShouldProduceDifferentOutputs()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(false);

        var inputs1 = CreateInputs(42);
        var inputs2 = CreateInputs(999);

        layer.ResetState();
        var output1 = ForwardMulti(layer, inputs1);
        layer.ResetState();
        var output2 = ForwardMulti(layer, inputs2);

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
        Assert.True(anyDifferent, "Layer produces identical output for different inputs.");
    }

    // =========================================================================
    // INVARIANT 4: Backward produces finite gradient
    // =========================================================================

    [Fact]
    public void Backward_ShouldProduceFiniteGradient()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(true);
        var inputs = CreateInputs();

        var output = ForwardMulti(layer, inputs);
        var outputGrad = CreateRandomTensor(output.Shape.ToArray(), seed: 99);

        var inputGrad = layer.Backward(outputGrad);

        Assert.True(inputGrad.Length > 0, "Input gradient should not be empty.");
        for (int i = 0; i < inputGrad.Length; i++)
        {
            Assert.False(double.IsNaN(inputGrad[i]), $"InputGradient[{i}] is NaN.");
            Assert.False(double.IsInfinity(inputGrad[i]), $"InputGradient[{i}] is Infinity.");
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
    }

    // =========================================================================
    // INVARIANT 6: ResetState doesn't break the layer
    // =========================================================================

    [Fact]
    public void ResetState_ShouldNotBreakForward()
    {
        var layer = CreateLayer();
        var inputs = CreateInputs();

        ForwardMulti(layer, inputs);
        layer.ResetState();

        var output = ForwardMulti(layer, inputs);
        Assert.True(output.Length > 0, "Output should not be empty after ResetState.");
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN after ResetState.");
    }

    // =========================================================================
    // INVARIANT 7: Serialize/Deserialize preserves behavior
    // =========================================================================

    [Fact]
    public void Serialize_Deserialize_ShouldPreserveBehavior()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(false);
        var inputs = CreateInputs();
        var originalOutput = ForwardMulti(layer, inputs);

        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            layer.Serialize(writer);

        var layer2 = CreateLayer();
        ms.Position = 0;
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            layer2.Deserialize(reader);

        layer2.SetTrainingMode(false);
        layer2.ResetState();
        var output2 = ForwardMulti(layer2, inputs);

        Assert.Equal(originalOutput.Length, output2.Length);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.True(Math.Abs(originalOutput[i] - output2[i]) < 1e-12,
                $"Output[{i}] differs after serialization: original={originalOutput[i]:G17}, deserialized={output2[i]:G17}");
        }
    }
}
