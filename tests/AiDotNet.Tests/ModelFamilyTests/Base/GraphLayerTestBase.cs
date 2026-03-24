using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for graph/mesh layers that require setup before Forward
/// (adjacency matrices, Laplacians, eigenbases, spiral indices, etc.).
/// Subclasses override SetupLayer() to provide domain-specific initialization.
/// </summary>
public abstract class GraphLayerTestBase
{
    protected abstract ILayer<double> CreateLayer();

    /// <summary>
    /// Perform domain-specific setup on the layer (set adjacency matrix, Laplacian, etc.).
    /// Called before every test's Forward pass.
    /// </summary>
    protected abstract void SetupLayer(ILayer<double> layer);

    /// <summary>Shape of the input tensor. Default: [4, 8] (4 nodes, 8 features).</summary>
    protected virtual int[] InputShape => [4, 8];

    /// <summary>Whether the layer has trainable parameters. Default: true.</summary>
    protected virtual bool ExpectsTrainableParameters => true;

    protected static Tensor<double> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 4.0 + 1.0; // [1.0, 5.0] — large positive to survive ReLU after matmul
        return tensor;
    }

    private ILayer<double> CreateAndSetup()
    {
        var layer = CreateLayer();
        SetupLayer(layer);
        return layer;
    }

    // =========================================================================
    // INVARIANT 1: Forward produces finite output
    // =========================================================================

    [Fact]
    public void Forward_ShouldProduceFiniteOutput()
    {
        var layer = CreateAndSetup();
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);

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
        var layer = CreateAndSetup();
        layer.SetTrainingMode(false);
        var input = CreateRandomTensor(InputShape);

        var out1 = layer.Forward(input);
        layer.ResetState();
        var out2 = layer.Forward(input);

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
        var layer = CreateAndSetup();
        layer.SetTrainingMode(false);

        var input1 = CreateRandomTensor(InputShape, seed: 1);
        var input2 = CreateRandomTensor(InputShape, seed: 2);

        layer.ResetState();
        var output1 = layer.Forward(input1);
        layer.ResetState();
        var output2 = layer.Forward(input2);

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
        var layer = CreateAndSetup();
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);
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
        var layer = CreateAndSetup();
        int count = layer.ParameterCount;
        var parameters = layer.GetParameters();

        Assert.True(count >= 0, "ParameterCount should be non-negative.");
        Assert.Equal(count, parameters.Length);

        if (ExpectsTrainableParameters)
            Assert.True(count > 0, "Layer should have trainable parameters.");
    }

    // =========================================================================
    // INVARIANT 6: ResetState doesn't break the layer
    // =========================================================================

    [Fact]
    public void ResetState_ShouldNotBreakForward()
    {
        var layer = CreateAndSetup();
        var input = CreateRandomTensor(InputShape);

        layer.Forward(input);
        layer.ResetState();

        var output = layer.Forward(input);
        Assert.True(output.Length > 0);
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN after ResetState.");
    }
}
