using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

/// <summary>
/// Tests for multi-input merge layers (AddLayer, ConcatenateLayer, MultiplyLayer).
/// These layers require multiple inputs so they can't use the single-input LayerTestBase.
/// Each test directly exercises the multi-input Forward API.
/// </summary>
public class AddLayerMultiInputTests
{
    [Fact]
    public void Forward_ShouldProduceFiniteOutput()
    {
        var layer = new AddLayer<double>(inputShapes: [[4], [4]],
            activationFunction: new IdentityActivation<double>() as IActivationFunction<double>);
        var input1 = new Tensor<double>([1, 4]);
        var input2 = new Tensor<double>([1, 4]);
        for (int i = 0; i < 4; i++) { input1[i] = 0.5; input2[i] = 0.3; }

        var output = layer.Forward(input1, input2);

        Assert.True(output.Length > 0);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] is Infinity");
        }
    }

    [Fact]
    public void Forward_ShouldAddInputsElementwise()
    {
        var layer = new AddLayer<double>(inputShapes: [[4], [4]],
            activationFunction: new IdentityActivation<double>() as IActivationFunction<double>);
        var input1 = new Tensor<double>([1, 4]);
        var input2 = new Tensor<double>([1, 4]);
        for (int i = 0; i < 4; i++) { input1[i] = 1.0; input2[i] = 2.0; }

        var output = layer.Forward(input1, input2);

        for (int i = 0; i < output.Length; i++)
            Assert.True(Math.Abs(output[i] - 3.0) < 1e-10, $"Expected 3.0 at [{i}], got {output[i]}");
    }
}

public class ConcatenateLayerMultiInputTests
{
    [Fact]
    public void Forward_ShouldProduceFiniteOutput()
    {
        var layer = new ConcatenateLayer<double>(inputShapes: [[2], [2]], axis: 0,
            activationFunction: new IdentityActivation<double>() as IActivationFunction<double>);
        var input1 = new Tensor<double>([1, 2]);
        var input2 = new Tensor<double>([1, 2]);
        for (int i = 0; i < 2; i++) { input1[i] = 0.5; input2[i] = 0.3; }

        var output = layer.Forward(input1, input2);

        Assert.True(output.Length > 0);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] is Infinity");
        }
    }
}
