using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Tests for the named input/output port infrastructure (#1058).
/// </summary>
public class LayerPortTests
{
    [Fact]
    public void DenseLayer_HasSingleInputPort()
    {
        var layer = new DenseLayer<double>(4, 8);
        Assert.Single(layer.InputPorts);
        Assert.Equal("input", layer.InputPorts[0].Name);
    }

    [Fact]
    public void DenseLayer_HasSingleOutputPort()
    {
        var layer = new DenseLayer<double>(4, 8);
        Assert.Single(layer.OutputPorts);
        Assert.Equal("output", layer.OutputPorts[0].Name);
    }

    [Fact]
    public void DenseLayer_MultiInputForward_DelegatesToSingleInput()
    {
        var layer = new DenseLayer<double>(4, 2);
        var input = new Tensor<double>([1, 4]);
        for (int i = 0; i < 4; i++) input[0, i] = (i + 1) * 0.1;

        var singleResult = layer.Forward(input);

        // Reset layer state
        layer.ResetState();

        var multiResult = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["input"] = input
        });

        Assert.Equal(singleResult.Length, multiResult.Length);
    }

    [Fact]
    public void AddLayer_HasTwoInputPorts()
    {
        var layer = new AddLayer<double>(new int[][] { new[] { 4 }, new[] { 4 } }, (IActivationFunction<double>?)null);
        Assert.Equal(2, layer.InputPorts.Count);
        Assert.Equal("input_a", layer.InputPorts[0].Name);
        Assert.Equal("input_b", layer.InputPorts[1].Name);
    }

    [Fact]
    public void AddLayer_MultiInputForward_AddsCorrectly()
    {
        var layer = new AddLayer<double>(new int[][] { new[] { 3 }, new[] { 3 } }, (IActivationFunction<double>?)null);
        var a = new Tensor<double>([3], new Vector<double>([1.0, 2.0, 3.0]));
        var b = new Tensor<double>([3], new Vector<double>([4.0, 5.0, 6.0]));

        var result = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["input_a"] = a,
            ["input_b"] = b
        });

        Assert.Equal(5.0, result[0], 10);
        Assert.Equal(7.0, result[1], 10);
        Assert.Equal(9.0, result[2], 10);
    }

    [Fact]
    public void DiffusionResBlock_HasTimeEmbedPort_WhenConfigured()
    {
        var block = new AiDotNet.Diffusion.NoisePredictors.DiffusionResBlock<double>(
            inChannels: 4, outChannels: 4, spatialSize: 8, timeEmbedDim: 64);

        Assert.Equal(2, block.InputPorts.Count);
        Assert.Equal("input", block.InputPorts[0].Name);
        Assert.Equal("time_embed", block.InputPorts[1].Name);
        Assert.Equal(64, block.InputPorts[1].Shape[0]);
    }

    [Fact]
    public void DiffusionResBlock_SinglePort_WhenNoTimeEmbed()
    {
        var block = new AiDotNet.Diffusion.NoisePredictors.DiffusionResBlock<double>(
            inChannels: 4, outChannels: 4, spatialSize: 8, timeEmbedDim: 0);

        Assert.Single(block.InputPorts);
        Assert.Equal("input", block.InputPorts[0].Name);
    }

    [Fact]
    public void LayerPort_Record_HasCorrectProperties()
    {
        var port = new LayerPort("query", [8, 64], Required: true);

        Assert.Equal("query", port.Name);
        Assert.Equal(2, port.Shape.Length);
        Assert.Equal(8, port.Shape[0]);
        Assert.Equal(64, port.Shape[1]);
        Assert.True(port.Required);
    }

    [Fact]
    public void LayerPort_OptionalPort()
    {
        var port = new LayerPort("mask", [8, 8], Required: false);
        Assert.False(port.Required);
    }
}
