using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public sealed class BiasFreeLinearLayerTests
{
    [Fact]
    public void Forward_RankOneInput_PreservesTheBatchOptionalShapeContract()
    {
        var layer = new BiasFreeLinearLayer<double>(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>([2.0, -3.0]));
        var input = new Tensor<double>([4.0, 5.0], [2]);

        Tensor<double> output = layer.Forward(input);

        Assert.Equal(new[] { 1 }, output.Shape.ToArray());
        Assert.Equal(-7.0, output[0], precision: 12);
    }

    [Fact]
    public void Forward_HigherRankInput_ProjectsTheLastAxisAndRestoresLeadingAxes()
    {
        var layer = new BiasFreeLinearLayer<double>(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>([2.0, -3.0]));
        var input = new Tensor<double>(
            [1.0, 2.0, 3.0, 4.0, -1.0, 2.0, 0.5, -0.5],
            [2, 2, 2]);

        Tensor<double> output = layer.Forward(input);

        Assert.Equal(new[] { 2, 2, 1 }, output.Shape.ToArray());
        Assert.Equal(new[] { -4.0, -6.0, -8.0, 2.5 }, output.ToArray());
    }
}
