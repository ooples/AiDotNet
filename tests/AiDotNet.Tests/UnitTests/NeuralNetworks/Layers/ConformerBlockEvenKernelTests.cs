using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class ConformerBlockEvenKernelTests
{
    [Theory]
    [InlineData(2, 16, false)]
    [InlineData(3, 16, false)]
    [InlineData(2, 32, false)]
    [InlineData(3, 32, false)]
    [InlineData(2, 15, false)]
    [InlineData(3, 32, true)]
    public void Convolution_PreservesTimeLength(int rank, int kernel, bool causal)
    {
        var shape = rank == 2 ? new[] { 6, 8 } : new[] { 2, 6, 8 };
        var input = new Tensor<double>(shape);
        for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(i * 0.13);
        var layer = new ConformerBlockLayer<double>(8, 2, convKernelSize: kernel, causalConvolution: causal);
        var output = layer.Forward(input);
        Assert.Equal(shape, output.Shape.ToArray());
        Assert.All(output.ToArray(), value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
    }
}
