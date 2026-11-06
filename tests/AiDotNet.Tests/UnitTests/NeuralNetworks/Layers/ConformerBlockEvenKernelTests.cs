using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class ConformerBlockEvenKernelTests
{
    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void SeededDepthwiseConvolutionIgnoresInterleavedInitialization(int rank)
    {
        var shape = rank == 3 ? new[] { 3, 5, 5 } : new[] { 2, 3, 5, 5 };
        var input = new Tensor<double>(shape);
        for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(i * 0.13);
        var first = new DepthwiseSeparableConvolutionalLayer<double>(4, 3, padding: 1) { RandomSeed = 19 };
        var second = new DepthwiseSeparableConvolutionalLayer<double>(4, 3, padding: 1) { RandomSeed = 19 };
        var other = new DepthwiseSeparableConvolutionalLayer<double>(4, 3, padding: 1) { RandomSeed = 23 };
        var expected = first.Forward(input);
        other.Forward(input);
        var actual = second.Forward(input);
        Assert.Equal(first.GetParameters().ToArray(), second.GetParameters().ToArray());
        Assert.Equal(expected.ToArray(), actual.ToArray());
        Assert.False(first.GetParameters().ToArray().SequenceEqual(other.GetParameters().ToArray()));
    }

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
