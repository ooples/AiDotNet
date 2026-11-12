using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using Xunit;

namespace AiDotNet.Tests.Video;

public sealed class UnsharpKernelTests
{
    [Fact]
    public void EvenKernelUsesEveryTapWithReflectedBorders()
    {
        var image = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { .2, .4, .6, .8 }));
        var result = RealBasicVsrCore<double>.UnsharpMask(image, .2, 2, 0);
        var expected = new[] { .14, .38, .62, .86 };
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], result[i], 12);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(51)]
    [InlineData(52)]
    public void ConstantImageIsUnchangedForOddAndEvenKernels(int size)
    {
        var image = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(Enumerable.Repeat(.4, 6)));
        var result = RealBasicVsrCore<double>.UnsharpMask(image, .5, size, 10);
        for (int i = 0; i < result.Length; i++) Assert.Equal(.4, result[i], 12);
    }
}
