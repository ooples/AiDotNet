using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The fused training step's lazy-initialization forward (which only makes lazily-sized weights exist) runs on one sample
/// when the input and target agree the leading axis is a batch of more than one. Its result is discarded, and at full
/// batch it paged in a whole inference forward. A leading axis the target does not share, a single sample, a rank-1
/// input or a missing target keep the whole input. The sample is the first one.
/// </summary>
public class LazyInitializationSampleTests
{
    private static Tensor<float> Counting(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = i;
        return t;
    }

    [Fact]
    public void ABatch_TheTargetShares_IsCutToItsFirstSample()
    {
        var input = Counting(4, 3, 2, 2);
        var sample = CompiledTapeTrainingStep<float>.LazyInitializationSample(input, new Tensor<float>(new[] { 4, 10 }));
        Assert.Equal(new[] { 1, 3, 2, 2 }, sample.Shape.ToArray());
        for (int i = 0; i < sample.Length; i++) Assert.Equal(input[i], sample[i]);
    }

    [Theory]
    [InlineData(new[] { 5, 4, 3 }, new[] { 4, 2 })]   // time-major: the target's leading axis is the batch, not the input's
    [InlineData(new[] { 1, 3, 2, 2 }, new[] { 1, 10 })] // already one sample
    [InlineData(new[] { 6 }, new[] { 6 })]              // rank 1: no batch axis to cut
    public void OtherwiseTheWholeInputIsKept(int[] inputShape, int[] targetShape)
    {
        var input = Counting(inputShape);
        Assert.Same(input, CompiledTapeTrainingStep<float>.LazyInitializationSample(input, new Tensor<float>(targetShape)));
    }

    [Fact]
    public void WithoutATarget_TheWholeInputIsKept()
    {
        var input = Counting(4, 3);
        Assert.Same(input, CompiledTapeTrainingStep<float>.LazyInitializationSample(input, null));
    }
}
