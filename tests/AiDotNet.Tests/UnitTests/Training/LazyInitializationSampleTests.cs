using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The fused training step's lazy-initialization forward (which only makes lazily-sized weights exist) runs on one sample
/// when the caller declared the leading axis the batch and the input and target agree it holds more than one. Its result
/// is discarded, and at full batch it paged in a whole inference forward. An undeclared leading axis (a model that owns
/// its input convention, such as an unbatched [C, H, W] image whose target happens to start with C), a leading axis the
/// target does not share, a single sample, a rank-1 input or a missing target keep the whole input. The sample is the
/// first one, and the declaration lasts exactly as long as its scope.
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
    public void ADeclaredBatch_TheTargetShares_IsCutToItsFirstSample()
    {
        var input = Counting(4, 3, 2, 2);
        var sample = CompiledTapeTrainingStep<float>.LazyInitializationSample(input, new Tensor<float>(new[] { 4, 10 }), true);
        Assert.Equal(new[] { 1, 3, 2, 2 }, sample.Shape.ToArray());
        for (int i = 0; i < sample.Length; i++) Assert.Equal(input[i], sample[i]);
    }

    [Fact]
    public void AnUnbatchedImage_WhoseTargetStartsWithItsChannels_IsNotCut()
    {
        // [3, H, W] with a [3, ...] target: the leading axis is channels. Cutting it would initialize the first
        // convolution with one input channel, and the full three-channel forward after it would throw.
        var input = Counting(3, 4, 4);
        Assert.Same(input, CompiledTapeTrainingStep<float>.LazyInitializationSample(input, new Tensor<float>(new[] { 3, 5 }), false));
    }

    [Theory]
    [InlineData(new[] { 5, 4, 3 }, new[] { 4, 2 })]   // time-major: the target's leading axis is the batch, not the input's
    [InlineData(new[] { 1, 3, 2, 2 }, new[] { 1, 10 })] // already one sample
    [InlineData(new[] { 6 }, new[] { 6 })]              // rank 1: no batch axis to cut
    public void OtherwiseTheWholeInputIsKept(int[] inputShape, int[] targetShape)
    {
        var input = Counting(inputShape);
        Assert.Same(input, CompiledTapeTrainingStep<float>.LazyInitializationSample(input, new Tensor<float>(targetShape), true));
    }

    [Fact]
    public void WithoutATarget_TheWholeInputIsKept()
    {
        var input = Counting(4, 3);
        Assert.Same(input, CompiledTapeTrainingStep<float>.LazyInitializationSample(input, null, true));
    }

    [Fact]
    public void TheDeclarationLastsForItsScope_AndADefaultScopeDeclaresNothing()
    {
        var input = Counting(4, 3);
        var target = new Tensor<float>(new[] { 4, 2 });
        bool Declared() => !ReferenceEquals(input, CompiledTapeTrainingStep<float>.LazyInitializationSample(input, target,
            CompiledTapeTrainingStep<float>.LeadingAxisIsBatchOnThisThread));

        Assert.False(Declared());
        using (CompiledTapeTrainingStep<float>.DeclareLeadingAxisIsBatch())
        {
            Assert.True(Declared());
            using (default(CompiledTapeTrainingStep<float>.BatchAxisScope)) { }
            Assert.True(Declared());
        }
        Assert.False(Declared());
    }
}
