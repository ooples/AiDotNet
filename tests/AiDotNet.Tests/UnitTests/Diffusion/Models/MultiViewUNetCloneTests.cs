using AiDotNet.Diffusion.ThreeD;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

public class MultiViewUNetCloneTests
{
    [Fact]
    public void Clone_CreatesIndependentInstanceWithoutPublicConstructorRebuild()
    {
        var source = new MultiViewUNet<float>(
            inputChannels: 4,
            outputChannels: 4,
            baseChannels: 4,
            numViews: 2,
            contextDim: 0,
            seed: 42);

        var clone = source.Clone();

        Assert.NotSame(source, clone);
        Assert.NotSame(source.BaseUNet, clone.BaseUNet);
        Assert.Equal(source.ParameterCount, clone.ParameterCount);
        // Value equality, not just shape: a Clone() that re-initialized weights would still
        // match on ParameterCount. Comparing the actual parameter values locks in that the
        // trained weights are preserved — the exact class of bug this PR fixes.
        Assert.Equal(source.GetParameters(), clone.GetParameters());
    }
}
