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
    }
}
