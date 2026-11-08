using System;
using AiDotNet.Models.Options;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

public class VideoOptionsCopyTests
{
    [Theory]
    [InlineData(typeof(InterpAnyClearerOptions))]
    [InlineData(typeof(SeedVROptions))]
    [InlineData(typeof(VFITOptions))]
    [InlineData(typeof(DAMVSROptions))]
    [InlineData(typeof(FlashVSROptions))]
    [InlineData(typeof(BiMVFIOptions))]
    [InlineData(typeof(ABMEOptions))]
    [InlineData(typeof(DOVEOptions))]
    [InlineData(typeof(FLAVROptions))]
    [InlineData(typeof(M2MOptions))]
    [InlineData(typeof(MGLDVSROptions))]
    [InlineData(typeof(StreamDiffVSROptions))]
    [InlineData(typeof(AMTOptions))]
    [InlineData(typeof(VFIformerOptions))]
    [InlineData(typeof(EMAVFIOptions))]
    [InlineData(typeof(IFRNetOptions))]
    [InlineData(typeof(RealisVSROptions))]
    [InlineData(typeof(XVFIOptions))]
    [InlineData(typeof(SoftSplatOptions))]
    [InlineData(typeof(Upscale4KAgentOptions))]
    public void CopyConstructorPreservesInheritedConfiguration(Type optionsType)
    {
        var source = (NeuralNetworkOptions)Activator.CreateInstance(optionsType)!;
        source.Seed = 7;
        source.EncoderLayerCount = 3;
        var copy = (NeuralNetworkOptions)Activator.CreateInstance(optionsType, source)!;
        Assert.Equal(7, copy.Seed);
        Assert.Equal(3, copy.EncoderLayerCount);
    }
}
