using System;
using AiDotNet.Models.Options;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

public class VideoOptionsCopyTests
{
    [Theory]
    [InlineData(typeof(InterpAnyClearerOptions))]
    [InlineData(typeof(BSVDOptions))]
    [InlineData(typeof(BasicVSROptions))]
    [InlineData(typeof(DualXVSROptions))]
    [InlineData(typeof(DynamiCrafterOptions))]
    [InlineData(typeof(FiganOptions))]
    [InlineData(typeof(FloRNNOptions))]
    [InlineData(typeof(GIMMVFIOptions))]
    [InlineData(typeof(IARTOptions))]
    [InlineData(typeof(IQVFIOptions))]
    [InlineData(typeof(IconVSROptions))]
    [InlineData(typeof(LiteDVDNetOptions))]
    [InlineData(typeof(MIAVSROptions))]
    [InlineData(typeof(McnetOptions))]
    [InlineData(typeof(MoGOptions))]
    [InlineData(typeof(MoMoOptions))]
    [InlineData(typeof(PSRTOptions))]
    [InlineData(typeof(PerVFIOptions))]
    [InlineData(typeof(RVRTOptions))]
    [InlineData(typeof(RealBasicVSROptions))]
    [InlineData(typeof(RealBasicVSRSharpOptions))]
    [InlineData(typeof(RealESRGANVideoOptions))]
    [InlineData(typeof(RealViformerOptions))]
    [InlineData(typeof(STMFNetOptions))]
    [InlineData(typeof(ShiftNetOptions))]
    [InlineData(typeof(StableVideoSROptions))]
    [InlineData(typeof(SwinVFIOptions))]
    [InlineData(typeof(TLBVFIOptions))]
    [InlineData(typeof(TTVSROptions))]
    [InlineData(typeof(ToonCrafterOptions))]
    [InlineData(typeof(UDVDOptions))]
    [InlineData(typeof(UPRNetOptions))]
    [InlineData(typeof(VFIMambaOptions))]
    [InlineData(typeof(VideoGigaGANOptions))]

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
    public void CopyPreservesInheritedConfiguration(Type optionsType)
    {
        var source = (ModelOptions)Activator.CreateInstance(optionsType)!;
        source.Seed = 7;
        var constructor = optionsType.GetConstructor(new[] { optionsType });
        var copy = (ModelOptions)(constructor is null ? AiDotNet.Models.CloneEngine.CopyConfiguration(source)
            : constructor.Invoke(new object[] { source }));
        Assert.NotSame(source, copy);
        Assert.Equal(7, copy.Seed);
    }
}
