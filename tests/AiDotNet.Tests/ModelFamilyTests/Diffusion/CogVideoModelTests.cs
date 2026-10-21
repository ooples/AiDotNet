using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087 / #1706: out of the HeavyTimeout lane. The default foundation-scale config is too large for
// the PR gate, so per the #1706 out-of-memory ladder the test caps the latent at 16x16 and injects
// a small same-architecture VideoUNetPredictor and TemporalVAE. Channel counts, context widths,
// patch sizes and scale factors keep the model's defaults; only width and depth shrink.
public class CogVideoModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 16, 16];
    protected override int[] OutputShape => [1, 16, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new CogVideoModel<float>(
            videoUnet: new VideoUNetPredictor<float>(
                inputChannels: 16, baseChannels: 32, contextDim: 4096, numTemporalLayers: 1,
                channelMultipliers: new[] { 1, 2 }, inputHeight: 16, inputWidth: 16,
                numResBlocks: 1, numHeads: 4, attentionResolutions: new[] { 1 }, seed: 42),
            temporalVae: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16, numTemporalLayers: 1,
                causalMode: true, channelMultipliers: new[] { 1, 2 }, seed: 42),
            seed: 42);
}
