using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087 / #1706: out of the HeavyTimeout lane. The default foundation-scale config is too large for
// the PR gate, so per the #1706 out-of-memory ladder the test caps the latent at 16x16 and injects
// a small same-architecture VideoUNetPredictor and TemporalVAE. Channel counts, context widths,
// patch sizes and scale factors keep the model's defaults; only width and depth shrink.
public class InstructVid2VidModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new InstructVid2VidModel<float>(
            predictor: new VideoUNetPredictor<float>(
                inputChannels: 4, baseChannels: 32, channelMultipliers: new[] { 1, 2 },
                numResBlocks: 1, numHeads: 4, contextDim: 768, inputHeight: 16, inputWidth: 16,
                numTemporalLayers: 1, attentionResolutions: new[] { 1 }, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1, temporalKernelSize: 3,
                causalMode: false, latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
