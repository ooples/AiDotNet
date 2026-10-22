using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087 / #1706: out of the HeavyTimeout lane. The default foundation-scale config is too large for
// the PR gate, so per the #1706 out-of-memory ladder the test caps the latent at 16x16 and injects
// a small same-architecture DiTNoisePredictor and TemporalVAE. Channel counts, context widths,
// patch sizes and scale factors keep the model's defaults; only width and depth shrink.
public class MinimaxVideoModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 16, 16];
    protected override int[] OutputShape => [1, 16, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new MinimaxVideoModel<float>(
            predictor: new DiTNoisePredictor<float>(
                inputChannels: 16, hiddenSize: 64, numLayers: 2, numHeads: 2, patchSize: 2,
                contextDim: 4096, latentSpatialSize: 16, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1, temporalKernelSize: 3,
                causalMode: true, latentScaleFactor: 0.13025, seed: 42),
            seed: 42);
}
