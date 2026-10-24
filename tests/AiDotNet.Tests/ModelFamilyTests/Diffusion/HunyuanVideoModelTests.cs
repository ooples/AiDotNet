using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087: out of the HeavyTimeout lane. At its default config (a 3072-wide, 40-layer DiT and a
// 128-base temporal VAE) the model ran out of memory on the nightly runner (run 39554054058). Per
// the #1706 out-of-memory ladder the test caps the latent at 16x16 and injects a small
// same-architecture DiT and temporal VAE: the latent channels (16), text context width (4096),
// patch size (2) and latent scale factor stay paper-correct, and only width and depth shrink.
public class HunyuanVideoModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 16, 16];
    protected override int[] OutputShape => [1, 16, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new HunyuanVideoModel<float>(
            dit: new DiTNoisePredictor<float>(
                inputChannels: 16, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, latentSpatialSize: 16, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1,
                temporalKernelSize: 3, latentScaleFactor: 0.13025, seed: 42),
            seed: 42);
}
