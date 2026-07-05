using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// HeavyTimeout: foundation-scale diffusion (video / paper-scale); correct but a single
// forward x N-step Generate exceeds the 120 s per-test gate. Runs in the nightly lane.
[Xunit.Trait("Category", "HeavyTimeout")]
public class Mochi1PreviewModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 12, 32, 32];
    protected override int[] OutputShape => [1, 12, 32, 32];

    // Mochi-1 Preview defaults to a foundation-scale AsymmDiT (contextDim 4096) over a 12-channel
    // latent + temporal VAE: a single forward exceeds the 120s model-family budget. Inject a tiny
    // same-architecture DiT + TemporalVAE — latentChannels (12) and contextDim (4096) stay paper-correct;
    // only hidden width / depth / VAE base channels shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new Mochi1PreviewModel<float>(
            predictor: new DiTNoisePredictor<float>(
                inputChannels: 12, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, latentSpatialSize: 32, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 12, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1,
                temporalKernelSize: 3, latentScaleFactor: 0.18215, seed: 42));
}
