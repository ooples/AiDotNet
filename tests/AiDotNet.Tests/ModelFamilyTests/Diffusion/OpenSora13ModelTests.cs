using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): correct but too slow for the default per-test gate (foundation-scale diffusion,
// ~100 s/forward x N-step Generate); runs in the nightly lane. Drop this trait once it fits the budget.
[Xunit.Trait("Category", "HeavyTimeout")]
public class OpenSora13ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    // Open-Sora 1.3 defaults to a foundation-scale STDiT (hidden 1152, 28 layers, contextDim 4096) +
    // a 128-base-channel temporal VAE: a single forward exceeds the 120s model-family test budget.
    // Inject a tiny same-architecture DiT + TemporalVAE so the invariants run against a REAL instance
    // at a runnable scale — latentChannels (4) and contextDim (4096) stay paper-correct; only the
    // hidden width / depth / VAE base channels are shrunk, keeping the model below the budget.
    protected override IDiffusionModel<float> CreateModel()
        => new OpenSora13Model<float>(
            predictor: new DiTNoisePredictor<float>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, latentSpatialSize: 32, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1,
                temporalKernelSize: 3, latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
