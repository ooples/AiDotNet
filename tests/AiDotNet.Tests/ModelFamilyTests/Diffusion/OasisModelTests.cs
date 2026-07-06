using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.WorldModels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
// HeavyTimeout: foundation-scale diffusion (video / paper-scale); correct but a single
// forward x N-step Generate exceeds the 120 s per-test gate. Runs in the nightly lane.
[Xunit.Trait("Category", "HeavyTimeout")]
public class OasisModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    // Build the DiT predictor + temporal VAE at a REDUCED scale instead of the default (DiT hiddenSize
    // 1536 x 24 layers + baseChannels-128 temporal VAE), which peaks ~50 GB and stalls the shard. Shape-
    // critical dims preserved (inputChannels = LATENT_CHANNELS 16, contextDim 4096, latentSpatialSize 32).
    protected override IDiffusionModel<float> CreateModel()
        => new OasisModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.DiTNoisePredictor<float>(
                inputChannels: 16, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, latentSpatialSize: 32, seed: 42),
            temporalVAE: new AiDotNet.Diffusion.VAE.TemporalVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1, temporalKernelSize: 3,
                causalMode: true, latentScaleFactor: 0.13025),
            seed: 42);
}
