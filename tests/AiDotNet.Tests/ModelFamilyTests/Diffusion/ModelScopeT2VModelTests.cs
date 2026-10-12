using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #1714: back in the default gate. The HeavyTimeout deferral here was measured away: in the
// nightly lane (run 38117342722, one class per process under a managed heap cap of 70% of
// runner RAM) this class ran 13 of 13 tests green, slowest 4.0 s and no OutOfMemoryException
// -- far inside the 120 s per-test gate. Re-defer only against a fresh measurement.
public class ModelScopeT2VModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    // ModelScope-T2V defaults to a 320-base-channel 3D-UNet ([1,2,4,4] levels) + 128-base VAE:
    // a single forward exceeds the 120s model-family budget. Inject a tiny same-architecture
    // VideoUNet + VAE — latentChannels (4), crossAttentionDim (1024) and the no-image-conditioning
    // flag stay paper-correct; only base channels / level count / res-blocks shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new ModelScopeT2VModel<float>(
            videoUNet: new VideoUNetPredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, numTemporalLayers: 1,
                contextDim: 1024, numHeads: 8, supportsImageConditioning: false,
                inputHeight: 16, inputWidth: 16, numFrames: 16, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
