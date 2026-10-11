using AiDotNet.Interfaces;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// #1714: back in the default gate. The HeavyTimeout deferral here was measured away: in the
// nightly lane (run 38117342722, one class per process under a managed heap cap of 70% of
// runner RAM) this class ran 13 of 13 tests green, slowest 3.0 s and no OutOfMemoryException
// -- far inside the 120 s per-test gate. Re-defer only against a fresh measurement.
public class SUPIRModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
    {
        var unet = new UNetNoisePredictor<float>(
            inputChannels: 4, outputChannels: 4,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        var vae = new StandardVAE<float>(
            inputChannels: 3, latentChannels: 4,
            baseChannels: 32, channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 42);

        return new SUPIRModel<float>(unet: unet, vae: vae, seed: 42);
    }
}
