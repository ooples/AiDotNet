using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class JEN1ModelTests : DiffusionModelTestBase<float>
{
    // JEN-1's latent is 128-channel (JEN1_LATENT_CHANNELS); the diffusion latent is therefore
    // [1, 128, 16, 16] (= 32768 elements). The injected tiny UNet MUST operate on 128 channels to
    // match — a 4-channel UNet produced a 1024-element output that mismatched the 32768 latent.
    protected override int[] InputShape => [1, 128, 16, 16];
    protected override int[] OutputShape => [1, 128, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
    {
        // Tiny same-architecture UNet — latentChannels (128) stays paper-correct; only base channels /
        // level count / res-blocks shrink so the forward stays within the 120s model-family budget.
        var unet = new UNetNoisePredictor<float>(
            inputChannels: 128, outputChannels: 128,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        return new JEN1Model<float>(unet: unet, seed: 42);
    }
}
