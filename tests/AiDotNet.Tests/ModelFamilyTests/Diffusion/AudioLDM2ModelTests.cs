using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class AudioLDM2ModelTests : DiffusionModelTestBase
{
    // AudioLDM2 (Liu et al. 2024) operates on an 8-channel mel-VAE latent
    // — see AudioLDM2Model.AUDIOLDM2_LATENT_CHANNELS / HF audioldm2 config
    // `latent_channels=8`. The UNet input/output channel count MUST match,
    // otherwise LatentDiffusionModelBase canonicalizes the sample to 8 ch
    // and the 4-ch UNet emits half the elements, tripping
    // "PredictNoise output length (1024) does not match the latent/sample
    // length (2048)" on every test that drives a denoising step.
    protected override int[] InputShape => [1, 8, 16, 16];
    protected override int[] OutputShape => [1, 8, 16, 16];

    protected override IDiffusionModel<double> CreateModel()
    {
        var unet = new UNetNoisePredictor<double>(
            inputChannels: 8, outputChannels: 8,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        return new AudioLDM2Model<double>(unet: unet, seed: 42);
    }
}
