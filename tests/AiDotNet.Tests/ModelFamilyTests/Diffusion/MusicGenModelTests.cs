using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MusicGenModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, MusicGenModel<double>.MUSICGEN_LATENT_CHANNELS, 16, 16];
    protected override int[] OutputShape => [1, MusicGenModel<double>.MUSICGEN_LATENT_CHANNELS, 16, 16];

    protected override IDiffusionModel<double> CreateModel()
    {
        var unet = new UNetNoisePredictor<double>(
            inputChannels: MusicGenModel<double>.MUSICGEN_LATENT_CHANNELS,
            outputChannels: MusicGenModel<double>.MUSICGEN_LATENT_CHANNELS,
            baseChannels: 16, channelMultipliers: [1, 2],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        var musicVAE = new AudioVAE<double>(
            melChannels: MusicGenModel<double>.MUSICGEN_MEL_CHANNELS,
            latentChannels: MusicGenModel<double>.MUSICGEN_LATENT_CHANNELS,
            baseChannels: 16,
            channelMultipliers: [1, 2],
            numResBlocks: 1,
            seed: 42);

        return new MusicGenModel<double>(unet: unet, musicVAE: musicVAE, seed: 42);
    }
}
