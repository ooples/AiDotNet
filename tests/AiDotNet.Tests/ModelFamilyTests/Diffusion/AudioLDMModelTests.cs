using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class AudioLDMModelTests : DiffusionModelTestBase
{
    // AudioLDM (Liu et al. 2023, Table 7) operates in an 8-channel latent
    // space — DOUBLE the 4-channel latent that image diffusion (Stable
    // Diffusion) uses, to capture audio's temporal complexity. The model's
    // AUDIOLDM_LATENT_CHANNELS constant and LatentDiffusionModelBase.Generate
    // both rely on this. Using 4 channels here was paper-incorrect and
    // caused LatentDiffusionModelBase.Generate to reshape the input to
    // [1, 8, ...] before handing it to a UNet expecting 4, throwing
    // "Expected input depth 4, but got 8" during forward.
    private const int LatentChannels = AudioLDMModel<double>.AUDIOLDM_LATENT_CHANNELS;

    protected override int[] InputShape => [1, LatentChannels, 16, 16];
    protected override int[] OutputShape => [1, LatentChannels, 16, 16];

    protected override IDiffusionModel<double> CreateModel()
    {
        var unet = new UNetNoisePredictor<double>(
            inputChannels: LatentChannels, outputChannels: LatentChannels,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        return new AudioLDMModel<double>(unet: unet, seed: 42);
    }
}
