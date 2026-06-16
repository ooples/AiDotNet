using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class MusicGenModelTests : DiffusionModelTestBase<float>
{
    // MusicGen operates on a 16-channel audio latent (MusicGenModel.MUSICGEN_LATENT_CHANNELS = 16).
    // The denoising loop tracks a LatentChannels-deep sample, so a 4-channel noise predictor was
    // paper-incorrect: LatentDiffusionModelBase canonicalizes the input to [1,16,16,16] (4096) while
    // the 4-channel U-Net emits [1,4,16,16] (1024), tripping "PredictNoise output length (1024) does
    // not match the latent/sample length (4096)". The U-Net channels and the input shape must both
    // match the model's latent channel count (same correction as AudioLDM2).
    private const int LatentChannels = MusicGenModel<float>.MUSICGEN_LATENT_CHANNELS;

    protected override int[] InputShape => [1, LatentChannels, 16, 16];
    protected override int[] OutputShape => [1, LatentChannels, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
    {
        var unet = new UNetNoisePredictor<float>(
            inputChannels: LatentChannels, outputChannels: LatentChannels,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        return new MusicGenModel<float>(unet: unet, seed: 42);
    }
}
