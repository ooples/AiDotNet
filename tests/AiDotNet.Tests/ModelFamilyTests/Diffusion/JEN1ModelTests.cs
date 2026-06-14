using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class JEN1ModelTests : DiffusionModelTestBase
{
    private const int Jen1LatentChannels = 128;
    private const int Jen1MelChannels = 128;

    protected override int[] InputShape => [1, Jen1LatentChannels, 16, 16];
    protected override int[] OutputShape => [1, Jen1LatentChannels, 16, 16];

    protected override IDiffusionModel<double> CreateModel()
    {
        var unet = new UNetNoisePredictor<double>(
            inputChannels: Jen1LatentChannels, outputChannels: Jen1LatentChannels,
            baseChannels: 16, channelMultipliers: [1, 2],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        var audioVae = new AudioVAE<double>(
            melChannels: Jen1MelChannels,
            latentChannels: Jen1LatentChannels,
            baseChannels: 16,
            channelMultipliers: [1, 2],
            numResBlocks: 1,
            seed: 42);

        return new JEN1Model<double>(unet: unet, audioVae: audioVae, seed: 42);
    }
}
