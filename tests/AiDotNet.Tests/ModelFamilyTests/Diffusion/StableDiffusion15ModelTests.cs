using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StableDiffusion15ModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 4, 8, 8];
    protected override int[] OutputShape => [1, 4, 8, 8];

    protected override IDiffusionModel<double> CreateModel()
    {
        // Use tiny dimensions for testing to avoid OOM with full 320-channel model
        var tinyUnet = new UNetNoisePredictor<double>(
            inputChannels: 4,
            outputChannels: 4,
            baseChannels: 16,
            channelMultipliers: [1, 2],
            numResBlocks: 1,
            attentionResolutions: [1],
            contextDim: 0,
            numHeads: 2,
            inputHeight: 8,
            seed: 42);

        var tinyVae = new StandardVAE<double>(
            inputChannels: 3,
            latentChannels: 4,
            baseChannels: 8,
            channelMultipliers: [1, 2],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 0.18215,
            seed: 42);

        return new StableDiffusion15Model<double>(
            unet: tinyUnet,
            vae: tinyVae,
            seed: 42);
    }
}
