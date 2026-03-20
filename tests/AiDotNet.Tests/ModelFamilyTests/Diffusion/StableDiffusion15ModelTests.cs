using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StableDiffusion15ModelTests : DiffusionModelTestBase
{
    // Proper latent dimensions matching a scaled-down SD15 architecture
    // Real SD15: [1, 4, 64, 64] latent at 320 base channels
    // Test: [1, 4, 16, 16] latent at 64 base channels — same proportions, real code paths
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<double> CreateModel()
    {
        // Scaled-down SD15 architecture that exercises real code paths:
        // - Proper DiffusionResBlock with GroupNorm → SiLU → Conv3x3 + time conditioning
        // - Attention at resolution levels 1 and 2
        // - Downsample/upsample with correct spatial tracking
        // - 3 resolution levels like real SD15 (not 4 since we're testing)
        var unet = new UNetNoisePredictor<double>(
            inputChannels: 4,
            outputChannels: 4,
            baseChannels: 64,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 1,
            attentionResolutions: [1, 2],
            contextDim: 0,
            numHeads: 4,
            inputHeight: 16,
            seed: 42);

        var vae = new StandardVAE<double>(
            inputChannels: 3,
            latentChannels: 4,
            baseChannels: 32,
            channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 0.18215,
            seed: 42);

        return new StableDiffusion15Model<double>(
            unet: unet,
            vae: vae,
            seed: 42);
    }
}
