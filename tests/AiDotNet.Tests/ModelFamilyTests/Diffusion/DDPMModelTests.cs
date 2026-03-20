using AiDotNet.Interfaces;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class DDPMModelTests : DiffusionModelTestBase
{
    // Per Ho et al. 2020: CIFAR-10 images are 32×32×3
    // UNet operates directly on pixel space (no VAE)
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [1, 3, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
    {
        // Per Ho et al. 2020 Table 1: CIFAR-10 architecture
        // Channel multipliers [1, 2, 2, 2], base channels 128,
        // 2 ResBlocks per level, attention at 16×16
        // Using scaled-down version (base=64) for CPU testing
        // but same architecture proportions as the paper
        var unet = new UNetNoisePredictor<double>(
            inputChannels: 3,
            outputChannels: 3,
            baseChannels: 64,
            channelMultipliers: [1, 2, 2, 2],
            numResBlocks: 2,
            attentionResolutions: [1],
            contextDim: 0,
            numHeads: 4,
            inputHeight: 32,
            seed: 42);

        return new DDPMModel<double>(
            unet: unet,
            seed: 42);
    }
}
