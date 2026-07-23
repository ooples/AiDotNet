using AiDotNet.Interfaces;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class DDPMModelTests : DiffusionModelTestBase<float>
{
    // Per Ho et al. 2020 Table 1: CIFAR-10 images are 32×32×3
    // UNet operates directly on pixel space (no VAE)
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [1, 3, 32, 32];

    // DDPM defaults to the paper UNet (base 128, [1,2,2,2], 2 ResBlocks): a single forward over the
    // 32×32 pixel space exceeds the 120s model-family budget. Inject a tiny same-architecture UNet —
    // channels (3, pixel-space) and contextDim (0, unconditional) stay paper-correct; only base channels
    // / level count / res-blocks shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new DDPMModel<float>(
            unet: new UNetNoisePredictor<float>(
                inputChannels: 3, outputChannels: 3, baseChannels: 32,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, contextDim: 0, numHeads: 4,
                inputHeight: 32, seed: 42));
}
