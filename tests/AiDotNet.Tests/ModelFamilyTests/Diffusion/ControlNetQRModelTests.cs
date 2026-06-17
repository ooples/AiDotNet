using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Control;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class ControlNetQRModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    // ControlNet-QR wraps a full SD1.5 UNet (320 base, [1,2,4,4]) + a mirrored control branch + VAE:
    // a single forward over the 64x64 latent exceeds the 120s model-family budget. Inject a tiny
    // same-architecture base UNet + VAE — latentChannels (4) and contextDim (768, SD1.5) stay
    // paper-correct; only base channels / level count / res-blocks shrink. The control branch derives
    // its width from the injected base UNet, so it shrinks to match.
    protected override IDiffusionModel<float> CreateModel()
        => new ControlNetQRModel<float>(
            baseUNet: new UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, contextDim: 768, numHeads: 4,
                inputHeight: 64, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215, seed: 42),
            seed: 42);
}
