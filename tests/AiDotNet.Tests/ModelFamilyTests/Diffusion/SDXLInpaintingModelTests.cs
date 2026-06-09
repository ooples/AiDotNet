using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SDXLInpaintingModelTests : DiffusionModelTestBase
{
    // SDXLInpaintingModel's paper-scale defaults (baseChannels=320,
    // channelMultipliers=[1,2,4], contextDim=2048) at the test's
    // [1, 4, 64, 64] latent put one UNet forward at ~50 GFLOPs and the
    // full DefaultInferenceSteps=10 denoising loop at >120 s on CI
    // hardware — the exact "Predict timeout" pattern in PR #1543. Same
    // mitigation as FlashDiffusion / SyncDiffusion / SpotDiffusion /
    // MultiDiffusion / StitchDiffusion tests: hand the SDXLInpainting
    // ctor a scaled-down UNet + VAE so Predict completes within the
    // 120 s xUnit cap. The architectural shape (9-channel inpainting
    // input, latent dim, EulerDiscreteScheduler, mask-based variant)
    // is preserved — only the channel widths and depths shrink.
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<double> CreateModel()
    {
        var unet = new UNetNoisePredictor<double>(
            inputChannels: 9,
            outputChannels: 4,
            baseChannels: 64,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 1,
            attentionResolutions: [1, 2],
            contextDim: 2048,
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

        return new SDXLInpaintingModel<double>(
            predictor: unet,
            vae: vae,
            seed: 42);
    }
}
