using AiDotNet.Interfaces;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.VirtualTryOn;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// CatVTON test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. CatVTON's production default is the full
/// SD-1.5-inpainting UNet (baseChannels 320, multipliers [1,2,4,4], ~860M
/// params): a single Predict is a multi-step reverse-sampling loop over
/// ~15s/forward UNet passes on CPU, which hangs the shared diffusion-test
/// process (host inactivity dump) and OOMs alongside parallel SD-scale
/// models. Per the AnimateDiff/Janus reduced-scale precedent, the scaffold
/// supplies a small UNet + VAE with the SAME architecture shape — critically
/// keeping CatVTON's defining 12-channel inflated input convolution (the
/// person/garment/mask latent concatenation of Chong et al. 2024,
/// "Concatenation Is All You Need") — so every code path runs in seconds.
/// The production defaults stay paper-faithful.
/// </summary>
public class CatVTONModelTests : DiffusionModelTestBase<float>
{
    // Reduced smoke latent: 32x32 (not the production 64x64). The model-family
    // Predict path is a full multi-step reverse-sampling loop, and at 64x64 the
    // resolution-1 self-attention runs over 4096 tokens — ~268 MB of attention
    // matrices per layer pass accumulating on the autodiff tape across sampling
    // steps, which paged the test host into a >14 GB swap stall (CPU frozen,
    // blame-hang fired on Clone_ShouldProduceIdenticalOutput). At 32x32 the
    // attention is ≤1024 tokens and the whole suite runs in minutes, matching
    // the AnimateDiff reduced-scaffold precedent.
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
    {
        var unet = new UNetNoisePredictor<float>(
            inputChannels: 12, outputChannels: 4,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 32, seed: 42);

        var vae = new StandardVAE<float>(
            inputChannels: 3, latentChannels: 4,
            baseChannels: 32, channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 42);

        return new CatVTONModel<float>(predictor: unet, vae: vae, seed: 42);
    }
}
