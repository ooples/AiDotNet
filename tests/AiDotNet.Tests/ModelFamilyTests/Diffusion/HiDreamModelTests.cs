using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class HiDreamModelTests : DiffusionModelTestBase<float>
{
    // HiDream's 16-channel latent; keep the latent at 32×32 so the token count stays modest.
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    // HiDream defaults to a foundation-scale MMDiT-X (2048–2560 hidden, 24–38 layers): a single forward
    // exceeds the 120s model-family budget. Inject a tiny same-architecture MMDiT-X + VAE via the new
    // size overrides — latentChannels (16), patchSize (2) and contextDim (4096) stay paper-correct; only
    // hidden width / depth / head count shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new HiDreamModel<float>(
            predictor: new MMDiTXNoisePredictor<float>(
                variant: MMDiTXVariant.Medium, inputChannels: 16, patchSize: 2, contextDim: 4096,
                seed: 42, hiddenSizeOverride: 64, numLayersOverride: 2, numHeadsOverride: 4),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.13025, seed: 42));
}
