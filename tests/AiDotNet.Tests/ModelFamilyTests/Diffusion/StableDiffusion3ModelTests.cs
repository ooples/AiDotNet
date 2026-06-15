using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StableDiffusion3ModelTests : DiffusionModelTestBase<float>
{
    // SD3's 16-channel latent. Keep the latent small (16×16) so the patchified token count stays modest;
    // latentChannels (16) and contextDim (4096) stay paper-correct.
    protected override int[] InputShape => [1, 16, 16, 16];
    protected override int[] OutputShape => [1, 16, 16, 16];

    // SD3 defaults to a foundation-scale MMDiT (hidden 1536, 24 joint layers): a single forward exceeds
    // the 120s model-family budget. Inject a tiny same-architecture MMDiT + VAE — latentChannels (16),
    // patchSize (2) and contextDim (4096) stay paper-correct; only hidden width / depth / head count shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new StableDiffusion3Model<float>(
            mmdit: new MMDiTNoisePredictor<float>(
                inputChannels: 16, hiddenSize: 64, numJointLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.13025, seed: 42),
            seed: 42);
}
