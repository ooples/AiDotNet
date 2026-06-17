using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class Flux1ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    // FLUX.1 defaults to a 12B-scale hybrid MMDiT (hidden 3072, 19 joint + 38 single layers):
    // it trips disk-backed weight streaming and cannot complete a single forward within the
    // 120s test budget. Inject a tiny MMDiT + VAE so these model-family invariants run against
    // a REAL instance of the same architecture at a runnable scale — latentChannels (16) and
    // contextDim (4096) stay paper-correct; only the hidden width / depth / VAE base channels
    // are shrunk, keeping the model well below the streaming threshold.
    protected override IDiffusionModel<float> CreateModel()
        => new Flux1Model<float>(
            mmdit: new MMDiTNoisePredictor<float>(
                inputChannels: 16, hiddenSize: 64, numJointLayers: 2, numSingleLayers: 2,
                numHeads: 2, patchSize: 2, contextDim: 4096, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 1.5305, seed: 42),
            seed: 42);
}
