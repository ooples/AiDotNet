using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class HunyuanDiTModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    // Hunyuan-DiT defaults to a foundation-scale DiT (hidden 1408, 40 layers): it trips
    // disk-backed weight streaming and cannot complete a single forward within the 120s test
    // budget. Inject a tiny DiT + VAE so these model-family invariants run against a REAL
    // instance of the same architecture at a runnable scale — latentChannels (4) and contextDim
    // (2048) stay paper-correct; only the hidden width / depth / VAE base channels are shrunk,
    // keeping the model well below the streaming threshold.
    protected override IDiffusionModel<double> CreateModel()
        => new HunyuanDiTModel<double>(
            dit: new DiTNoisePredictor<double>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 2048, seed: 42),
            vae: new StandardVAE<double>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.13025, seed: 42),
            seed: 42);
}
