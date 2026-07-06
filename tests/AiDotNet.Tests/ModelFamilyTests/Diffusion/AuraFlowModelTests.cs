using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class AuraFlowModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    // AuraFlow defaults to a foundation-scale DiT (hidden 1536, 24 layers, ~680M params): it
    // trips disk-backed weight streaming and cannot complete a single forward within the 120s
    // test budget. Inject a tiny DiT + VAE so these model-family invariants run against a REAL
    // instance of the same architecture at a runnable scale — latentChannels (4) and contextDim
    // (4096) stay paper-correct; only the hidden width / depth / VAE base channels are shrunk,
    // keeping the model well below the streaming threshold.
    protected override IDiffusionModel<float> CreateModel()
        => new AuraFlowModel<float>(
            dit: new DiTNoisePredictor<float>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.13025, seed: 42),
            seed: 42);
}
