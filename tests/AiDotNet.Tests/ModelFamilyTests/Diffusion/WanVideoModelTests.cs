using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class WanVideoModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    // WanVideoModel defaults to the paper 14B variant (3072 hidden, 40 layers): a foundation-
    // scale model whose lazy weights trip disk-backed weight streaming and cannot complete a
    // single forward within the 120s test budget (and leak streaming reservations across
    // tests). Inject a tiny DiT + TemporalVAE so these model-family invariants run against a
    // REAL instance of the same architecture at a runnable scale — latentChannels (16) and
    // contextDim (4096) stay paper-correct; only the hidden width / depth / VAE base channels
    // are shrunk, keeping the model well below the streaming threshold.
    protected override IDiffusionModel<float> CreateModel()
        => new WanVideoModel<float>(
            dit: new DiTNoisePredictor<float>(
                inputChannels: 16, hiddenSize: 64, numLayers: 2, numHeads: 2,
                patchSize: 2, contextDim: 4096, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1,
                temporalKernelSize: 3, causalMode: true, latentScaleFactor: 0.13025, seed: 42),
            seed: 42);
}
