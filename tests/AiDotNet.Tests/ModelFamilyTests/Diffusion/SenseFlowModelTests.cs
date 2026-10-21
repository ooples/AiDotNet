using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087 / #1706: out of the HeavyTimeout lane. The default foundation-scale config is too large for
// the PR gate, so per the #1706 out-of-memory ladder the test caps the latent at 16x16 and injects
// a small same-architecture FluxDoubleStreamPredictor and StandardVAE. Channel counts, context
// widths, patch sizes and scale factors keep the model's defaults; only width and depth shrink.
public class SenseFlowModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 16, 16];
    protected override int[] OutputShape => [1, 16, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new SenseFlowModel<float>(
            predictor: new FluxDoubleStreamPredictor<float>(
                inputChannels: 16, patchSize: 2, contextDim: 4096,
                variant: FluxPredictorVariant.Dev, numHeads: 2, hiddenSize: 64, numSingleLayers: 1,
                numJointLayers: 1, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.3611, seed: 42),
            seed: 42);
}
