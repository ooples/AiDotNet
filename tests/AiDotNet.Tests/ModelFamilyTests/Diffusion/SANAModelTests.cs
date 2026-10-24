using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087 / #1706: out of the HeavyTimeout lane. The default foundation-scale config is too large for
// the PR gate, so per the #1706 out-of-memory ladder the test caps the latent at 16x16 and injects
// a small same-architecture EMMDiTPredictor and DeepCompressionVAE. Channel counts, context widths,
// patch sizes and scale factors keep the model's defaults; only width and depth shrink.
public class SANAModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 32, 16, 16];
    protected override int[] OutputShape => [1, 32, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new SANAModel<float>(
            predictor: new EMMDiTPredictor<float>(
                inputChannels: 32, contextDim: 2048, numLayersOverride: 2, numHeadsOverride: 4,
                hiddenSizeOverride: 64, seed: 42),
            vae: new DeepCompressionVAE<float>(
                inputChannels: 3, latentChannels: 32, downsampleFactor: 32, baseChannels: 16,
                seed: 42),
            seed: 42);
}
