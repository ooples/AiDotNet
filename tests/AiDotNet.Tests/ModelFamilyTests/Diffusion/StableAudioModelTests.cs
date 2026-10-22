using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087 / #1706: out of the HeavyTimeout lane. The default DiT (1536 wide, 24 layers) put
// the class past the PR gate. Per the #1706 out-of-memory ladder the test caps the latent at 16x16
// and injects a small DiT of the same shape: 64 latent channels, patch size 1 and context width
// 768, 64 wide and 2 layers. The AudioVAE keeps its default construction.
public class StableAudioModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 64, 16, 16];
    protected override int[] OutputShape => [1, 64, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new StableAudioModel<float>(
            dit: new DiTNoisePredictor<float>(
                inputChannels: 64, hiddenSize: 64, numLayers: 2, numHeads: 2, patchSize: 1,
                contextDim: 768, latentSpatialSize: 16, seed: 42));
}
