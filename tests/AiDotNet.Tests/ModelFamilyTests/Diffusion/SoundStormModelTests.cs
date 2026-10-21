using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// #2087 / #1706: out of the HeavyTimeout lane. The default DiT (1024 wide, 12 layers) put
// the class past the PR gate. Per the #1706 out-of-memory ladder the test caps the latent at 16x16
// and injects a small DiT of the same shape: 8 latent channels, patch size 1 and context width
// 1024, 64 wide and 2 layers. The AudioVAE keeps its default construction.
public class SoundStormModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 8, 16, 16];
    protected override int[] OutputShape => [1, 8, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new SoundStormModel<float>(
            conformer: new DiTNoisePredictor<float>(
                inputChannels: 8, hiddenSize: 64, numLayers: 2, numHeads: 2, patchSize: 1,
                contextDim: 1024, latentSpatialSize: 16, seed: 42),
            seed: 42);
}
