using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")]
// LGM's diffusion stage is its multi-view model, so the family contract (noise prediction on a [1, 4, 16, 16]
// latent) runs through a small same-architecture MVDream, as MVDreamModelTests does. The Gaussian reconstructor uses a
// small same-structure configuration; LGMArchitectureTests covers its shapes, parameter count and activations.
public class LGMModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new LGMModel<float>(
            config: new LGMConfig
            {
                InputSize = 16,
                DownChannels = new[] { 8, 16, 16 },
                DownAttention = new[] { false, false, true },
                UpChannels = new[] { 16, 16 },
                UpAttention = new[] { true, false },
                LayersPerBlock = 1,
                AttentionHeads = 2,
                NormGroups = 4
            },
            multiViewModel: new MVDreamModel<float>(
                multiViewUNet: new MultiViewUNet<float>(
                    inputChannels: 4, outputChannels: 4, baseChannels: 32, numViews: 4,
                    contextDim: 1024, seed: 42),
                imageVAE: new StandardVAE<float>(
                    inputChannels: 3, latentChannels: 4, baseChannels: 16,
                    channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
                seed: 42),
            seed: 42);
}
