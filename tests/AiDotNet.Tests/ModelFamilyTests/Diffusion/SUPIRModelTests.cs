using AiDotNet.Interfaces;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale SUPIR restoration diffusion — verified OOM
// (System.OutOfMemoryException at CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI
// ceiling; Metadata_ShouldExist alone OOMs), OS-OOM-kills the Diffusion Step-Sync shard. Nightly heavy lane.
[Xunit.Trait("Category", "HeavyTimeout")]
public class SUPIRModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
    {
        var unet = new UNetNoisePredictor<float>(
            inputChannels: 4, outputChannels: 4,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        var vae = new StandardVAE<float>(
            inputChannels: 3, latentChannels: 4,
            baseChannels: 32, channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 42);

        return new SUPIRModel<float>(unet: unet, vae: vae, seed: 42);
    }
}
