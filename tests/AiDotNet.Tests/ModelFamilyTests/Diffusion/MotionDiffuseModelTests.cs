using AiDotNet.Interfaces;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// HeavyTimeout (#1706): foundation-scale motion diffusion transformer. Verified genuine OOM — throws
// System.OutOfMemoryException during CONSTRUCTION under a 16 GB DOTNET_GCHeapHardLimit reproducing the CI
// runner ceiling (Metadata_ShouldExist alone OOMs in GC.AllocateNewArray), OS-OOM-killing the Diffusion
// J-M shard. Runs in the nightly heavy lane. Drop once weight streaming lets it fit.
[Xunit.Trait("Category", "HeavyTimeout")]
public class MotionDiffuseModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    // Build the SiT predictor + VAE at a REDUCED scale instead of the default DiT-XL-class SiT
    // (hiddenSize 1152 x 28 layers). At full scale the train-then-forward loop runs ~117s solo — just
    // under the 120s gate, so it tips over the timeout under shard load. The small SiT (64 x 2) drops it
    // to a few seconds. Shape-critical dims preserved (inputChannels = VAE_LATENT_CHANNELS 4).
    protected override IDiffusionModel<float> CreateModel()
        => new MotionDiffuseModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.SiTPredictor<float>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 2, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
