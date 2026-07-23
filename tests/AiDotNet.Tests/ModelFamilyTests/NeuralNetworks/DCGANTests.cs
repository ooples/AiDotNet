using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for DCGAN. The auto-generated GAN family scaffold
/// supplies a rank-1 [16] input and rank-1 [4] output, which match neither
/// the latent-space input (default depth 100) nor the image-space output
/// (default [3, 64, 64]) of DCGAN's parameterless ctor. Override both shapes
/// here so <see cref="GANModelTestBase"/>'s Generator-from-latent invariants
/// (GeneratorOutput_ShouldHaveCorrectShape, DifferentLatentInputs_ProduceDifferentOutputs)
/// actually exercise the generator path.
/// </summary>
public class DCGANTests : GANModelTestBase<float>
{
    // DCGAN (Radford et al. 2015 §3, NCHW). The paper default is 64x64, but the
    // smoke-test fixture uses 32x32: MoreData_ShouldNotDegrade runs 50 + 200 = 250
    // training iterations, and at 64x64 the (native-GEMM-bound) convolution stack
    // costs ~235 ms/iter, overrunning the 120s per-test budget on the slower CI
    // runner. Convolution cost scales ~quadratically with spatial size, so 32x32 is
    // ~4x cheaper (~60 ms/iter) and the FULL 250-iteration invariant fits the budget
    // — the transposed-conv generator / strided-conv discriminator architecture stays
    // paper-faithful; only the smoke-test resolution shrinks.
    protected override int[] InputShape => [100];
    protected override int[] OutputShape => [3, 32, 32];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new DCGAN<float>(latentSize: 100, imageChannels: 3, imageHeight: 32, imageWidth: 32);
}
