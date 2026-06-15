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
    // DCGAN parameterless defaults: latentSize=100, imageChannels=3,
    // imageHeight=64, imageWidth=64 (Radford et al. 2015 §3, NCHW).
    protected override int[] InputShape => [100];
    protected override int[] OutputShape => [3, 64, 64];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new DCGAN<float>(latentSize: 100, imageChannels: 3, imageHeight: 64, imageWidth: 64);
}
