using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual GAN test scaffold for ProgressiveGAN. Like SAGAN/BigGAN it takes
/// generator/discriminator architectures and consumes image-space tensors, which
/// the generic auto-gen GAN scaffold can't supply. Build a small ProgressiveGAN
/// (1-channel, latent 16, maxResolutionLevel 2 → small images, baseFeatureMaps 8)
/// and declare the latent InputShape so <see cref="GANModelTestBase{T}"/>'s
/// generator-from-latent and adversarial-training invariants run correctly.
/// </summary>
public class ProgressiveGANTests : GANModelTestBase<float>
{
    protected override int[] InputShape => [16];
    protected override int[] OutputShape => [1, 8, 8];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Generative, NetworkComplexity.Simple,
            inputSize: 64, inputHeight: 8, inputWidth: 8, inputDepth: 1, outputSize: 64);
        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.BinaryClassification, NetworkComplexity.Simple,
            inputSize: 64, inputHeight: 8, inputWidth: 8, inputDepth: 1, outputSize: 1);
        return new ProgressiveGAN<float>(generatorArchitecture, discriminatorArchitecture,
            latentSize: 16, imageChannels: 1, maxResolutionLevel: 2, baseFeatureMaps: 8);
    }
}
