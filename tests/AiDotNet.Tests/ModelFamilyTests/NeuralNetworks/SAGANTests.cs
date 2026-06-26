using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual GAN test scaffold for SAGAN. SAGAN takes generator/discriminator
/// architectures (no parameterless ctor) and its discriminator consumes
/// image-space tensors, so the auto-generated generic GAN scaffold fed a
/// wrong-shaped input and the discriminator threw a channel-depth mismatch.
/// Build a small SAGAN (1-channel 8x8 images, latent 16) and declare the latent
/// InputShape + image OutputShape so <see cref="GANModelTestBase{T}"/>'s
/// generator-from-latent and adversarial-training invariants run correctly.
/// </summary>
public class SAGANTests : GANModelTestBase<float>
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
        return new SAGAN<float>(generatorArchitecture, discriminatorArchitecture,
            latentSize: 16, imageChannels: 1, imageHeight: 8, imageWidth: 8,
            generatorChannels: 8, discriminatorChannels: 8);
    }
}
