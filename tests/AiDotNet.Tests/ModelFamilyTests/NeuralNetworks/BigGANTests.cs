using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual GAN test scaffold for BigGAN. Like SAGAN it takes generator/discriminator
/// architectures and consumes image-space tensors in its (class-conditional)
/// discriminator, which the generic auto-gen GAN scaffold can't supply. Build a
/// small conditional BigGAN (1-channel 8x8, 10 classes, latent 16). BigGAN's
/// generator expects a 2D latent code [batch, latentSize], so the InputShape is
/// [1, 16] rather than the rank-1 [16] DCGAN uses.
/// </summary>
public class BigGANTests : GANModelTestBase<float>
{
    protected override int[] InputShape => [1, 16];
    protected override int[] OutputShape => [1, 8, 8];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Generative, NetworkComplexity.Simple,
            inputSize: 64, inputHeight: 8, inputWidth: 8, inputDepth: 1, outputSize: 64);
        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.BinaryClassification, NetworkComplexity.Simple,
            inputSize: 64, inputHeight: 8, inputWidth: 8, inputDepth: 1, outputSize: 1);
        return new BigGAN<float>(generatorArchitecture, discriminatorArchitecture,
            latentSize: 16, numClasses: 10, classEmbeddingDim: 8, imageChannels: 1,
            imageHeight: 8, imageWidth: 8, generatorChannels: 8, discriminatorChannels: 8);
    }
}
