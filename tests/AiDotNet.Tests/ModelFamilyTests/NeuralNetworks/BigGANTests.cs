using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Model-family tests for BigGAN. BigGAN derives from GenerativeAdversarialNetwork and,
/// like every GAN, takes a 1D latent vector as the generator input (Brock et al. 2018 /
/// Goodfellow 2014): the InputShape is the latent [16], the OutputShape is the image
/// [1, 8, 8]. A small class-conditional BigGAN (latent 16, 10 classes, 1-channel 8x8
/// images, 8 base feature maps) keeps the adversarial-training invariants fast.
/// </summary>
public class BigGANTests : GANModelTestBase<float>
{
    protected override int[] InputShape => [16];
    protected override int[] OutputShape => [1, 8, 8];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new BigGAN<float>(
            latentSize: 16,
            numClasses: 10,
            classEmbeddingDim: 8,
            imageChannels: 1,
            imageHeight: 8,
            imageWidth: 8,
            generatorChannels: 8,
            discriminatorChannels: 8);
}
