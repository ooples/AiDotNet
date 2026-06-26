using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Model-family tests for SAGAN. SAGAN derives from GenerativeAdversarialNetwork and,
/// like every GAN, takes a 1D latent vector as the generator input (Zhang et al. 2019 /
/// Goodfellow 2014): the InputShape is the latent [16], the OutputShape is the image
/// [1, 8, 8]. A small SAGAN (latent 16, 1-channel 8x8 images, 8 base feature maps) keeps
/// the adversarial-training invariants fast.
/// </summary>
public class SAGANTests : GANModelTestBase<float>
{
    protected override int[] InputShape => [16];
    protected override int[] OutputShape => [1, 8, 8];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SAGAN<float>(
            latentSize: 16,
            imageChannels: 1,
            imageHeight: 8,
            imageWidth: 8,
            generatorChannels: 8,
            discriminatorChannels: 8);
}
