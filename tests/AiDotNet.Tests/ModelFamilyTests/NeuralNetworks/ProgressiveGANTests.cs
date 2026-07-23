using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Model-family tests for ProgressiveGAN. ProgressiveGAN derives from
/// GenerativeAdversarialNetwork and, like every GAN, takes a 1D latent vector as the
/// generator input (Karras et al. 2017 / Goodfellow 2014): the InputShape is the latent
/// [16], the OutputShape is the image [1, 8, 8]. The target resolution is
/// 4·2^maxResolutionLevel, so maxResolutionLevel 1 → 8x8. A small model (latent 16,
/// 1-channel, 8 base feature maps) keeps the adversarial-training invariants fast.
/// </summary>
public class ProgressiveGANTests : GANModelTestBase<float>
{
    protected override int[] InputShape => [16];
    protected override int[] OutputShape => [1, 8, 8];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new ProgressiveGAN<float>(
            latentSize: 16,
            imageChannels: 1,
            maxResolutionLevel: 1,
            baseFeatureMaps: 8);
}
