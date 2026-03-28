using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Tests for ConditionalGAN per Mirza &amp; Osindero (2014).
/// Input: latentDim(100) + numClasses(10) = 110. Output: 784 (28x28 image).
/// </summary>
public class ConditionalGANTests : GANModelTestBase
{
    protected override int[] InputShape => [110];
    protected override int[] OutputShape => [784];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new ConditionalGAN<double>();
}
