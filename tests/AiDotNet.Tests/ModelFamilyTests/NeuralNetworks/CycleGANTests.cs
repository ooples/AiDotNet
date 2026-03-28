using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Tests for CycleGAN per Zhu et al. (2017) "Unpaired Image-to-Image Translation".
/// Input/Output: 784 (28x28 image, flattened).
/// </summary>
public class CycleGANTests : GANModelTestBase
{
    protected override int[] InputShape => [784];
    protected override int[] OutputShape => [784];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new CycleGAN<double>();
}
