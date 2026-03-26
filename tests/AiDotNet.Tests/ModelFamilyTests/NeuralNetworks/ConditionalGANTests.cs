using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class ConditionalGANTests : GANModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new ConditionalGAN<double>();
}
