using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class HopeNetworkTests : AssociativeMemoryTestBase
{
    protected override int[] InputShape => [256];
    protected override int[] OutputShape => [256];

    // HopeNetwork uses gradient-based training toward a target
    protected override bool IsAutoAssociative => false;

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new HopeNetwork<double>();
}
