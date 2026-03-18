using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphAttentionNetworkTests : GraphNNModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GraphAttentionNetwork<double>();
}
