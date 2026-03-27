using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphAttentionNetworkTests : GraphNNModelTestBase
{
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    // Ensure all network instances start from identical weights.
    // Tensor<T>.CreateRandom uses cryptographic seeds, so each call produces
    // different weights. We save the first network's params and reuse them.
    private static Vector<double>? _savedParams;

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var network = new GraphAttentionNetwork<double>();
        if (_savedParams == null)
            _savedParams = network.GetParameters();
        else
            network.UpdateParameters(_savedParams);
        return network;
    }
}
