using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphAttentionNetworkTests : GraphNNModelTestBase<float>
{
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    // Ensure all network instances start from identical weights.
    // Tensor<T>.CreateRandom uses cryptographic seeds, so each call produces
    // different weights. We save the first network's params and reuse them.
    private static Vector<float>? _savedParams;

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var network = new GraphAttentionNetwork<float>();
        if (_savedParams == null)
            _savedParams = network.GetParameters();
        else
            network.UpdateParameters(_savedParams);
        return network;
    }
}
