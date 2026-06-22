using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphIsomorphismNetworkTests : GraphNNModelTestBase<float>
{
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    private static Vector<float>? _savedParams;

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var network = new GraphIsomorphismNetwork<float>();
        if (_savedParams == null)
            _savedParams = network.GetParameters();
        else
            network.UpdateParameters(_savedParams);
        return network;
    }
}
