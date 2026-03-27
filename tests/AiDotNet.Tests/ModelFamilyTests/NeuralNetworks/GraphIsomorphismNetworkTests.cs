using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphIsomorphismNetworkTests : GraphNNModelTestBase
{
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    private static Vector<double>? _savedParams;

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var network = new GraphIsomorphismNetwork<double>();
        if (_savedParams == null)
            _savedParams = network.GetParameters();
        else
            network.UpdateParameters(_savedParams);
        return network;
    }
}
