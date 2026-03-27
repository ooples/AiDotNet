using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphGenerationModelTests : GraphNNModelTestBase
{
    protected override int[] InputShape => [10, 16];
    protected override int[] OutputShape => [10, 10];

    private static Vector<double>? _savedParams;

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var network = new GraphGenerationModel<double>(inputFeatures: 16, maxNodes: 10);
        if (_savedParams == null)
            _savedParams = network.GetParameters();
        else
            network.UpdateParameters(_savedParams);
        return network;
    }
}
