using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphSAGENetworkTests : GraphNNModelTestBase
{
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    private static Vector<double>? _savedParams;

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var network = new GraphSAGENetwork<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.MultiClassClassification,
                inputSize: 128,
                outputSize: 7),
            normalize: false);
        if (_savedParams == null)
            _savedParams = network.GetParameters();
        else
            network.UpdateParameters(_savedParams);
        return network;
    }
}
