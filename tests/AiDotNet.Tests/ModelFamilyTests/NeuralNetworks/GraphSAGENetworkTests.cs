using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphSAGENetworkTests : GraphNNModelTestBase
{
    // GraphSAGE default: inputSize=128, outputSize=7, 10 nodes
    // Disable L2 normalization: with constant-valued test inputs on a
    // fully-connected graph, normalization removes magnitude info, making
    // DifferentInputs/ScaledInput tests degenerate.
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GraphSAGENetwork<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.MultiClassClassification,
                inputSize: 128,
                outputSize: 7),
            normalize: false);
}
