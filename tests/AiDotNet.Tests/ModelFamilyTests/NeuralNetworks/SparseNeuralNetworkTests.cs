using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SparseNeuralNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new SparseNeuralNetwork<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                inputSize: 128,
                outputSize: 1),
            sparsity: 0.5); // 50% sparsity for testing (default 90% is too sparse for single layer)
}
