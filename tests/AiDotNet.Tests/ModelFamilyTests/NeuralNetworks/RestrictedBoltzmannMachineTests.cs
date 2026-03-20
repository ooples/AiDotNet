using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class RestrictedBoltzmannMachineTests : NeuralNetworkModelTestBase
{
    // Use smaller dimensions to avoid sigmoid saturation flakiness
    protected override int[] InputShape => [16];
    protected override int[] OutputShape => [8];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new RestrictedBoltzmannMachine<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                inputSize: 16,
                outputSize: 8),
            visibleSize: 16, hiddenSize: 8,
            scalarActivation: (IActivationFunction<double>?)null);
}
