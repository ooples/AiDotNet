using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class AttentionNetworkTests : NeuralNetworkModelTestBase
{
    // Use smaller dimensions for faster/more stable tests
    protected override int[] InputShape => [32];
    protected override int[] OutputShape => [32];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new AttentionNetwork<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                inputSize: 32,
                outputSize: 32),
            sequenceLength: 8, embeddingSize: 16);
}
