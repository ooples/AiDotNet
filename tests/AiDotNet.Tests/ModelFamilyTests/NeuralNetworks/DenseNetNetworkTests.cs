using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DenseNetNetworkTests : NeuralNetworkModelTestBase
{
    // Use small config per CreateForTesting: 32x32 input with small growth rate
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [10];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var config = DenseNetConfiguration.CreateForTesting(numClasses: 10);
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 10);
        return new DenseNetNetwork<double>(arch, config);
    }
}
