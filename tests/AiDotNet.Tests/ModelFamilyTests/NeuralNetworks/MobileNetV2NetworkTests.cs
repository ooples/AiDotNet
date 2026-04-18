using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class MobileNetV2NetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [3, 64, 64];
    protected override int[] OutputShape => [10];

    // The parameterless MobileNetV2Network() constructor defaults to 1000
    // ImageNet classes and 224x224 input — incompatible with this test's
    // small 3x64x64 probe and 10-class OutputShape assertion. Use the
    // architecture-aware overload so the network classifier head matches
    // the test's expected output dimension.
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new MobileNetV2Network<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.MultiClassClassification,
                inputHeight: 64, inputWidth: 64, inputDepth: 3,
                outputSize: 10),
            MobileNetV2Configuration.CreateStandard(numClasses: 10));
}
