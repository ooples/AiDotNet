using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class MobileNetV3NetworkTests : NeuralNetworkModelTestBase
{
    // Per Howard et al. ICCV 2019: MobileNetV3-Large
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [1000];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 1000);
        var config = new MobileNetV3Configuration(
            MobileNetV3Variant.Large, 1000,
            inputHeight: 32, inputWidth: 32);
        // MSE loss matches test evaluation (CategoricalCrossEntropy derivative
        // assumes softmax-normalized probabilities, wrong for raw logits)
        return new MobileNetV3Network<double>(arch, config,
            lossFunction: new MeanSquaredErrorLoss<double>());
    }
}
