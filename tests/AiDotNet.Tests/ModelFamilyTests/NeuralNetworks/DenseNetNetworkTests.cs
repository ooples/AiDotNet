using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DenseNetNetworkTests : NeuralNetworkModelTestBase
{
    // Per Huang et al. 2017: CIFAR-10 uses 32x32 input with 10 classes
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [10];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        // CIFAR variant per paper: 3 blocks, growth rate 12, small stem
        var config = new DenseNetConfiguration(
            variant: DenseNetVariant.Custom,
            numClasses: 10,
            inputHeight: 32,
            inputWidth: 32,
            inputChannels: 3,
            growthRate: 12,
            compressionFactor: 0.5,
            customBlockLayers: [4, 4, 4]); // 3 blocks per CIFAR paper

        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 10);

        return new DenseNetNetwork<double>(arch, config,
            lossFunction: new AiDotNet.LossFunctions.MeanSquaredErrorLoss<double>());
    }
}
