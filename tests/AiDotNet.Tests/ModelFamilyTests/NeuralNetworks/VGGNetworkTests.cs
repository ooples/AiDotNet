using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class VGGNetworkTests : NeuralNetworkModelTestBase
{
    // Use a CIFAR-sized VGG11 (32x32x3, 10 classes, no BN) instead of the
    // parameterless ctor's VGG16_BN + 224x224 + 1000 classes (138M params).
    // Rationale:
    //   1. Runtime: VGG16_BN at 224x224 takes ~1m50s per Predict call on CI
    //      runners; the shard hits the 5-min runner shutdown window before
    //      finishing the suite (every test after the first gets cancelled).
    //   2. Dead-neuron invariants: VGG16_BN in eval mode with un-trained
    //      BatchNorm (running_mean=0, running_var=1) normalizes constant
    //      inputs to zero, then ReLU collapses them — so
    //      DifferentInputs_ShouldProduceDifferentOutputs reports "identical
    //      outputs" even though the underlying dense path has distinct biases.
    // The VGG architecture itself (Simonyan & Zisserman 2014) is unchanged;
    // the smoke suite just uses a smaller variant from the same paper family.
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [10];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 10);
        var config = VGGConfiguration.CreateForCIFAR(VGGVariant.VGG11, numClasses: 10);
        return new VGGNetwork<double>(arch, config);
    }
}
