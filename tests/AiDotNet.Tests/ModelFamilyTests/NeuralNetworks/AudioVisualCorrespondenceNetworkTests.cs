using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class AudioVisualCorrespondenceNetworkTests : NeuralNetworkModelTestBase
{
    // Default: inputSize=512, final layer outputs 2 (binary classification per L3-Net)
    protected override int[] InputShape => [512];
    protected override int[] OutputShape => [2]; // Binary classification per Arandjelovic & Zisserman 2017

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new AudioVisualCorrespondenceNetwork<double>();
}
