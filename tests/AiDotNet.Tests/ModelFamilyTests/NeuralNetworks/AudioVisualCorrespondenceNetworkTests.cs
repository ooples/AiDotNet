using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class AudioVisualCorrespondenceNetworkTests : NeuralNetworkModelTestBase
{
    // Default: inputSize=512, final layer outputs 128 (separation mask head)
    protected override int[] InputShape => [512];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new AudioVisualCorrespondenceNetwork<double>();
}
