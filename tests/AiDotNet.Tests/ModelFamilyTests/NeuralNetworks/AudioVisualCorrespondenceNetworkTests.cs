using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class AudioVisualCorrespondenceNetworkTests : NeuralNetworkModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new AudioVisualCorrespondenceNetwork<double>();
}
