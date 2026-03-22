using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class AudioVisualEventLocalizationNetworkTests : NeuralNetworkModelTestBase
{
    // Default: inputSize=512, outputSize=1 (binary classification)
    protected override int[] InputShape => [512];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new AudioVisualEventLocalizationNetwork<double>();
}
