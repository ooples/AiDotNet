using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class VGGNetworkTests : NeuralNetworkModelTestBase
{
    // Paper-canonical VGG16-BN on ImageNet input per Simonyan & Zisserman 2014
    // ("Very Deep Convolutional Networks for Large-Scale Image Recognition"):
    // 224 × 224 × 3 → 1000 classes. Matches the parameterless VGGNetwork() ctor.
    protected override int[] InputShape => [1, 3, 224, 224];
    protected override int[] OutputShape => [1000];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new VGGNetwork<double>();
}
