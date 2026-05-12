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

    // VGG16-BN at paper-scale (138M params, 224×224 input) is roughly 3
    // orders of magnitude more expensive per Train() call than the
    // CIFAR-sized 32×32 configs other invariant tests use. The base
    // class's MoreData iteration counts (50 / 200) would burn through
    // xUnit's 120s per-test budget several times over. Override down to
    // values that still let MoreData_ShouldNotDegrade observe a
    // monotonic-loss signal but fit inside the timeout.
    protected override int TrainingIterations => 2;
    protected override int MoreDataShortIterations => 2;
    protected override int MoreDataLongIterations => 4;

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new VGGNetwork<double>();
}
