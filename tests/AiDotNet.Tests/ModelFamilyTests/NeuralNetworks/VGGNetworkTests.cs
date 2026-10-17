using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// In the default gate again (#1394). It was deferred to the nightly heavy lane at ~7 s/iter (#1706); nightly
// 38859953869 ran every invariant inside its own timeout without the lane's 4x scaling, in parallel with the rest
// of its chunk: the longest was the memorization test at 2 m 3 s of 180 s (a fixed 120 s training budget), and
// no other test took more than 40 s of its 120 s.
public class VGGNetworkTests : NeuralNetworkModelTestBase<float>
{
    // Paper-canonical VGG16-BN on ImageNet input per Simonyan & Zisserman 2014
    // ("Very Deep Convolutional Networks for Large-Scale Image Recognition"):
    // 224 × 224 × 3 → 1000 classes. Matches the parameterless VGGNetwork() ctor.
    // Production-default iteration counts (10 / 50 / 200) are kept on
    // purpose — they exist to catch perf regressions in VGG / training
    // hot paths. If a CI run times out at this scale, fix the perf bug
    // in the implementation, do not weaken the test.
    protected override int[] InputShape => [1, 3, 224, 224];
    protected override int[] OutputShape => [1000];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new VGGNetwork<float>();
}
