using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class ResNetNetworkTests : NeuralNetworkModelTestBase<float>
{
    // ResNet requires 4D input [batch, channels, height, width] for convolutional layers.
    //
    // Use the purpose-built test-scale variant (ResNetNetwork.ForTesting): ResNet18
    // (the smallest variant, BasicBlock) at CIFAR-style 32x32x3 / 10-class. The default
    // ctor is ImageNet-scale ResNet50 at 224x224x3 / 1000-class — running the base
    // training probes against it (LossStrictlyDecreasesOnMemorizationTask = 100 Train
    // steps, MoreData = 50+200 steps) builds gigabytes of activation tape per iteration
    // and OOM-KILLS the 16GB CI runner ("runner received a shutdown signal" partway
    // through the heavy training tests, after the light ones pass). This is a test-SCALE
    // choice only — identical to ODISE's 32x32-vs-paper-512 — the architecture, residual
    // blocks, and every invariant (convergence, clone, gradient flow) are unchanged and
    // hold for ResNet18 exactly as for ResNet50.
    private const int NumClasses = 10;
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [NumClasses];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => ResNetNetwork<float>.ForTesting(numClasses: NumClasses);
}
