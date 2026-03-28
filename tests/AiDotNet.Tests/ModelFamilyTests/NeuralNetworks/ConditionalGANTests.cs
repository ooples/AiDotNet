using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class ConditionalGANTests : GANModelTestBase
{
    // Default ConditionalGAN: generator inputSize=110 (latentDim=100 + numClasses=10), outputSize=784
    // The public API accepts noise-only input (100), conditions are added internally by Predict/Train
    protected override int[] InputShape => [1, 100];
    protected override int[] OutputShape => [1, 784];

    // GAN training is adversarial — generator and discriminator compete, so the MSE
    // (computed between generated output and random target) oscillates rather than
    // monotonically decreasing. A higher tolerance accounts for this.
    protected override double MoreDataTolerance => 0.01;

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new ConditionalGAN<double>();
}
