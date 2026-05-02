using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Paper-faithful invariant tests for EfficientNet-B0 per Tan &amp; Le 2019,
/// "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
/// </summary>
/// <remarks>
/// Default ctor instantiates EfficientNet-B0 with the ImageNet-1k classification
/// head (NumClasses = 1000), which is the paper's reported configuration.
/// OutputShape mirrors that contract — overriding to a smaller class count
/// here would not match the paper-faithful model defaults.
/// </remarks>
public class EfficientNetNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [3, 64, 64];
    protected override int[] OutputShape => [1, 1000];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new EfficientNetNetwork<double>();
}
