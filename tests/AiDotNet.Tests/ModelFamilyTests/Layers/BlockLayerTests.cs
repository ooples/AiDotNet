using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class BasicBlockTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new BasicBlock<double>(inChannels: 1, outChannels: 1, stride: 1);
    // BasicBlock expects [batch, channels, height, width]
    protected override int[] InputShape => [1, 1, 8, 8];
}

public class BottleneckBlockTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new BottleneckBlock<double>(inChannels: 4, baseChannels: 4, stride: 1, inputHeight: 8, inputWidth: 8);
    protected override int[] InputShape => [1, 4, 8, 8];
    // BatchNorm normalizes constant inputs to zero — identical outputs expected
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}
