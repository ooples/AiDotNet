using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class InstanceNormalizationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new InstanceNormalizationLayer<double>(numChannels: 2);
    // InstanceNorm expects [batch, channels, spatial...]
    protected override int[] InputShape => [1, 2, 4];
    // InstanceNorm normalizes each instance/channel — constant inputs normalize identically
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}

public class GroupNormalizationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new GroupNormalizationLayer<double>(numGroups: 2, numChannels: 4);
    // GroupNorm expects [batch, channels, height, width]
    protected override int[] InputShape => [1, 4, 2, 2];
    // GroupNorm normalizes within groups — constant inputs normalize identically
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}
