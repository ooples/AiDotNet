using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class LayerNormalizationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new LayerNormalizationLayer<double>(4);

    protected override int[] InputShape => [2, 4];

    // LayerNorm normalizes each sample to mean=0, std=1. With constant inputs
    // (all 0.1 or all 0.9), every element equals the mean, so normalized output
    // is zero for both — identical outputs are mathematically correct.
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}
