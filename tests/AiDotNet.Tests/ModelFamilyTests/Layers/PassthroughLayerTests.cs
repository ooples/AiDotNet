using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

/// <summary>
/// Tests for layers that pass data through without trainable parameters.
/// </summary>
public class InputLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new InputLayer<double>(4);
    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class FlattenLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new FlattenLayer<double>([2, 3]);
    protected override int[] InputShape => [1, 2, 3];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class DropoutLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new DropoutLayer<double>(0.5);
    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}
