using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class MambaBlockTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MambaBlock<double>(sequenceLength: 4, modelDimension: 8);
    protected override int[] InputShape => [1, 4, 8]; // [batch, seq, model_dim]
}

public class S4DLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new S4DLayer<double>(sequenceLength: 4, modelDimension: 8, stateDimension: 4);
    protected override int[] InputShape => [1, 4, 8];
}

public class RetNetLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new RetNetLayer<double>(sequenceLength: 4, modelDimension: 8, numHeads: 2);
    protected override int[] InputShape => [1, 4, 8];
}

public class RWKVLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new RWKVLayer<double>(sequenceLength: 4, modelDimension: 8, numHeads: 2);
    protected override int[] InputShape => [1, 4, 8];
}

public class HyenaLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new HyenaLayer<double>(sequenceLength: 4, modelDimension: 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class LinearRecurrentUnitLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new LinearRecurrentUnitLayer<double>(sequenceLength: 4, modelDimension: 8, stateDimension: 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class MinGRULayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MinGRULayer<double>(sequenceLength: 4, modelDimension: 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class MinLSTMLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MinLSTMLayer<double>(sequenceLength: 4, modelDimension: 8);
    protected override int[] InputShape => [1, 4, 8];
}
