using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class FullyConnectedLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new FullyConnectedLayer<double>(4, 8, new ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4];
}

public class FeedForwardLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new FeedForwardLayer<double>(4, 8, new ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4];
}

public class SparseLinearLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SparseLinearLayer<double>(inputFeatures: 4, outputFeatures: 8, sparsity: 0.5);
    protected override int[] InputShape => [1, 4];
}

public class HyperbolicLinearLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new HyperbolicLinearLayer<double>(inputFeatures: 4, outputFeatures: 8);
    protected override int[] InputShape => [1, 4];
}

public class OctonionLinearLayerTests : LayerTestBase
{
    // Octonion: inputFeatures=2 means 2 octonions * 8 components = 16 input elements
    protected override ILayer<double> CreateLayer()
        => new OctonionLinearLayer<double>(inputFeatures: 2, outputFeatures: 4);
    protected override int[] InputShape => [1, 16]; // 2 octonions * 8
}

public class GatedLinearUnitLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new GatedLinearUnitLayer<double>(inputDimension: 4, outputDimension: 8,
            gateActivation: new SigmoidActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4];
}
