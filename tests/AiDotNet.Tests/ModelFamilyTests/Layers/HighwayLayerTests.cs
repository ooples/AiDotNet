using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class HighwayLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new HighwayLayer<double>(inputDimension: 4,
            transformActivation: new AiDotNet.ActivationFunctions.ReLUActivation<double>() as IActivationFunction<double>);

    protected override int[] InputShape => [1, 4];
}
