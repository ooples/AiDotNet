using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class ResidualLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ResidualLayer<double>(inputShape: [4], innerLayer: new DenseLayer<double>(4, 4),
            activationFunction: new AiDotNet.ActivationFunctions.ReLUActivation<double>() as IActivationFunction<double>);

    protected override int[] InputShape => [1, 4];
}
