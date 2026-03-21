using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class GRULayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new GRULayer<double>(inputSize: 4, hiddenSize: 8, returnSequences: false,
            activation: new AiDotNet.ActivationFunctions.TanhActivation<double>() as IActivationFunction<double>);

    protected override int[] InputShape => [1, 4];
}
