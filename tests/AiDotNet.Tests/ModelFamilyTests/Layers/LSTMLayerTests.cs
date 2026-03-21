using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class LSTMLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new LSTMLayer<double>(inputSize: 4, hiddenSize: 8, inputShape: [1, 4],
            activation: new AiDotNet.ActivationFunctions.TanhActivation<double>() as IActivationFunction<double>);

    protected override int[] InputShape => [1, 4];
}
