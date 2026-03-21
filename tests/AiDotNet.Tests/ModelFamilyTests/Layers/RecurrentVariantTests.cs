using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class BidirectionalLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new BidirectionalLayer<double>(
            new RecurrentLayer<double>(inputSize: 4, hiddenSize: 8,
                activationFunction: new TanhActivation<double>()),
            mergeMode: true,
            activationFunction: new TanhActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4];
}

// ConvLSTMLayer: Constructor crashes with IndexOutOfRange on inputShape [1,4,4]
// TODO: Investigate correct inputShape format for ConvLSTM
