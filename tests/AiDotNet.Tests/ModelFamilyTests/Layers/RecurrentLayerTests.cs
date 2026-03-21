using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class RecurrentLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new RecurrentLayer<double>(inputSize: 4, hiddenSize: 8,
            activationFunction: new TanhActivation<double>());

    // RecurrentLayer expects [seqLen, features] or [seqLen, batch, features]
    protected override int[] InputShape => [1, 4];
}
