using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class ReservoirLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ReservoirLayer<double>(inputSize: 4, reservoirSize: 16);

    protected override int[] InputShape => [1, 4];

    // ReservoirLayer has fixed weights (not trained via backprop)
    // Gradients pass through but don't produce weight updates
    protected override bool ExpectsNonZeroGradients => false;
}
