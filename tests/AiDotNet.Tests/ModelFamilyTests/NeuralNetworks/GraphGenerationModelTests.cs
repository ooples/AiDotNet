using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphGenerationModelTests : GraphNNModelTestBase
{
    // GraphGenerationModel: inputFeatures=16, latentDim=16, 10 nodes
    // Predict outputs adjacency matrix [numNodes, numNodes]
    protected override int[] InputShape => [10, 16];
    protected override int[] OutputShape => [10, 10];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new GraphGenerationModel<double>(inputFeatures: 16, maxNodes: 10);
}
