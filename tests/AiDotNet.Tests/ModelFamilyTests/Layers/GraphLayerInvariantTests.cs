using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class GraphConvolutionalLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new GraphConvolutionalLayer<double>(inputFeatures: 4, outputFeatures: 8,
            (IActivationFunction<double>?)null);
        // Graph layers need adjacency matrix set before Forward
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1;
        adj[0, 1, 0] = 1; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        return layer;
    }
    // Graph layers expect [batch, nodes, features]
    protected override int[] InputShape => [1, 2, 4];
}
