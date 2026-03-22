using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class DirectionalGraphLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new DirectionalGraphLayer<double>(inputFeatures: 4, outputFeatures: 8,
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1; adj[0, 1, 0] = 0; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        return layer;
    }
    protected override int[] InputShape => [1, 2, 4];
}

public class EdgeConditionalConvLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new EdgeConditionalConvolutionalLayer<double>(
            inputFeatures: 4, outputFeatures: 8, edgeFeatures: 2,
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1; adj[0, 1, 0] = 1; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        // Edge features: [batch, numEdges, edgeFeatures] — rank 3
        // With 2 nodes fully connected: 4 edges, 2 edge features
        var ef = new Tensor<double>([1, 4, 2]);
        for (int i = 0; i < ef.Length; i++) ef[i] = 0.5;
        layer.SetEdgeFeatures(ef);
        return layer;
    }
    protected override int[] InputShape => [1, 2, 4];
}

public class PrincipalNeighbourhoodAggregationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new PrincipalNeighbourhoodAggregationLayer<double>(
            inputFeatures: 4, outputFeatures: 8);
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1; adj[0, 1, 0] = 1; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        return layer;
    }
    protected override int[] InputShape => [1, 2, 4];
}
