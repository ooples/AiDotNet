using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class GraphTransformerLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new GraphTransformerLayer<double>(inputFeatures: 4, outputFeatures: 8, numHeads: 2,
            headDim: 4, dropoutRate: 0.0,
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1; adj[0, 1, 0] = 1; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        return layer;
    }
    protected override int[] InputShape => [1, 2, 4];
}

public class MessagePassingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new MessagePassingLayer<double>(inputFeatures: 4, outputFeatures: 8,
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1; adj[0, 1, 0] = 1; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        return layer;
    }
    protected override int[] InputShape => [1, 2, 4];
}

public class ExpertLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        // Second dense receives 8 features from first dense's output
        var experts = new System.Collections.Generic.List<ILayer<double>>
        {
            new DenseLayer<double>(4, 8),
            new DenseLayer<double>(8, 8)
        };
        return new ExpertLayer<double>(experts, inputShape: [4], outputShape: [8],
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
    }
    protected override int[] InputShape => [1, 4];
}

public class LambdaLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new LambdaLayer<double>(
            inputShape: [4], outputShape: [4],
            forwardFunction: x => x,
            backwardFunction: (grad, _) => grad,
            activationFunction: new IdentityActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

/// <summary>
/// MultiplyLayer requires multiple inputs — tested with multi-input API.
/// </summary>
public class MultiplyLayerMultiInputTests
{
    [Fact]
    public void Forward_ShouldMultiplyElementwise()
    {
        var layer = new MultiplyLayer<double>(inputShapes: [[4], [4]],
            activationFunction: new IdentityActivation<double>() as IActivationFunction<double>);
        var input1 = new Tensor<double>([1, 4]);
        var input2 = new Tensor<double>([1, 4]);
        for (int i = 0; i < 4; i++) { input1[i] = 2.0; input2[i] = 3.0; }

        var output = layer.Forward(input1, input2);

        for (int i = 0; i < output.Length; i++)
            Assert.True(System.Math.Abs(output[i] - 6.0) < 1e-10);
    }
}
