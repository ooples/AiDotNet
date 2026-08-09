using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphNeuralNetworkTests : GraphNNModelTestBase<float>
{
    // GraphNeuralNetwork default ctor (Kipf & Welling 2017 "Semi-Supervised
    // Classification with Graph Convolutional Networks"): Cora's 1433 input
    // features → 7 classes, 10 nodes in this synthetic graph.
    protected override int[] InputShape => [10, 1433];
    protected override int[] OutputShape => [10, 7];

    // Kipf and Welling train node classification with categorical labels. The
    // generic NN scaffold's continuous random target is a regression target,
    // not a valid Cora-style class distribution. Give every synthetic node one
    // class so the invariant exercises the paper's softmax cross-entropy task.
    protected override Tensor<float> CreateRandomTargetTensor(int[] shape, Random rng)
    {
        var target = new Tensor<float>(shape);
        int numClasses = shape[^1];
        int numNodes = target.Length / numClasses;
        for (int node = 0; node < numNodes; node++)
            target[node * numClasses + rng.Next(numClasses)] = 1.0f;
        return target;
    }

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new GraphNeuralNetwork<float>();

    [Fact]
    public void Defaults_ShouldMatchKipfWellingReferenceConfiguration()
    {
        var options = new GraphNeuralNetworkOptions();

        Assert.Equal(1433, options.NodeFeatureSize);
        Assert.Equal(7, options.NumClasses);
        Assert.Equal(16, options.HiddenSize);
        Assert.Equal(2, options.NumLayers);
        Assert.Equal(0.5, options.DropoutRate);
        Assert.Equal(0.01, options.LearningRate);
        Assert.Equal(5e-4, options.L2Regularization);
        Assert.False(options.UseBias);
        Assert.False(options.UseAuxiliaryLoss);

        using var model = new GraphNeuralNetwork<float>(options);
        Assert.Equal(1433L * 16L + 16L * 7L, model.ParameterCount);
    }

    [Fact]
    public void Options_ShouldFullyCustomizeDefaultArchitecture()
    {
        var options = new GraphNeuralNetworkOptions
        {
            NodeFeatureSize = 4,
            NumClasses = 3,
            HiddenSize = 8,
            NumLayers = 3,
            DropoutRate = 0.25,
            LearningRate = 0.02,
            L2Regularization = 0.001,
            UseBias = true,
            UseAuxiliaryLoss = true,
            AuxiliaryLossWeight = 0.07,
            Seed = 42
        };

        using var model = new GraphNeuralNetwork<float>(options);
        var graphLayers = model.Layers.OfType<GraphConvolutionalLayer<float>>().ToArray();

        Assert.Equal(3, graphLayers.Length);
        Assert.Collection(
            graphLayers,
            layer =>
            {
                Assert.Equal(4, layer.InputFeatures);
                Assert.Equal(8, layer.OutputFeatures);
            },
            layer =>
            {
                Assert.Equal(8, layer.InputFeatures);
                Assert.Equal(8, layer.OutputFeatures);
            },
            layer =>
            {
                Assert.Equal(8, layer.InputFeatures);
                Assert.Equal(3, layer.OutputFeatures);
            });
        Assert.All(graphLayers, layer => Assert.True(layer.UseBias));
        Assert.Equal(139L, model.ParameterCount);
        Assert.True(model.UseAuxiliaryLoss);
        Assert.Equal(0.07f, model.AuxiliaryLossWeight, precision: 6);

        using var cloned = Assert.IsType<GraphNeuralNetwork<float>>(model.Clone());
        var clonedOptions = Assert.IsType<GraphNeuralNetworkOptions>(cloned.GetOptions());
        Assert.NotSame(model.GetOptions(), clonedOptions);
        Assert.Equal(4, clonedOptions.NodeFeatureSize);
        Assert.Equal(3, clonedOptions.NumClasses);
        Assert.Equal(8, clonedOptions.HiddenSize);
        Assert.Equal(3, clonedOptions.NumLayers);
        Assert.Equal(0.25, clonedOptions.DropoutRate);
        Assert.Equal(0.02, clonedOptions.LearningRate);
        Assert.Equal(0.001, clonedOptions.L2Regularization);
        Assert.True(clonedOptions.UseBias);
        Assert.Equal(139L, cloned.ParameterCount);
        Assert.All(
            cloned.Layers.OfType<GraphConvolutionalLayer<float>>(),
            layer => Assert.True(layer.UseBias));
    }
}
