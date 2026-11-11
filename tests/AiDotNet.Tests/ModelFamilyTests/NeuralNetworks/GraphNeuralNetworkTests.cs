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

    // A 4-node path 0-1-2-3, Kipf-Welling normalized: D^-1/2 (A + I) D^-1/2 with degrees 2, 3, 3, 2.
    private static Tensor<float> NormalizedPathGraph()
    {
        int[] degree = [2, 3, 3, 2];
        var adjacency = new Tensor<float>([4, 4]);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (Math.Abs(i - j) <= 1)
                    adjacency[i, j] = (float)(1.0 / Math.Sqrt(degree[i] * degree[j]));
            }
        }

        return adjacency;
    }

    private static GraphNeuralNetwork<float> SmallNetwork() => new(new GraphNeuralNetworkOptions
    {
        NodeFeatureSize = 3,
        NumClasses = 2,
        HiddenSize = 5,
        DropoutRate = 0.0,
        Seed = 7
    });

    private static Tensor<float> NodeFeatures()
    {
        var features = new Tensor<float>([4, 3]);
        for (int i = 0; i < features.Length; i++)
            features[i / 3, i % 3] = (float)Math.Sin(i + 1);
        return features;
    }

    [Fact]
    public void Predict_UsesTheGraphSetThroughSetAdjacencyMatrix()
    {
        using var model = SmallNetwork();
        var features = NodeFeatures();
        var graph = NormalizedPathGraph();

        var isolated = model.Predict(features).ToVector().ToArray();
        model.SetAdjacencyMatrix(graph);
        var connected = model.Predict(features).ToVector().ToArray();
        var explicitGraph = model.PredictGraph(features, graph).ToVector().ToArray();

        // Predict over the set graph is exactly PredictGraph over the same graph...
        Assert.Equal(explicitGraph.Length, connected.Length);
        for (int i = 0; i < connected.Length; i++)
            Assert.Equal(explicitGraph[i], connected[i], precision: 6);

        // ...and not the identity graph, under which a node never sees its neighbours.
        Assert.Contains(Enumerable.Range(0, connected.Length), i => Math.Abs(connected[i] - isolated[i]) > 1e-4f);

        model.SetAdjacencyMatrix(null);
        var cleared = model.Predict(features).ToVector().ToArray();
        for (int i = 0; i < cleared.Length; i++)
            Assert.Equal(isolated[i], cleared[i], precision: 6);
    }

    [Fact]
    public void Train_PropagatesOverTheGraphSetThroughSetAdjacencyMatrix()
    {
        var features = NodeFeatures();
        var target = new Tensor<float>([4, 2]);
        for (int node = 0; node < 4; node++)
            target[node, node % 2] = 1.0f;

        using var isolated = SmallNetwork();
        using var connected = SmallNetwork();
        connected.SetAdjacencyMatrix(NormalizedPathGraph());
        var start = isolated.GetParameters().ToArray();
        Assert.Equal(start, connected.GetParameters().ToArray());

        isolated.Train(features, target);
        connected.Train(features, target);

        // Same seed, data and target: the updates differ only if training saw the graph.
        var a = isolated.GetParameters().ToArray();
        var b = connected.GetParameters().ToArray();
        Assert.NotEqual(start, b);
        Assert.Contains(Enumerable.Range(0, a.Length), i => Math.Abs(a[i] - b[i]) > 1e-7f);
    }

    [Fact]
    public void Clone_KeepsTheGraphSetThroughSetAdjacencyMatrix()
    {
        using var model = SmallNetwork();
        model.SetAdjacencyMatrix(NormalizedPathGraph());
        var features = NodeFeatures();

        using var clone = Assert.IsType<GraphNeuralNetwork<float>>(model.Clone());

        var expected = model.Predict(features).ToVector().ToArray();
        var actual = clone.Predict(features).ToVector().ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 6);
    }

    [Fact]
    public void SetAdjacencyMatrix_RejectsANonSquareMatrixAndANodeCountMismatch()
    {
        using var model = SmallNetwork();

        Assert.Throws<ArgumentException>(() => model.SetAdjacencyMatrix(new Tensor<float>([4, 3])));

        model.SetAdjacencyMatrix(NormalizedPathGraph());
        Assert.Throws<ArgumentException>(() => model.Predict(new Tensor<float>([5, 3])));
    }

    [Fact]
    public void SaveAndLoad_KeepsTheGraphSetThroughSetAdjacencyMatrix()
    {
        // The graph lives in the graph layers as input-sized state, outside the parameter vector. The network format
        // stored only each layer's vector and extras, so a loaded copy, and every clone that fell back to the same
        // round trip, came back with an all-zero adjacency and predicted a uniform distribution.
        using var model = SmallNetwork();
        model.SetAdjacencyMatrix(NormalizedPathGraph());
        var features = NodeFeatures();
        var bytes = model.Serialize();

        using var loaded = SmallNetwork();
        loaded.Deserialize(bytes);

        var expected = model.Predict(features).ToVector().ToArray();
        var actual = loaded.Predict(features).ToVector().ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 6);
    }
}
