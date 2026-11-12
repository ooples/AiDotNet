using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Graph;
using AiDotNet.Models.Options;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// R-GCN (Schlichtkrull et al. 2018) against its paper: the layer rule of Eq. 2,
/// h_i' = ReLU(W_0 h_i + sum_r sum_{j in N_i^r} (1/c_{i,r}) W_r h_j), the basis decomposition of Eq. 3,
/// W_r = sum_b a_rb V_b, the block decomposition of Eq. 4, and training of those weights by backpropagation.
/// </summary>
public class RelationalGCNPaperTests
{
    private const double Tolerance = 1e-12;

    // Relation 0: node 0 receives from nodes 1 and 2, node 1 from node 0. Relation 1: node 2 receives
    // from nodes 0 and 1. Entry [r, i, j] is the edge of type r along which i receives from j.
    private static Tensor<double> TinyGraph()
    {
        var edges = new double[2 * 3 * 3];
        edges[0 * 9 + 0 * 3 + 1] = 1;
        edges[0 * 9 + 0 * 3 + 2] = 1;
        edges[0 * 9 + 1 * 3 + 0] = 1;
        edges[1 * 9 + 2 * 3 + 0] = 1;
        edges[1 * 9 + 2 * 3 + 1] = 1;
        return new Tensor<double>(new[] { 2, 3, 3 }, new Vector<double>(edges));
    }

    private static RelationalGCN<double> CreateModel(
        Tensor<double> adjacency,
        int numNodes,
        int hiddenDimension,
        int numLayers,
        string aggregation = "sum",
        bool useBasis = true,
        bool useBlocks = false,
        int numBlocks = 1,
        int sequenceLength = 1,
        double regularization = 0.0)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: sequenceLength,
            outputSize: 1);
        var options = new RelationalGCNOptions<double>
        {
            NumNodes = numNodes,
            NumRelations = 2,
            NumFeatures = 1,
            SequenceLength = sequenceLength,
            ForecastHorizon = 1,
            HiddenDimension = hiddenDimension,
            NumLayers = numLayers,
            NumBases = 2,
            NumBlocks = numBlocks,
            UseBasisDecomposition = useBasis,
            UseBlockDecomposition = useBlocks,
            UseSelfLoop = true,
            Aggregation = aggregation,
            DropoutRate = 0.0,
            Regularization = regularization,
            Seed = 7,
        };
        return new RelationalGCN<double>(architecture, options, adjacency);
    }

    private static void Fill(Tensor<double> tensor, params double[] values)
    {
        Assert.Equal(values.Length, tensor.Length);
        for (int i = 0; i < values.Length; i++)
            tensor[i] = values[i];
    }

    /// <summary>
    /// Eq. 2 with Eq. 3 on a three-node, two-relation graph, worked by hand (row-vector convention, h W):
    /// W_r0 = 1 V_0 + 0.5 V_1 = [[1, 2.5], [0.5, 1]] and W_r1 = -1 V_0 + 2 V_1 = [[-1, 0], [2, -1]].
    /// Mean: node 0 = (0.5, 1) W_r0 + (0.5, 0) = (1.5, 2.25); node 1 = (1, 0) W_r0 + (0, 0.25) = (1, 2.75);
    /// node 2 = ReLU((0.5, 0.5) W_r1 + (0.5, 0.25)) = ReLU(1, -0.25) = (1, 0).
    /// Sum: node 0 = (1, 2) W_r0 + (0.5, 0) = (2.5, 4.5); node 1 as before; node 2 = ReLU((1, 1) W_r1 +
    /// (0.5, 0.25)) = ReLU(1.5, -0.75) = (1.5, 0).
    /// </summary>
    [Theory]
    [InlineData("mean", new[] { 1.5, 2.25, 1.0, 2.75, 1.0, 0.0 })]
    [InlineData("sum", new[] { 2.5, 4.5, 1.0, 2.75, 1.5, 0.0 })]
    public void Relational_layer_matches_the_hand_computed_tiny_graph(string aggregation, double[] expected)
    {
        var model = CreateModel(TinyGraph(), numNodes: 3, hiddenDimension: 2, numLayers: 1, aggregation: aggregation);
        var weights = model.RelationalWeightTensors;
        Assert.Equal(3, weights.Count);
        Fill(weights[0], 1, 2, 0, 1, 0, 1, 1, 0);
        Fill(weights[1], 1, 0.5, -1, 2);
        Fill(weights[2], 0.5, 0, 0, 0.25);

        var states = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 }));
        var next = model.ApplyRelationalLayer(states, 0);

        Assert.Equal(new[] { 3, 2 }, next.Shape.ToArray());
        var actual = next.ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], Tolerance);
    }

    /// <summary>
    /// Eq. 4: under block decomposition every W_r is block-diagonal, so every entry off the NumBlocks diagonal
    /// blocks is zero and every entry on them is the trained value.
    /// </summary>
    [Fact]
    public void Block_decomposition_keeps_only_the_diagonal_blocks()
    {
        var model = CreateModel(TinyGraph(), numNodes: 3, hiddenDimension: 4, numLayers: 1, useBlocks: true, numBlocks: 2);
        var raw = model.RelationalWeightTensors[0].ToArray();
        var stacked = model.RelationWeights(0).ToArray();

        Assert.Equal(raw.Length, stacked.Length);
        for (int index = 0; index < raw.Length; index++)
        {
            int row = index / 4 % 4;
            int column = index % 4;
            bool onBlock = row / 2 == column / 2;
            Assert.Equal(onBlock ? raw[index] : 0.0, stacked[index], Tolerance);
        }
    }

    /// <summary>
    /// Every R-GCN weight -- each layer's bases V_b, coefficients a_rb and self-connection W_0 -- receives a
    /// nonzero gradient through the training forward. The positive control is the model's dense encoder and head:
    /// they sit on the same tape, so a harness failure would show up there too.
    /// </summary>
    [Fact]
    public void Gradients_reach_every_relational_weight_and_the_dense_control()
    {
        var model = CreateModel(TinyGraph(), numNodes: 3, hiddenDimension: 4, numLayers: 2, sequenceLength: 2);
        var input = new Tensor<double>(new[] { 3, 2, 1 }, new Vector<double>(new[] { 0.3, 0.9, 1.2, 0.4, 0.8, 1.5 }));
        model.SetTrainingMode(true);
        model.ForwardForTraining(input);

        var relational = model.RelationalWeightTensors;
        Assert.Equal(6, relational.Count);
        var control = model.GetParameterStateChunks()
            .Where(chunk => chunk.Role == ParameterSlotRole.Trainable)
            .Select(chunk => chunk.Tensor)
            .Where(tensor => !relational.Any(weight => ReferenceEquals(weight, tensor)))
            .ToList();
        Assert.NotEmpty(control);

        using var tape = new GradientTape<double>();
        var output = model.ForwardForTraining(input);
        var loss = AiDotNetEngine.Current.ReduceSum(output, null);
        var gradients = tape.ComputeGradients(loss, relational.Concat(control).ToArray());

        bool HasSignal(Tensor<double> tensor) =>
            gradients.TryGetValue(tensor, out var gradient) && gradient is not null
            && gradient.ToArray().Any(value => Math.Abs(value) > 0)
            && gradient.ToArray().All(value => !double.IsNaN(value) && !double.IsInfinity(value));

        Assert.True(control.Any(HasSignal), "The positive control (the dense encoder and head) received no gradient.");
        for (int i = 0; i < relational.Count; i++)
            Assert.True(HasSignal(relational[i]), $"R-GCN weight tensor {i} received no gradient.");
    }

    /// <summary>
    /// A training step moves the R-GCN weights by their gradients: before the fix they were only perturbed by
    /// random noise after the step.
    /// </summary>
    [Fact]
    public void A_training_step_updates_the_relational_weights()
    {
        var model = CreateModel(TinyGraph(), numNodes: 3, hiddenDimension: 4, numLayers: 1, sequenceLength: 2);
        var input = new Tensor<double>(new[] { 3, 2, 1 }, new Vector<double>(new[] { 0.3, 0.9, 1.2, 0.4, 0.8, 1.5 }));
        var target = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 1.0, -1.0, 2.0 }));
        var before = model.RelationalWeightTensors.Select(weight => weight.ToArray()).ToList();

        model.Train(input, target);

        var after = model.RelationalWeightTensors.Select(weight => weight.ToArray()).ToList();
        Assert.Contains(Enumerable.Range(0, before.Count), i => !before[i].SequenceEqual(after[i]));
    }

    /// <summary>
    /// The relation adjacencies must match the configured relations and nodes.
    /// </summary>
    [Fact]
    public void Adjacency_with_the_wrong_shape_is_rejected()
    {
        var wrong = new Tensor<double>(new[] { 2, 4, 4 });
        Assert.Throws<ArgumentException>(() => CreateModel(wrong, numNodes: 3, hiddenDimension: 2, numLayers: 1));
    }
}
