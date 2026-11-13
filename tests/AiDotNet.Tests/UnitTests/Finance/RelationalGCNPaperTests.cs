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
        RelationalAggregation aggregation = RelationalAggregation.Sum,
        bool useBasis = true,
        bool useBlocks = false,
        int numBlocks = 1,
        int sequenceLength = 1,
        double regularization = 0.0,
        bool useSelfLoop = true,
        double dropout = 0.0,
        int seed = 7,
        AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? optimizer = null)
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
            UseSelfLoop = useSelfLoop,
            Aggregation = aggregation,
            DropoutRate = dropout,
            Regularization = regularization,
            Seed = seed,
        };
        return new RelationalGCN<double>(architecture, options, adjacency, optimizer);
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
    [InlineData(RelationalAggregation.Mean, new[] { 1.5, 2.25, 1.0, 2.75, 1.0, 0.0 })]
    [InlineData(RelationalAggregation.Sum, new[] { 2.5, 4.5, 1.0, 2.75, 1.5, 0.0 })]
    public void Relational_layer_matches_the_hand_computed_tiny_graph(
        RelationalAggregation aggregation, double[] expected)
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
    /// blocks is zero and every entry on them is the trained value. Only the blocks are stored: 2 relations x
    /// 2 blocks of 2 x 2 are 16 trainable values, half of the 32 that full 4 x 4 relation weights would take.
    /// </summary>
    [Fact]
    public void Block_decomposition_keeps_only_the_diagonal_blocks()
    {
        var model = CreateModel(TinyGraph(), numNodes: 3, hiddenDimension: 4, numLayers: 1, useBlocks: true, numBlocks: 2);
        var raw = model.RelationalWeightTensors[0].ToArray();
        var stacked = model.RelationWeights(0).ToArray();

        Assert.Equal(new[] { 2, 2, 2, 2 }, model.RelationalWeightTensors[0].Shape.ToArray());
        Assert.Equal(2 * 4 * 4, stacked.Length);
        for (int index = 0; index < stacked.Length; index++)
        {
            int relation = index / 16;
            int row = index / 4 % 4;
            int column = index % 4;
            bool onBlock = row / 2 == column / 2;
            double stored = raw[relation * 8 + row / 2 * 4 + row % 2 * 2 + column % 2];
            Assert.Equal(onBlock ? stored : 0.0, stacked[index], Tolerance);
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
    /// A plain SGD training step moves every R-GCN weight tensor (the bases, the coefficients and the
    /// self-connection) along its negative loss gradient. Before the fix they were only perturbed by random
    /// noise, which is nearly orthogonal to the gradient: cosine near 0, not near 1.
    /// </summary>
    [Fact]
    public void A_training_step_moves_every_relational_weight_along_its_negative_gradient()
    {
        var sgd = new AiDotNet.Optimizers.StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });
        var model = CreateModel(TinyGraph(), numNodes: 3, hiddenDimension: 4, numLayers: 1, sequenceLength: 2, optimizer: sgd);
        var input = new Tensor<double>(new[] { 3, 2, 1 }, new Vector<double>(new[] { 0.3, 0.9, 1.2, 0.4, 0.8, 1.5 }));
        var target = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 1.0, -1.0, 2.0 }));
        var relational = model.RelationalWeightTensors;
        Assert.Equal(3, relational.Count);
        var before = relational.Select(weight => weight.ToArray()).ToList();

        // The squared-error gradient at the starting weights; MSE differs only by a positive factor.
        model.SetTrainingMode(true);
        double[][] gradient;
        using (var tape = new GradientTape<double>())
        {
            var engine = AiDotNetEngine.Current;
            var error = engine.TensorSubtract(model.ForwardForTraining(input), target);
            var loss = engine.ReduceSum(engine.TensorMultiply(error, error), null);
            var gradients = tape.ComputeGradients(loss, relational.ToArray());
            gradient = relational.Select(weight => gradients[weight].ToArray()).ToArray();
        }

        model.Train(input, target);

        for (int i = 0; i < relational.Count; i++)
        {
            var step = relational[i].ToArray().Zip(before[i], (after, start) => after - start).ToArray();
            double dot = step.Zip(gradient[i], (a, b) => a * b).Sum();
            double norms = Math.Sqrt(step.Sum(v => v * v)) * Math.Sqrt(gradient[i].Sum(v => v * v));
            Assert.True(norms > 0, $"R-GCN weight tensor {i} did not move or had no gradient.");
            Assert.True(-dot / norms > 0.9, $"R-GCN weight tensor {i} moved off its negative gradient (cosine {-dot / norms}).");
        }
    }

    /// <summary>A NaN or infinite edge weight is rejected before it can reach normalization or messages.</summary>
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Non_finite_adjacency_weights_are_rejected(double weight)
    {
        var adjacency = TinyGraph();
        adjacency[0, 0, 1] = weight;
        Assert.Throws<ArgumentException>(() => CreateModel(adjacency, numNodes: 3, hiddenDimension: 2, numLayers: 1));
    }

    /// <summary>
    /// A Sum checkpoint loaded into a model built for Mean propagates by Sum: the aggregation is restored with
    /// the options instead of kept from the receiving constructor.
    /// </summary>
    [Fact]
    public void Restoring_a_sum_checkpoint_into_a_mean_model_propagates_by_sum()
    {
        var adjacency = new Tensor<double>(new[] { 2, 2, 2 });
        adjacency[0, 0, 1] = 2;
        using var source = CreateModel(adjacency, 2, 2, 1, RelationalAggregation.Sum, useBasis: false, useSelfLoop: false);
        using var restored = CreateModel(adjacency, 2, 2, 1, RelationalAggregation.Mean, useBasis: false, useSelfLoop: false);
        restored.Deserialize(source.Serialize());
        Fill(restored.RelationalWeightTensors[0], 1, 0, 0, 1, 9, 9, 9, 9);
        var states = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }));
        // Sum: node 0 receives 2 * h_1 = (6, 8); Mean would give (3, 4).
        Assert.Equal(new[] { 6.0, 8.0, 0.0, 0.0 }, restored.ApplyRelationalLayer(states, 0).ToArray());
    }

    /// <summary>
    /// The ONNX forward returns the native [numNodes, forecastHorizon] shape. The ONNX graph is a single Identity
    /// node, so with SequenceLength = ForecastHorizon and one feature it hands every node history back as its
    /// forecast.
    /// </summary>
    [Fact]
    public void Onnx_forward_returns_the_native_node_by_horizon_shape()
    {
        string path = System.IO.Path.GetTempFileName();
        try
        {
            System.IO.File.WriteAllBytes(path, IdentityOnnxModel());
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple, inputSize: 2, outputSize: 1);
            var options = new RelationalGCNOptions<double>
            {
                NumNodes = 3, NumRelations = 2, NumFeatures = 1, SequenceLength = 2, ForecastHorizon = 2,
            };
            using var model = new RelationalGCN<double>(architecture, path, options, TinyGraph());
            var input = new Tensor<double>(new[] { 3, 2, 1 }, new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
            var output = model.Forward(input);
            Assert.Equal(new[] { 3, 2 }, output.Shape.ToArray());
            Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, output.ToArray());
        }
        finally
        {
            System.IO.File.Delete(path);
        }
    }

    private static byte[] Varint(ulong value)
    {
        var bytes = new System.Collections.Generic.List<byte>();
        do
        {
            byte b = (byte)(value & 0x7F);
            value >>= 7;
            bytes.Add(value != 0 ? (byte)(b | 0x80) : b);
        }
        while (value != 0);
        return bytes.ToArray();
    }

    private static byte[] Field(int number, byte[] payload) =>
        Varint((ulong)(number << 3 | 2)).Concat(Varint((ulong)payload.Length)).Concat(payload).ToArray();

    private static byte[] Field(int number, string text) => Field(number, System.Text.Encoding.UTF8.GetBytes(text));

    private static byte[] Number(int number, ulong value) => Varint((ulong)(number << 3)).Concat(Varint(value)).ToArray();

    /// <summary>A ModelProto (IR 8, opset 13) whose graph maps float "input" to "output" through one Identity node.</summary>
    private static byte[] IdentityOnnxModel()
    {
        byte[] FloatTensor(string name) => Field(1, name).Concat(Field(2, Field(1, Number(1, 1)))).ToArray();
        var node = Field(1, "input").Concat(Field(2, "output")).Concat(Field(4, "Identity")).ToArray();
        var graph = Field(1, node).Concat(Field(2, "identity"))
            .Concat(Field(11, FloatTensor("input"))).Concat(Field(12, FloatTensor("output"))).ToArray();
        var opset = Field(1, "").Concat(Number(2, 13)).ToArray();
        return Number(1, 8).Concat(Field(7, graph)).Concat(Field(8, opset)).ToArray();
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

    [Theory]
    [InlineData(RelationalAggregation.Sum, 6.0, 8.0)]
    [InlineData(RelationalAggregation.Mean, 3.0, 4.0)]
    public void Full_relation_weights_without_self_loops_handle_weighted_edges_and_isolated_nodes(
        RelationalAggregation aggregation, double first, double second)
    {
        var adjacency = new Tensor<double>(new[] { 2, 2, 2 });
        adjacency[0, 0, 1] = 2;
        using var model = CreateModel(adjacency, 2, 2, 1, aggregation,
            useBasis: false, useSelfLoop: false);
        Assert.Single(model.RelationalWeightTensors);
        Fill(model.RelationalWeightTensors[0], 1, 0, 0, 1, 9, 9, 9, 9);
        var states = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }));
        var output = model.ApplyRelationalLayer(states, 0);
        Assert.Equal(new[] { first, second, 0.0, 0.0 }, output.ToArray());
        // The model owns its graph: a caller's subsequent edits cannot change propagation.
        adjacency.Fill(0);
        Assert.Equal(output.ToArray(), model.ApplyRelationalLayer(states, 0).ToArray());
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(false, true)]
    public void Save_load_restores_relation_parameters_graph_and_predictions(bool basis, bool blocks)
    {
        using var source = CreateModel(TinyGraph(), 3, 4, 2, useBasis: basis,
            useBlocks: blocks, numBlocks: 2, sequenceLength: 2);
        using var restored = CreateModel(new Tensor<double>(new[] { 2, 3, 3 }), 3, 4, 2,
            useBasis: basis, useBlocks: blocks, numBlocks: 2, sequenceLength: 2, seed: 19);
        var input = new Tensor<double>(new[] { 3, 2, 1 }, new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        source.SetTrainingMode(false);
        restored.SetTrainingMode(false);
        source.Forward(input);
        restored.Forward(input);
        foreach (var chunk in source.GetParameterStateChunks().Where(c => c.Role == ParameterSlotRole.Trainable))
            chunk.Tensor.Fill(0.2);
        foreach (var chunk in restored.GetParameterStateChunks().Where(c => c.Role == ParameterSlotRole.Trainable))
            chunk.Tensor.Fill(0);
        var expected = source.Forward(input).ToArray();
        Assert.Contains(expected, value => value > 0);
        Assert.All(restored.Forward(input).ToArray(), value => Assert.Equal(0, value));
        restored.Deserialize(source.Serialize());
        Assert.Equal(source.GetParameters().ToArray(), restored.GetParameters().ToArray());
        Assert.Equal(expected, restored.Forward(input).ToArray());
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Invalid_regularization_is_rejected(double strength) =>
        Assert.Throws<ArgumentException>(() => CreateModel(TinyGraph(), 3, 2, 1, regularization: strength));

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Invalid_dropout_is_rejected(double probability) =>
        Assert.Throws<ArgumentException>(() => CreateModel(TinyGraph(), 3, 2, 1, dropout: probability));

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Non_positive_history_length_is_rejected(int length) =>
        Assert.Throws<ArgumentException>(() => CreateModel(TinyGraph(), 3, 2, 1, sequenceLength: length));
}
