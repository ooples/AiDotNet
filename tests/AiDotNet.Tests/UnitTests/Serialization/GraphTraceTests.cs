using AiDotNet.Autodiff;
using AiDotNet.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

/// <summary>
/// Round-trip tests for the traced-graph description of a delegate.
/// </summary>
/// <remarks>
/// These run the trace and the replay, because a graph that serializes without throwing is not
/// evidence that it rebuilds the same function -- which is the only thing it exists to do.
/// </remarks>
public class GraphTraceTests
{
    private static Tensor<double> Probe(params double[] values)
    {
        var tensor = new Tensor<double>([values.Length]);
        for (var i = 0; i < values.Length; i++) tensor[i] = values[i];
        return tensor;
    }

    private static Tensor<double> Run(Func<ComputationNode<double>, ComputationNode<double>> f, Tensor<double> input)
        => f(TensorOperations<double>.Variable(input)).Value;

    [Fact]
    public void Trace_records_a_closure_that_has_no_name_to_refer_to()
    {
        // The case a method reference cannot describe: the delegate is a lambda, so tier 3 declines.
        Func<ComputationNode<double>, ComputationNode<double>> expression =
            x => TensorOperations<double>.Square(TensorOperations<double>.Tanh(x));

        var graph = GraphTrace.Trace(expression, [3]);

        Assert.NotNull(graph);
        Assert.Contains("Tanh", graph);
        Assert.Contains("Square", graph);
    }

    [Fact]
    public void Replayed_graph_computes_what_the_original_computed()
    {
        Func<ComputationNode<double>, ComputationNode<double>> expression =
            x => TensorOperations<double>.Square(TensorOperations<double>.Tanh(x));

        var graph = GraphTrace.Trace(expression, [3]);
        var replayed = GraphTrace.Compile<double>(graph!, "TestLayer", "expression");

        var input = Probe(0.25, -1.5, 2.0);
        var expected = Run(expression, input);
        var actual = Run(replayed, input);

        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(expected[i], actual[i], precision: 10);
        }
    }

    [Fact]
    public void Operation_parameters_survive_the_round_trip()
    {
        // Softmax records its axis in OperationParams under "Axis" while the parameter is "axis",
        // so this is what proves the binding is case-insensitive rather than accidentally aligned.
        Func<ComputationNode<double>, ComputationNode<double>> expression =
            x => TensorOperations<double>.Softmax(x, axis: -1);

        var graph = GraphTrace.Trace(expression, [4]);
        Assert.NotNull(graph);

        var replayed = GraphTrace.Compile<double>(graph!, "TestLayer", "expression");
        var input = Probe(1.0, 2.0, 3.0, 4.0);

        var expected = Run(expression, input);
        var actual = Run(replayed, input);

        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(expected[i], actual[i], precision: 10);
        }
    }

    [Fact]
    public void A_captured_constant_abandons_the_graph_rather_than_saving_a_partial_one()
    {
        // The captured tensor is a leaf that is not the input, so its value is not recoverable from
        // the graph. Declining sends the caller to a weaker tier instead of rebuilding a different
        // function, which is the whole point of recording all-or-nothing.
        var captured = TensorOperations<double>.Constant(Probe(2.0, 2.0, 2.0));
        Func<ComputationNode<double>, ComputationNode<double>> expression =
            x => TensorOperations<double>.Add(x, captured);

        Assert.Null(GraphTrace.Trace(expression, [3]));
    }

    [Fact]
    public void DelegateState_falls_through_to_the_graph_when_the_delegate_has_no_name()
    {
        Func<ComputationNode<double>, ComputationNode<double>> expression =
            x => TensorOperations<double>.Tanh(x);

        // Tier 3 cannot describe a lambda at all.
        Assert.Equal(string.Empty, DelegateState.Save(expression));

        var saved = DelegateState.SaveTraceable(expression, [3]);
        Assert.StartsWith(DelegateState.GraphScheme, saved);

        var restored = DelegateState.Load<Func<ComputationNode<double>, ComputationNode<double>>>(
            saved, "TestLayer", "expression");

        var input = Probe(0.5, -0.5, 1.25);
        var expected = Run(expression, input);
        var actual = Run(restored, input);

        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(expected[i], actual[i], precision: 10);
        }
    }
}
