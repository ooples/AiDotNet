using System.Linq.Expressions;
using AiDotNet.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

/// <summary>
/// Round-trip tests for the expression-tree description of a delegate.
/// </summary>
/// <remarks>
/// These compile the rebuilt tree and compare its output to the original's, because a tree that
/// serializes is not evidence it rebuilds the same function. The allowlist test matters most: it is
/// the one property that separates this from Keras's Lambda layer.
/// </remarks>
public class ExpressionStateTests
{
    private static Tensor<double> Probe(params double[] values)
    {
        var tensor = new Tensor<double>([values.Length]);
        for (var i = 0; i < values.Length; i++) tensor[i] = values[i];
        return tensor;
    }

    [Fact]
    public void A_closure_over_a_captured_constant_round_trips()
    {
        // The case a method reference cannot describe and a traced graph declines: the captured
        // scale is a constant leaf. In a tree it is simply a node.
        var scale = 3.0;
        Expression<Func<double, double>> expression = x => x * scale + 1.0;

        var saved = ExpressionState.Save(expression);
        Assert.NotEqual(string.Empty, saved);

        var restored = ExpressionState.Load<Func<double, double>>(saved, "TestLayer", "expression").Compile();

        Assert.Equal(expression.Compile()(2.0), restored(2.0), precision: 10);
        Assert.Equal(7.0, restored(2.0), precision: 10);
    }

    [Fact]
    public void A_call_into_an_allowed_type_round_trips()
    {
        Expression<Func<double, double>> expression = x => Math.Sqrt(x);

        var saved = ExpressionState.Save(expression);
        Assert.Contains("Sqrt", saved);

        var restored = ExpressionState.Load<Func<double, double>>(saved, "TestLayer", "expression").Compile();
        Assert.Equal(3.0, restored(9.0), precision: 10);
    }

    [Fact]
    public void A_call_into_a_type_outside_the_allowlist_is_refused_before_it_is_compiled()
    {
        // System.IO is not on the allowlist, so a saved model naming it cannot reach Compile().
        // This is the hazard that makes loading a Keras Lambda layer arbitrary code execution.
        Expression<Func<string, bool>> expression = p => System.IO.File.Exists(p);

        var saved = ExpressionState.Save(expression);
        Assert.NotEqual(string.Empty, saved);
        Assert.False(ExpressionState.IsAllowed(typeof(System.IO.File)));

        var failure = Assert.Throws<InvalidOperationException>(
            () => ExpressionState.Load<Func<string, bool>>(saved, "TestLayer", "expression"));

        Assert.Contains("not a type a restored expression is allowed to call", failure.Message);
    }

    [Fact]
    public void A_captured_object_abandons_the_tree_rather_than_saving_a_partial_one()
    {
        // The captured tensor is a constant of a non-primitive type; serializing it whole is not
        // what construction state means, so the tree declines and the caller falls to another tier.
        var captured = Probe(1.0, 2.0);
        Expression<Func<int, Tensor<double>>> expression = _ => captured;

        Assert.Equal(string.Empty, ExpressionState.Save(expression));
    }

    [Fact]
    public void The_allowlist_admits_the_types_that_define_the_layers()
    {
        Assert.True(ExpressionState.IsAllowed(typeof(LayerStateBag)));
        Assert.True(ExpressionState.IsAllowed(typeof(Math)));
        Assert.False(ExpressionState.IsAllowed(typeof(System.Diagnostics.Process)));
    }
}
