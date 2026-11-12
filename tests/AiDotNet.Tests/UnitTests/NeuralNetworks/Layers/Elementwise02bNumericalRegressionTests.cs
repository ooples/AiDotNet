using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class Elementwise02bNumericalRegressionTests
{
    [Theory]
    [InlineData("backward")]
    [InlineData("bilinear")]
    public void HalfHippoSequence_SolvesBothForwardAndBackwardRecurrences(string mode)
    {
        using var layer = new HippoMemoryCellLayer<Half>(hiddenSize: 1, inputSize: 1, memoryOrder: 1,
            measure: "legs", discretization: mode, timeStep: 0.25, useGate: false);
        var input = new Tensor<Half>(new Half[] { (Half)0.25, (Half)0.5 }, new[] { 2, 1 });
        using var tape = new GradientTape<Half>();
        var output = layer.Forward(input);
        Assert.All(output.ToArray(), value => Assert.True(Half.IsFinite(value)));
        var loss = AiDotNetEngine.Current.ReduceSum(output, new[] { 0, 1 }, false);
        var gradients = tape.ComputeGradients(loss, new[] { input });
        Assert.True(gradients.TryGetValue(input, out var gradient));
        Assert.NotNull(gradient);
        Assert.Equal(new[] { 2, 1 }, gradient.Shape.ToArray());
        Assert.All(gradient.ToArray(), value => Assert.True(Half.IsFinite(value)));
    }

    [Theory]
    [InlineData("forward")]
    [InlineData("backward")]
    [InlineData("bilinear")]
    [InlineData("zoh")]
    public void HippoScalarTransitions_MatchClosedFormsInDoubleAndFloat(string mode)
    {
        double state = mode switch { "forward" => 0.75, "backward" => 0.8,
            "bilinear" => 7.0 / 9.0, _ => Math.Exp(-0.25) };
        double input = mode switch { "forward" => 0.25, "backward" => 0.2,
            "bilinear" => 2.0 / 9.0, _ => 1 - Math.Exp(-0.25) };
        var precise = HippoTransition<double>(mode);
        var single = HippoTransition<float>(mode);
        Assert.InRange(Math.Abs(precise.State - state), 0, 1e-12);
        Assert.InRange(Math.Abs(precise.Input - input), 0, 1e-12);
        Assert.InRange(Math.Abs(single.State - state), 0, 1e-5);
        Assert.InRange(Math.Abs(single.Input - input), 0, 1e-5);
    }

    [Theory]
    [InlineData("forward", 0.75, 0.25)]
    [InlineData("backward", 0.8, 0.2)]
    [InlineData("bilinear", 0.7777777777777778, 0.2222222222222222)]
    [InlineData("zoh", 0.7788007830714049, 0.22119921692859512)]
    public void HippoScalarTransitions_HalfRemainsSupported(string mode, double state, double input)
    {
        var actual = HippoTransition<Half>(mode);
        Assert.InRange(Math.Abs(actual.State - state), 0, 0.002);
        Assert.InRange(Math.Abs(actual.Input - input), 0, 0.002);
    }

    private static (double State, double Input) HippoTransition<T>(string mode)
    {
        using var layer = new HippoMemoryCellLayer<T>(hiddenSize: 1, inputSize: 1, memoryOrder: 1,
            measure: "legs", discretization: mode, timeStep: 0.25, useGate: false);
        var method = typeof(HippoMemoryCellLayer<T>).GetMethod("BuildDiscreteTransition",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        var transition = ((Tensor<T>, Tensor<T>))method.Invoke(layer, new object[] { 0.25 })!;
        return (transition.Item1.Cast<double>().GetFlat(0), transition.Item2.Cast<double>().GetFlat(0));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HalfKosmosAttention_CausalMaskIsFiniteAndBlocksFutureValues(bool xPos)
    {
        var query = new Tensor<Half>(new[] { 2, 2 });
        var key = new Tensor<Half>(new[] { 2, 2 });
        var values = new Tensor<Half>(new Half[] { (Half)1, (Half)2, (Half)8, (Half)16 }, new[] { 2, 2 });
        var actual = MultiHeadAttentionMath<Half>.Attend(query, key, values, heads: 1, causal: true, xPos: xPos);
        Assert.Equal(new Half[] { (Half)1, (Half)2, (Half)4.5, (Half)9 }, actual.ToArray());
    }

    [Fact]
    public void StateCacheCompression_SubnormalFiniteRangePreservesTheEndpoints()
    {
        var cache = new SSMStateCache<float>(enableCompression: true, compressionBitWidth: 8);
        var state = new Tensor<float>(new[] { 0.0f, 1e-40f }, new[] { 2 });
        cache.CacheSSMState(0, state);
        Assert.Equal(state.ToArray(), cache.GetSSMState(0)!.ToArray());
    }
}
