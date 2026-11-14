using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// TrainWithTape and TrainWithGradientAccumulation name the trainable parameters as backward sources, which lets the
/// tape release each intermediate gradient after its last use. Restricted sources once dropped leaves reached through
/// views and recurrent-weight permutes. These tests check that every parameter gets the same gradient from the
/// restricted walk as from the full walk (sources: null). Each tape holds more than 100 entries, so the tape's
/// source-relevance pruning is active on every Tensors build. How many entries the layer itself records depends on the
/// build (one that fuses the recurrence records a handful), so the output is padded with exact identity adds.
/// </summary>
[Collection("NonParallelIntegration")]
public sealed class RestrictedBackwardSourcesParityTests
{
    private const int Batch = 2;
    private const int SequenceLength = 48;
    private const int Features = 3;
    private const int IdentityPadding = 128;

    public static IEnumerable<object[]> RecurrentLayers()
    {
        yield return new object[] { "LSTM", new Func<ILayer<float>>(() => new LSTMLayer<float>(4)) };
        yield return new object[] { "GRU", new Func<ILayer<float>>(() => new GRULayer<float>(4, returnSequences: true)) };
    }

    [Theory]
    [MemberData(nameof(RecurrentLayers))]
    public void RestrictedSources_GiveEveryParameterTheFullWalkGradient(string name, Func<ILayer<float>> make)
    {
        var input = new Tensor<float>(new[] { Batch, SequenceLength, Features });
        for (int i = 0; i < input.Length; i++) input[i] = (float)Math.Sin(0.37 * i);

        var layer = (LayerBase<float>)make();
        layer.SetTrainingMode(true);
        _ = layer.Forward(input);   // resolve the lazy weights before either tape opens
        var parameters = layer.GetTrainableParameters();
        Assert.NotEmpty(parameters);
        for (int p = 0; p < parameters.Count; p++)
            for (int i = 0; i < parameters[p].Length; i++)
                parameters[p][i] = (float)(0.05 * Math.Cos(0.11 * i + p));

        var full = Backward(layer, input, sources: null, out int fullEntries);
        var restricted = Backward(layer, input, sources: parameters, out _);
        Assert.True(fullEntries >= 100, $"{name}: {fullEntries} tape entries leaves relevance pruning off.");

        foreach (var parameter in parameters)
        {
            Assert.True(full.ContainsKey(parameter), $"{name}: the full walk found no gradient.");
            Assert.True(restricted.ContainsKey(parameter), $"{name}: restricted sources dropped a parameter's gradient.");
            var e = full[parameter].AsSpan().ToArray();
            var a = restricted[parameter].AsSpan().ToArray();
            Assert.Contains(e, v => v != 0f);
            Assert.Equal(e.Length, a.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.Equal(e[i], a[i]);
        }
    }

    public static IEnumerable<object[]> OtherLayers()
    {
        yield return new object[] { "Dense", new Func<ILayer<float>>(() => new DenseLayer<float>(5)), new[] { 2, 3 } };
        yield return new object[] { "MultiHeadAttention", new Func<ILayer<float>>(() => new MultiHeadAttentionLayer<float>(2, 4)), new[] { 2, 6, 8 } };
        yield return new object[] { "Convolution", new Func<ILayer<float>>(() => new ConvolutionalLayer<float>(4, 3, 1, 1)), new[] { 2, 3, 8, 8 } };
        yield return new object[] { "BatchNorm", new Func<ILayer<float>>(() => new BatchNormalizationLayer<float>()), new[] { 4, 6 } };
        yield return new object[] { "LayerNorm", new Func<ILayer<float>>(() => new LayerNormalizationLayer<float>()), new[] { 2, 6, 8 } };
    }

    /// <summary>
    /// The layers the training funnel sees most, beyond the recurrent ones: each parameter the full walk reaches must
    /// get the same gradient from the restricted walk, and a parameter the full walk gives no gradient must get none.
    /// </summary>
    [Theory]
    [MemberData(nameof(OtherLayers))]
    public void RestrictedSources_MatchTheFullWalk_ForNonRecurrentLayers(string name, Func<ILayer<float>> make, int[] inputShape)
    {
        var input = new Tensor<float>(inputShape);
        for (int i = 0; i < input.Length; i++) input[i] = (float)Math.Sin(0.37 * i + 0.2);

        var layer = (LayerBase<float>)make();
        layer.SetTrainingMode(true);
        _ = layer.Forward(input);
        var parameters = layer.GetTrainableParameters();
        Assert.NotEmpty(parameters);

        var full = Backward(layer, input, sources: null, out int fullEntries);
        var restricted = Backward(layer, input, sources: parameters, out _);
        Assert.True(fullEntries >= 100, $"{name}: {fullEntries} tape entries leaves relevance pruning off.");

        int reached = 0;
        foreach (var parameter in parameters)
        {
            bool inFull = full.TryGetValue(parameter, out var expected);
            bool inRestricted = restricted.TryGetValue(parameter, out var actual);
            Assert.True(inFull == inRestricted || !inFull,
                $"{name}: restricted sources dropped the gradient of a parameter of shape [{string.Join(",", parameter.Shape.ToArray())}].");
            if (!inFull || expected is null || actual is null) continue;
            reached++;
            var e = expected.AsSpan().ToArray();
            var a = actual.AsSpan().ToArray();
            Assert.Equal(e.Length, a.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.Equal(e[i], a[i]);
        }

        Assert.True(reached > 0, $"{name}: the full walk reached no parameter.");
    }

    /// <summary>
    /// A layer applied twice shares its parameters between two places in the graph: the restricted walk must sum both
    /// contributions, as the full walk does.
    /// </summary>
    [Fact]
    public void RestrictedSources_SumBothUsesOfASharedParameter()
    {
        var input = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)Math.Sin(0.37 * i + 0.2);

        var layer = new DenseLayer<float>(3);
        layer.SetTrainingMode(true);
        _ = layer.Forward(input);
        var parameters = layer.GetTrainableParameters();

        var full = Backward(layer, input, sources: null, out _, applications: 2);
        var restricted = Backward(layer, input, sources: parameters, out _, applications: 2);
        var once = Backward(layer, input, sources: parameters, out _, applications: 1);

        foreach (var parameter in parameters)
        {
            var e = full[parameter].AsSpan().ToArray();
            var a = restricted[parameter].AsSpan().ToArray();
            for (int i = 0; i < e.Length; i++)
                Assert.Equal(e[i], a[i]);
            // The second application really contributes: otherwise this test would pass on one use alone.
            Assert.NotEqual(once[parameter].AsSpan().ToArray(), a);
        }
    }

    /// <summary>
    /// Naming only some parameters (the rest frozen) must not change the gradient of the named ones, and must not
    /// hand back gradients for the frozen ones.
    /// </summary>
    [Fact]
    public void RestrictedSources_OfASubset_LeaveTheNamedGradientsUnchanged()
    {
        var input = new Tensor<float>(new[] { Batch, SequenceLength, Features });
        for (int i = 0; i < input.Length; i++) input[i] = (float)Math.Sin(0.37 * i);

        var layer = new LSTMLayer<float>(4);
        layer.SetTrainingMode(true);
        _ = layer.Forward(input);
        var parameters = layer.GetTrainableParameters();
        Assert.True(parameters.Count >= 2);
        var named = new[] { parameters[0] };

        var full = Backward(layer, input, sources: null, out _);
        var restricted = Backward(layer, input, sources: named, out _);

        var e = full[parameters[0]].AsSpan().ToArray();
        var a = restricted[parameters[0]].AsSpan().ToArray();
        Assert.Contains(e, v => v != 0f);
        for (int i = 0; i < e.Length; i++)
            Assert.Equal(e[i], a[i]);
        for (int p = 1; p < parameters.Count; p++)
            Assert.False(restricted.ContainsKey(parameters[p]), "a frozen parameter was handed a gradient.");
    }

    private static Dictionary<Tensor<float>, Tensor<float>> Backward(
        LayerBase<float> layer, Tensor<float> input, IReadOnlyList<Tensor<float>>? sources, out int entries,
        int applications = 1)
    {
        layer.ResetState();   // both walks start from the same recurrent state
        using var tape = new GradientTape<float>();
        var output = layer.Forward(input);
        for (int k = 1; k < applications; k++)
            output = layer.Forward(output);
        var zeros = new Tensor<float>(output.Shape.ToArray());
        for (int k = 0; k < IdentityPadding; k++)
            output = AiDotNetEngine.Current.TensorAdd(output, zeros);   // x + 0 is exact: gradients are unchanged
        var axes = new int[output.Shape.Length];
        for (int i = 0; i < axes.Length; i++) axes[i] = i;
        var loss = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(output, output), axes, keepDims: false);
        entries = tape.EntryCount;
        return tape.ComputeGradients(loss, sources);
    }
}
