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

    private static Dictionary<Tensor<float>, Tensor<float>> Backward(
        LayerBase<float> layer, Tensor<float> input, IReadOnlyList<Tensor<float>>? sources, out int entries)
    {
        layer.ResetState();   // both walks start from the same recurrent state
        using var tape = new GradientTape<float>();
        var output = layer.Forward(input);
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
