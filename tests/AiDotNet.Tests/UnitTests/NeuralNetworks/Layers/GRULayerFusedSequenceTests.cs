using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// GRULayer with the default activations runs the whole sequence as AiDotNet.Tensors' fused GruSequenceForward
/// (ooples/AiDotNet.Tensors#1568) in its reset-gate-before form. These tests hold it to the per-step loop it replaces:
/// a layer whose activations are subclasses of the same sigmoid and tanh keeps the loop, so with identical weights the
/// two must agree on the output and on the gradients of the input and every gate's weights.
/// </summary>
public class GRULayerFusedSequenceTests
{
    private sealed class LoopSigmoid<T> : SigmoidActivation<T> { }
    private sealed class LoopTanh<T> : TanhActivation<T> { }

    private const int Batch = 3, Steps = 5, Inputs = 4, Hidden = 6;

    private static Tensor<double> Input(int seed)
    {
        var rng = new Random(seed);
        var data = new double[Batch * Steps * Inputs];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 2 - 1;
        return new Tensor<double>(data, new[] { Batch, Steps, Inputs });
    }

    private static (GRULayer<double> Fused, GRULayer<double> Loop) Pair(bool returnSequences)
    {
        var fused = new GRULayer<double>(Hidden, returnSequences);
        var loop = new GRULayer<double>(Hidden, returnSequences, new LoopTanh<double>(), new LoopSigmoid<double>());
        fused.SetTrainingMode(false);
        loop.SetTrainingMode(false);
        var warm = Input(0);
        fused.Forward(warm);          // materializes the lazily shaped weights
        loop.Forward(warm);
        loop.SetParameters(fused.GetParameters());
        return (fused, loop);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Forward_MatchesThePerStepLoop(bool returnSequences)
    {
        var (fused, loop) = Pair(returnSequences);
        var input = Input(1);
        var expected = loop.Forward(input).ToArray();
        var actual = fused.Forward(input).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12, $"output {i}: loop {expected[i]}, fused {actual[i]}.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gradients_MatchThePerStepLoop(bool returnSequences)
    {
        var (fused, loop) = Pair(returnSequences);
        Dictionary<string, double[]> Grads(GRULayer<double> layer, out int tapeEntries)
        {
            var input = Input(2);
            var sources = new List<Tensor<double>> { input };
            sources.AddRange(layer.GetTrainableParameters());
            using var tape = new GradientTape<double>();
            var y = layer.Forward(input);
            var weights = new double[y.Length];
            for (int i = 0; i < weights.Length; i++) weights[i] = Math.Sin(i + 1);
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            var loss = engine.ReduceSum(engine.TensorMultiply(y, new Tensor<double>(weights, y.Shape.ToArray())), null);
            tapeEntries = tape.EntryCount;
            var g = tape.ComputeGradients(loss, sources.ToArray());
            var result = new Dictionary<string, double[]>();
            for (int s = 0; s < sources.Count; s++)
                result[$"source{s}"] = g.TryGetValue(sources[s], out var grad) ? grad.ToArray() : Array.Empty<double>();
            return result;
        }

        var expected = Grads(loop, out int loopEntries);
        var actual = Grads(fused, out int fusedEntries);
        // The fused route must actually have run: the loop records a dozen ops per step, the fused op one node (plus
        // the weight stacking and the loss). Equal values from two loops would otherwise pass.
        Assert.True(fusedEntries * 4 < loopEntries, $"fused route not taken: {fusedEntries} tape entries against the loop's {loopEntries}.");
        foreach (var key in expected.Keys)
        {
            Assert.True(expected[key].Length > 0, $"{key}: the loop produced no gradient.");
            Assert.Equal(expected[key].Length, actual[key].Length);
            for (int i = 0; i < expected[key].Length; i++)
                Assert.True(Math.Abs(expected[key][i] - actual[key][i]) <= 1e-10 + 1e-8 * Math.Abs(expected[key][i]),
                    $"{key}[{i}]: loop {expected[key][i]}, fused {actual[key][i]}.");
        }
    }
}
