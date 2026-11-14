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

    private static (GRULayer<double> Fused, GRULayer<double> Loop) Pair(bool returnSequences, bool resetAfter = true)
    {
        var fused = new GRULayer<double>(Hidden, returnSequences, resetAfter: resetAfter);
        var loop = new GRULayer<double>(Hidden, returnSequences, new LoopTanh<double>(), new LoopSigmoid<double>(),
            resetAfter: resetAfter);
        fused.SetTrainingMode(false);
        loop.SetTrainingMode(false);
        var warm = Input(0);
        fused.Forward(warm);          // materializes the lazily shaped weights
        loop.Forward(warm);
        loop.SetParameters(fused.GetParameters());
        return (fused, loop);
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, true)]
    [InlineData(false, false)]
    [InlineData(true, false)]
    public void Forward_MatchesThePerStepLoop(bool returnSequences, bool resetAfter)
    {
        var (fused, loop) = Pair(returnSequences, resetAfter);
        var input = Input(1);
        var expected = loop.Forward(input).ToArray();
        var actual = fused.Forward(input).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12, $"output {i}: loop {expected[i]}, fused {actual[i]}.");
    }

    [Fact]
    public void Inference_AfterAnInPlaceGateWrite_UsesTheNewWeights()
    {
        // The fused inference path packs the gate weights once and reuses the pack. A write through the element setter
        // changes only the tensor's version, not its identity, so a pack keyed on identity alone would keep serving
        // the old weights here and drift from the loop, which reads the gates directly.
        var (fused, loop) = Pair(returnSequences: true);
        var input = Input(3);
        fused.Forward(input);
        loop.Forward(input);

        var before = fused.GetParameters().ToArray();
        var gates = fused.GetTrainableParameters();
        Assert.NotEmpty(gates);
        foreach (var gate in gates) gate.SetFlat(0, gate.GetFlat(0) + 0.75);
        var after = fused.GetParameters();
        var afterValues = after.ToArray();
        // The writes reached the weights the layer reports, so a fused output that ignores them is the stale pack.
        int changed = 0;
        for (int i = 0; i < afterValues.Length; i++)
            if (afterValues[i] != before[i]) changed++;
        Assert.Equal(gates.Count, changed);
        loop.SetParameters(after);

        var expected = loop.Forward(input).ToArray();
        var actual = fused.Forward(input).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12, $"output {i}: loop {expected[i]}, fused {actual[i]}.");
    }

    [Fact]
    public void ResetPlacement_ChangesTheRecurrence_AndSurvivesAClone()
    {
        // Same weights, different reset-gate placement: the two forms are different functions, so a flag that was
        // dropped (by the layer, or by a clone that rebuilds it from its metadata) shows as equal outputs here.
        var (after, _) = Pair(returnSequences: true, resetAfter: true);
        var (before, _) = Pair(returnSequences: true, resetAfter: false);
        before.SetParameters(after.GetParameters());
        Assert.True(after.ResetAfter);
        Assert.False(before.ResetAfter);

        var input = Input(4);
        var afterOutput = after.Forward(input).ToArray();
        var beforeOutput = before.Forward(input).ToArray();
        double largest = 0;
        for (int i = 0; i < afterOutput.Length; i++)
            largest = Math.Max(largest, Math.Abs(afterOutput[i] - beforeOutput[i]));
        Assert.True(largest > 1e-6, $"reset-after and reset-before agree to {largest}: the flag has no effect.");

        var afterClone = (GRULayer<double>)after.Clone();
        var beforeClone = (GRULayer<double>)before.Clone();
        Assert.True(afterClone.ResetAfter);
        Assert.False(beforeClone.ResetAfter);
        beforeClone.SetTrainingMode(false);
        var cloneOutput = beforeClone.Forward(input).ToArray();
        for (int i = 0; i < beforeOutput.Length; i++)
            Assert.True(Math.Abs(beforeOutput[i] - cloneOutput[i]) <= 1e-12, $"output {i}: original {beforeOutput[i]}, clone {cloneOutput[i]}.");
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, true)]
    [InlineData(false, false)]
    [InlineData(true, false)]
    public void Gradients_MatchThePerStepLoop(bool returnSequences, bool resetAfter)
    {
        var (fused, loop) = Pair(returnSequences, resetAfter);
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
