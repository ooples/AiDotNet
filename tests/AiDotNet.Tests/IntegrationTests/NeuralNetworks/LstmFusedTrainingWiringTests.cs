using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Validates the fused LSTM training wiring: LSTMLayer.Forward (training) → tape-connected
/// stacked weights via Engine.Concat → CpuEngine.LstmSequenceForward fused BPTT node, instead
/// of the per-timestep loop's hundreds of nodes (ooples/AiDotNet#1566). The fused path needs
/// the tape-aware LstmSequenceForward from AiDotNet.Tensors#587 (released in 0.94.0); the layer
/// still falls back to the per-step loop if a future/older package doesn't record (GradFn null),
/// so training is never broken either way.
/// </summary>
public class LstmFusedTrainingWiringTests
{
    private static Tensor<float> Rand(int[] shape, int seed, float scale = 0.4f)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static float Mse(Tensor<float> pred, Tensor<float> target)
    {
        var p = pred.AsSpan();
        var g = target.AsSpan();
        int n = System.Math.Min(p.Length, g.Length);
        float sum = 0f;
        for (int i = 0; i < n; i++) { float d = p[i] - g[i]; sum += d * d; }
        return sum / System.Math.Max(1, n);
    }

    /// <summary>
    /// Confirms the consumed AiDotNet.Tensors package exposes the tape-aware LstmSequenceForward
    /// (the #587 fused training path) the wiring depends on: under a GradientTape the float
    /// primitive records a node (output.GradFn != null) and produces gradients for input + both
    /// weight matrices. If this fails, the package predates #587 and the layer would fall back to
    /// the per-step loop (correct, but not the fused fast path this PR delivers).
    /// </summary>
    [Fact]
    public void EngineLstmSequenceForward_IsTapeAware_AndProducesGradients()
    {
        var engine = new CpuEngine();
        int batch = 2, seq = 3, inF = 4, hidden = 5, gateRows = 4 * hidden;

        var input = Rand(new[] { batch, seq, inF }, 1);
        var wIh = Rand(new[] { gateRows, inF }, 2);
        var wHh = Rand(new[] { gateRows, hidden }, 3);

        using var tape = new GradientTape<float>();
        var output = engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null, returnSequences: true);

        Assert.NotNull(output.GradFn); // tape-connected ⇒ fused training path engaged, not the old throw/inference

        var loss = engine.ReduceSum(engine.TensorMultiply(output, output), null);
        var grads = tape.ComputeGradients(loss, new[] { input, wIh, wHh });

        Assert.True(grads.ContainsKey(wIh), "no gradient flowed to wIh");
        Assert.True(grads.ContainsKey(wHh), "no gradient flowed to wHh");
        Assert.True(grads.ContainsKey(input), "no gradient flowed to input");
        foreach (var g in grads[wIh].AsSpan().ToArray())
            Assert.True(!float.IsNaN(g) && !float.IsInfinity(g), "wIh gradient has NaN/Inf");
    }

    /// <summary>
    /// End-to-end: a correctly-wired training loop (fused path on 0.94.0) must drive the loss down.
    /// </summary>
    [Fact]
    public void LstmTraining_ReducesLoss()
    {
        int seq = 6, features = 4, outputs = 3;
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: seq,
            inputWidth: features,
            outputSize: outputs);

        var network = new LSTMNeuralNetwork<float>(architecture, lossFunction: null, outputActivation: null);

        var input = Rand(new[] { seq, features }, 11, scale: 1f);

        // Match the network's actual output shape (the simple LSTM net returns the full
        // per-timestep [seq, outputs] stack). Use a fixed random target to overfit.
        var probe = network.Predict(input);
        var target = new Tensor<float>(new[] { seq, outputs });
        var tg = target.AsWritableSpan();
        var trng = new System.Random(99);
        for (int i = 0; i < tg.Length; i++) tg[i] = (float)(trng.NextDouble() * 2 - 1);

        float lossBefore = Mse(probe, target);

        for (int step = 0; step < 60; step++)
            network.Train(input, target);

        float lossAfter = Mse(network.Predict(input), target);

        Assert.True(lossAfter < lossBefore,
            $"LSTM training did not reduce loss (wiring regression): {lossBefore:F5} -> {lossAfter:F5}");
    }
}
