using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization;
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
// The compiled-training case switches TensorCodecOptions, which is process-wide state.
[Collection("NonParallelIntegration")]
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

    /// <summary>
    /// Compiled CPU training records the fused LSTM node into the plan (instead of the per-timestep loop). One training
    /// step through that plan must update every parameter the way the eager tape step does: same starting weights, same
    /// data, compilation on vs off.
    /// </summary>
    [Fact]
    public void CompiledTrainingStep_MatchesEagerStep()
    {
        int seq = 6, features = 4, outputs = 3;
        LSTMNeuralNetwork<float> Build() => new LSTMNeuralNetwork<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple,
                inputHeight: seq,
                inputWidth: features,
                outputSize: outputs),
            lossFunction: null, outputActivation: null);

        var input = Rand(new[] { seq, features }, 21, scale: 1f);
        var target = Rand(new[] { seq, outputs }, 22, scale: 1f);
        var compiled = Build();
        var eager = Build();
        compiled.Predict(input); // materialize lazy weights before copying them across
        eager.Predict(input);
        eager.SetParameters(compiled.GetParameters());
        var start = compiled.GetParameters().ToArray();

        var saved = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            compiled.Train(input, target);
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            eager.Train(input, target);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(saved);
        }

        var a = compiled.GetParameters().ToArray();
        var b = eager.GetParameters().ToArray();
        Assert.Equal(b.Length, a.Length);
        double dot = 0, na = 0, nb = 0, maxAbs = 0, maxMag = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double da = a[i] - start[i], db = b[i] - start[i];
            dot += da * db; na += da * da; nb += db * db;
            maxAbs = System.Math.Max(maxAbs, System.Math.Abs(a[i] - b[i]));
            maxMag = System.Math.Max(maxMag, System.Math.Abs(db));
        }
        Assert.True(nb > 0, "the eager step changed no parameter");
        double cos = dot / System.Math.Sqrt(na * nb);
        Assert.True(cos > 0.9999, $"compiled and eager updates point different ways: cos {cos:R}");
        Assert.True(maxAbs <= 1e-3 * maxMag + 1e-7, $"compiled vs eager parameters differ by {maxAbs:E3} (largest update {maxMag:E3})");
    }

    /// <summary>
    /// The fused forward records no final (h, c), so a later <c>ForwardFromState</c> must still return the real final
    /// state: it runs the per-step loop, whose last hidden state equals the output at the last timestep, and whose cell
    /// state is a computed value rather than zeros.
    /// </summary>
    [Fact]
    public void ForwardFromState_AfterFusedForward_ReturnsRealFinalState()
    {
        int batch = 3, seq = 5, features = 4, hidden = 6;
        var layer = new LSTMLayer<float>(hidden, (AiDotNet.Interfaces.IActivationFunction<float>?)null);
        layer.SetTrainingMode(true);
        var input = Rand(new[] { batch, seq, features }, 31, scale: 1f);

        var fusedOut = layer.Forward(input); // CPU float: the fused path
        var stepped = layer.ForwardFromState(input, null, null, out var finalHidden, out var finalCell);

        Assert.Equal(new[] { batch, hidden }, finalHidden.Shape.ToArray());
        Assert.Equal(new[] { batch, hidden }, finalCell.Shape.ToArray());
        var h = finalHidden.AsSpan();
        var f = fusedOut.AsSpan();
        var s = stepped.AsSpan();
        for (int b = 0; b < batch; b++)
            for (int j = 0; j < hidden; j++)
            {
                int last = (b * seq + seq - 1) * hidden + j;
                Assert.True(System.Math.Abs(h[b * hidden + j] - s[last]) < 1e-6f, $"final h[{b},{j}] is not the last step's output");
                Assert.True(System.Math.Abs(f[last] - s[last]) < 1e-4f, $"fused and per-step outputs differ at [{b},{seq - 1},{j}]");
            }
        bool anyNonZeroCell = false;
        foreach (var v in finalCell.AsSpan()) if (v != 0f) { anyNonZeroCell = true; break; }
        Assert.True(anyNonZeroCell, "final cell state is all zeros: a fabricated state, not the recurrence's");
    }
}
