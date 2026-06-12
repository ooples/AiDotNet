using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Validates the fused LSTM training wiring (LSTMLayer.Forward → tape-connected stacked
/// weights → CpuEngine.LstmSequenceForward fused BPTT node). The fused training path
/// requires AiDotNet.Tensors with tape-aware LstmSequenceForward
/// (ooples/AiDotNet.Tensors#587); on an older package the layer catches the throw and
/// falls back to the per-timestep loop. EITHER WAY training must reduce the loss — this
/// test guards that the wiring never breaks training, and validates the fused path once
/// the package ships.
/// </summary>
public class LstmFusedTrainingWiringTests
{
    private static Tensor<float> SeqInput(int seq, int features, int seed)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<float>(new[] { seq, features });
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
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

    [Fact]
    public void LstmTraining_ReducesLoss_FusedOrFallback()
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

        var input = SeqInput(seq, features, seed: 11);

        // Match the network's actual output shape (the simple LSTM net returns the full
        // per-timestep [seq, outputs] stack). Use a fixed random target to overfit.
        var probe = network.Predict(input);
        var target = new Tensor<float>(new[] { seq, outputs });
        var tg = target.AsWritableSpan();
        var trng = new System.Random(99);
        for (int i = 0; i < tg.Length; i++) tg[i] = (float)(trng.NextDouble() * 2 - 1);

        float lossBefore = Mse(probe, target);

        // Overfit a single fixed sample — a correctly-wired training loop must drive the
        // loss down regardless of which forward path (fused vs per-step) is taken.
        for (int step = 0; step < 60; step++)
            network.Train(input, target);

        float lossAfter = Mse(network.Predict(input), target);

        Assert.True(lossAfter < lossBefore,
            $"LSTM training did not reduce loss (wiring regression): {lossBefore:F5} -> {lossAfter:F5}");
    }
}
