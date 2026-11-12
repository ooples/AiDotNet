using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

/// <summary>
/// CTC against torch.nn.functional.ctc_loss (<c>tools/reference-data/ctc_loss_reference.py</c>), with targets that
/// repeat a label: <see cref="CTCLoss{T}"/>'s loss and gradient, and the engine's tape-tracked <c>TensorCTCLoss</c>.
/// Gradients are compared with respect to the logits under the log-softmax.
/// </summary>
public class CtcLossReferenceTests
{
    private const int Time = 9, Batch = 3, Classes = 5;

    private static JObject Fixture()
    {
        const string fileName = "ctc_loss_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "UnitTests", "LossFunctions", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "UnitTests", "LossFunctions", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static void Close(double expected, double actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1e-9 * Math.Max(1, Math.Abs(expected)), $"{what}: torch {expected:R}, port {actual:R}");

    [Fact]
    public void CtcLoss_LossAndGradient_MatchTorch()
    {
        var f = Fixture();
        var logProbs = f["log_probs"]!.Select(v => (double)v).ToArray();                                   // [T, N, C]
        var targets = f["targets"]!.Select(t => t.Select(v => (int)v).ToArray()).ToArray();
        var lengths = f["input_lengths"]!.Select(v => (int)v).ToArray();
        var losses = f["losses"]!.Select(v => (double)v).ToArray();
        var gradient = f["gradient"]!.Select(v => (double)v).ToArray();
        var ctc = new CTCLoss<double>(Classes, blankIndex: 0);
        for (int n = 0; n < Batch; n++)
        {
            // One sequence, batch-first [1, T, C].
            var single = new Tensor<double>(new[] { 1, Time, Classes });
            for (int t = 0; t < Time; t++)
                for (int c = 0; c < Classes; c++) single[0, t, c] = logProbs[(t * Batch + n) * Classes + c];
            Close(losses[n], ctc.CalculateLoss(single, new[] { targets[n] }, new[] { lengths[n] }, new[] { targets[n].Length }), $"loss {n}");
            var g = ctc.CalculateGradient(single, new[] { targets[n] }, new[] { lengths[n] }, new[] { targets[n].Length });
            for (int t = 0; t < Time; t++)
            {
                // Through the log-softmax: ∂/∂z = g − softmax · Σ g.
                double sum = Enumerable.Range(0, Classes).Sum(c => g[0, t, c]);
                for (int c = 0; c < Classes; c++)
                {
                    double softmax = Math.Exp(logProbs[(t * Batch + n) * Classes + c]);
                    Close(gradient[(t * Batch + n) * Classes + c], g[0, t, c] - softmax * sum, $"gradient {n}, t {t}, class {c}");
                }
            }
        }
    }

    [Fact]
    public void EngineCtcLoss_LossAndGradient_MatchTorch()
    {
        var f = Fixture();
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<double>(new[] { Time, Batch, Classes }, new Vector<double>(f["logits"]!.Select(v => (double)v).ToArray()));
        var targets = f["targets"]!.Select(t => t.Select(v => (int)v).ToArray()).ToArray();
        var flat = new Tensor<int>(new[] { targets.Sum(t => t.Length) }, new Vector<int>(targets.SelectMany(t => t).ToArray()));
        var lengths = f["input_lengths"]!.Select(v => (int)v).ToArray();
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> losses;
        using (var tape = new GradientTape<double>())
        {
            losses = engine.TensorCTCLoss(engine.TensorLogSoftmax(logits, axis: 2), flat, lengths, targets.Select(t => t.Length).ToArray(), 0);
            grads = tape.ComputeGradients(engine.ReduceSum(losses, null, false), sources: new[] { logits });
        }
        var expected = f["losses"]!.Select(v => (double)v).ToArray();
        for (int n = 0; n < Batch; n++) Close(expected[n], losses[n], $"loss {n}");
        var gradient = f["gradient"]!.Select(v => (double)v).ToArray();
        for (int i = 0; i < gradient.Length; i++) Close(gradient[i], grads[logits][i], $"gradient {i}");
    }
}
