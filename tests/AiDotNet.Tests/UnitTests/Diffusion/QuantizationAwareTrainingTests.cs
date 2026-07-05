using System;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// G5 (#1624) quantization-aware training on the diffusion training path. QAT is opt-in and lossy — it
/// fake-quantizes the weights the forward uses (keeping full-precision shadows updated via a
/// straight-through estimator). The locally-verifiable contract is that turning it on does NOT break or
/// diverge training: a small DDPM still reduces (does not blow up) its noise-prediction error. (The
/// memory payoff manifests only at foundation scale and is a CI concern.)
/// </summary>
public class QuantizationAwareTrainingTests
{
    private static double Mse(Tensor<double> a, Tensor<double> b)
    {
        double s = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) { double d = a[i] - b[i]; s += d * d; }
        return s / Math.Max(1, n);
    }

    [Fact]
    public void QAT_SmallModelOffByDefault_And_Toggles()
    {
        // The tiny DDPM is far below the foundation-scale default threshold, so QAT is off by default.
        var model = new DDPMModel<double>(channels: 3, imageSize: 8, seed: 42);
        Assert.False(model.IsQuantizationAwareTrainingEnabled);
        model.EnableQuantizationAwareTraining();
        Assert.True(model.IsQuantizationAwareTrainingEnabled);
        model.DisableQuantizationAwareTraining();
        Assert.False(model.IsQuantizationAwareTrainingEnabled);
    }

    [Fact]
    public void QAT_IsOptIn_NeverAutoEngagesBySize_AndExplicitWins()
    {
        // G5/#1624: QAT is opt-in and OFF by default at EVERY model size — model size never auto-engages
        // it. The previous auto-engage-by-parameter-count (>= 500M) was removed: it fake-quantized the
        // full weight set every train step (~96s/iter of overhead at foundation scale) and, being lossy,
        // perturbed the vanilla-DDPM contract tests. Foundation models that target int8 deployment opt in
        // explicitly via EnableQuantizationAwareTraining().
        var model = new DDPMModel<double>(channels: 3, imageSize: 8, seed: 1);
        Assert.False(model.IsQuantizationAwareTrainingEnabled);

        // An explicit enable wins over the opt-in default.
        model.EnableQuantizationAwareTraining();
        Assert.True(model.IsQuantizationAwareTrainingEnabled);

        // An explicit disable wins in the other direction.
        model.DisableQuantizationAwareTraining();
        Assert.False(model.IsQuantizationAwareTrainingEnabled);
    }

    [Fact]
    public void QAT_Training_DoesNotDiverge_AndStaysCompetitiveWithFullPrecision()
    {
        // The noise-prediction error at a FIXED probe timestep, measured while training on RANDOMLY
        // sampled timesteps, is a noisy single-point sample — its step-to-step trajectory oscillates with
        // an amplitude comparable to the value itself (a snapshot after N steps is luck, not a convergence
        // signal), and the forward pass is not bit-deterministic across constructions. So the robust,
        // documented contract (QAT "does not break or diverge training" and "still reduces … its error")
        // is checked three ways over a training run, against a full-precision control on the identical
        // setup: (a) QAT never yields a non-finite error (no divergence/explosion), (b) QAT is still ABLE
        // to reduce the error below its untrained value (best-achieved < initial), and (c) QAT stays
        // competitive with full precision — its best is within a small factor of the full-precision best
        // (QAT is lossy via the straight-through estimator, Ho et al. 2020 Alg. 1, but must not materially
        // wreck training).
        const int steps = 40;
        var (fpBest, _) = TrainAndTrackBestProbeError(enableQat: false, steps);
        var (qatBest, qatErr0) = TrainAndTrackBestProbeError(enableQat: true, steps);

        Assert.False(double.IsNaN(qatBest) || double.IsInfinity(qatBest),
            "QAT training produced a non-finite noise-prediction error (divergence/explosion).");
        Assert.True(qatBest < qatErr0,
            $"QAT training never reduced the noise-prediction error below its untrained value: " +
            $"err0={qatErr0:F6}, best={qatBest:F6}.");
        Assert.True(qatBest <= fpBest * 3.0,
            $"QAT training was not competitive with full precision: fpBest={fpBest:F6}, qatBest={qatBest:F6}.");
    }

    /// <summary>
    /// Trains a fresh tiny DDPM (optionally with QAT) for <paramref name="steps"/> steps and returns the
    /// BEST (minimum) noise-prediction error achieved at a fixed probe timestep, plus the initial
    /// (untrained) probe error. Tracking the best over the run — rather than the final step — is robust to
    /// the fixed-timestep probe's step-to-step oscillation under random-timestep training.
    /// </summary>
    private static (double best, double err0) TrainAndTrackBestProbeError(bool enableQat, int steps)
    {
        // A tiny UNet (baseChannels 16, no attention) exercises the identical QAT code path
        // (fake-quantize the forward weights + straight-through shadow updates) but trains in
        // milliseconds/step, so the 40-step FP + 40-step QAT run stays far under the CI per-test
        // budget. The default DDPM UNet is Stable-Diffusion-scale (baseChannels 320) — ~seconds/step,
        // which would overflow the budget for a multi-step convergence probe.
        var unet = new UNetNoisePredictor<double>(
            inputChannels: 3, outputChannels: 3, baseChannels: 16, channelMultipliers: new[] { 1, 2 },
            numResBlocks: 1, attentionResolutions: Array.Empty<int>(), contextDim: 16, numHeads: 1,
            inputHeight: 8, seed: 42);
        var model = new DDPMModel<double>(channels: 3, imageSize: 8, unet: unet, seed: 42);
        var rng = new Random(123);
        var shape = new[] { 1, 3, 8, 8 };

        var x0 = new Tensor<double>(shape);
        for (int i = 0; i < x0.Length; i++) x0[i] = rng.NextDouble() * 2.0 - 1.0;

        int probeT = Math.Max(1, model.Scheduler.TrainTimesteps / 2);
        var probeNoiseVec = new Vector<double>(x0.Length);
        for (int i = 0; i < probeNoiseVec.Length; i++) probeNoiseVec[i] = rng.NextDouble() * 2.0 - 1.0;
        var noisyProbe = new Tensor<double>(shape, model.Scheduler.AddNoise(x0.ToVector(), probeNoiseVec, probeT));
        var probeNoise = new Tensor<double>(shape, probeNoiseVec);

        double err0 = Mse(model.PredictNoise(noisyProbe, probeT), probeNoise);
        if (enableQat) model.EnableQuantizationAwareTraining();

        double best = err0;
        for (int i = 0; i < steps; i++)
        {
            model.Train(x0, x0);
            // The standalone probe runs on the full-precision (shadow) weights the optimizer updated.
            double e = Mse(model.PredictNoise(noisyProbe, probeT), probeNoise);
            if (e < best) best = e;
        }
        return (best, err0);
    }
}
