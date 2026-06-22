using System;
using AiDotNet.Diffusion;
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
    public void QAT_AutoEngages_ForFoundationScale_AndExplicitOverridesWin()
    {
        var prev = DiffusionModelBase<double>.QatThresholdOverride;
        try
        {
            // Drop the threshold below the tiny model's size → QAT auto-engages by default (G5 default-ON
            // for foundation-scale models, which a real 500M+ DiT exceeds).
            DiffusionModelBase<double>.QatThresholdOverride = 0;
            var auto = new DDPMModel<double>(channels: 3, imageSize: 8, seed: 1);
            Assert.True(auto.IsQuantizationAwareTrainingEnabled);

            // An explicit disable beats the threshold default.
            auto.DisableQuantizationAwareTraining();
            Assert.False(auto.IsQuantizationAwareTrainingEnabled);
        }
        finally
        {
            DiffusionModelBase<double>.QatThresholdOverride = prev;
        }
    }

    [Fact]
    public void QAT_Training_DoesNotDivergeAndConverges()
    {
        var model = new DDPMModel<double>(channels: 3, imageSize: 8, seed: 42);
        var rng = new Random(123);
        var shape = new[] { 1, 3, 8, 8 };

        var x0 = new Tensor<double>(shape);
        for (int i = 0; i < x0.Length; i++) x0[i] = rng.NextDouble() * 2.0 - 1.0;

        int probeT = Math.Max(1, model.Scheduler.TrainTimesteps / 2);
        var probeNoiseVec = new Vector<double>(x0.Length);
        for (int i = 0; i < probeNoiseVec.Length; i++) probeNoiseVec[i] = rng.NextDouble() * 2.0 - 1.0;
        var noisyProbe = new Tensor<double>(shape, model.Scheduler.AddNoise(x0.ToVector(), probeNoiseVec, probeT));
        var probeNoise = new Tensor<double>(shape, probeNoiseVec);

        double errBefore = Mse(model.PredictNoise(noisyProbe, probeT), probeNoise);

        model.EnableQuantizationAwareTraining();
        for (int i = 0; i < 8; i++) model.Train(x0, x0);

        // The standalone probe runs on the full-precision (shadow) weights the optimizer updated.
        double errAfter = Mse(model.PredictNoise(noisyProbe, probeT), probeNoise);

        Assert.False(double.IsNaN(errAfter) || double.IsInfinity(errAfter),
            "QAT training produced a non-finite noise-prediction error (divergence/explosion).");
        // QAT injects quantization noise, so allow a small slack vs. the strict full-precision bound, but
        // it must not blow the error up — training should still descend (Ho et al. 2020, Alg. 1) under STE.
        Assert.True(errAfter <= errBefore + 1e-3,
            $"QAT training increased the noise-prediction error: before={errBefore:F6}, after={errAfter:F6}.");
    }
}
