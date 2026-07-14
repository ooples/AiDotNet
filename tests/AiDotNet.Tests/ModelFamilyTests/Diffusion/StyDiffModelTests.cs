using System;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class StyDiffModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion. Use a 16x16 latent (not the paper's 64x64) so the U-Net's
    // self-attention runs over 256 tokens instead of 4096 — the multi-iteration Training loop then
    // finishes inside the 120s gate rather than timing out at the SD1.5-scale default.
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    // Build the U-Net + VAE at a REDUCED width instead of the SD1.5-scale default (baseChannels 320 x
    // [1,2,4,4]), which peaks ~49 GB and blows the gate. Shape-critical dims preserved (inputChannels =
    // LATENT_CHANNELS 4, contextDim 768) so the forward path is exercised identically; the test stays
    // exact, fast, and in the default PR gate.
    protected override IDiffusionModel<float> CreateModel()
        => new StyDiffModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2, 4 }, numResBlocks: 1,
                attentionResolutions: new[] { 1, 2 }, contextDim: 768, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);

    // TEMP DIAGNOSTIC (#1789 Step-Sync shard): StyDiff Clone_ShouldProduceIdenticalOutput fails ONLY
    // on the Linux CI runner with a ~1-ULP float divergence (Windows local passes solo). This
    // localizes the cause by comparing three things and reporting the numbers in the trx Message:
    //   1. original.Predict called twice   -> is the ORIGINAL non-deterministic across calls?
    //   2. original vs clone.Predict       -> does the clone compute a different path?
    //   3. the same two, with TensorCodecOptions.EnableCompilation = false -> does disabling the
    //      compiled-replay path (pre-packed-B / autotune memo, Tensors #782) make the divergence
    //      vanish? If OFF is clean and ON diverges, compiled replay is the cause.
    // Passes on Windows (all diffs exactly 0); fails with the breakdown on Linux. Temporary.
    [Fact]
    public void Diag_StyDiff_CloneDeterminism()
    {
        var sb = new StringBuilder();
        bool bad = false;

        RunProbe("compile=ON", sb, ref bad);

        var codec = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = codec.EnableCompilation;
        try
        {
            codec.EnableCompilation = false;
            RunProbe("compile=OFF", sb, ref bad);
        }
        finally { codec.EnableCompilation = saved; }

        if (bad)
            Assert.Fail("StyDiff clone-determinism diagnostic (non-zero divergence):\n" + sb.ToString());
    }

    private void RunProbe(string label, StringBuilder sb, ref bool bad)
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var o1 = model.Predict(input);
        var o2 = model.Predict(input);
        var clone = model.Clone();
        var c1 = clone.Predict(input);
        var c2 = clone.Predict(input);

        var origRepeat = MaxDiff(o1, o2);
        var origVsClone = MaxDiff(o1, c1);
        var cloneRepeat = MaxDiff(c1, c2);

        sb.AppendLine($"[{label}] original-repeat={origRepeat.diff:E3}@{origRepeat.idx}  " +
                      $"orig-vs-clone={origVsClone.diff:E3}@{origVsClone.idx}  " +
                      $"clone-repeat={cloneRepeat.diff:E3}@{cloneRepeat.idx}");
        if (origRepeat.diff > 0)
            sb.AppendLine($"    ORIGINAL non-deterministic across calls: [{origRepeat.idx}] o1={origRepeat.a:R} o2={origRepeat.b:R}");
        if (origVsClone.diff > 0)
            sb.AppendLine($"    ORIG vs CLONE differ: [{origVsClone.idx}] orig={origVsClone.a:R} clone={origVsClone.b:R}");
        if (cloneRepeat.diff > 0)
            sb.AppendLine($"    CLONE non-deterministic across calls: [{cloneRepeat.idx}] c1={cloneRepeat.a:R} c2={cloneRepeat.b:R}");

        if (origRepeat.diff > 0 || origVsClone.diff > 0 || cloneRepeat.diff > 0) bad = true;
    }

    private static (double diff, int idx, double a, double b) MaxDiff(Tensor<float> x, Tensor<float> y)
    {
        var sx = x.Data.Span; var sy = y.Data.Span;
        int n = Math.Min(sx.Length, sy.Length);
        double best = 0; int bi = -1; double ba = 0, bb = 0;
        for (int i = 0; i < n; i++)
        {
            double d = Math.Abs((double)sx[i] - (double)sy[i]);
            if (d > best) { best = d; bi = i; ba = sx[i]; bb = sy[i]; }
        }
        return (best, bi, ba, bb);
    }
}
