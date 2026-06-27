using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

// Validates the paper-faithful stochastic-depth (drop-path) added to SwinTransformerBlockLayer
// (Liu et al. 2021): identity at inference, active per-sample dropping during training, and an
// exact no-op when the rate is 0.
public class SwinDropPathTests
{
    private static Tensor<float> RandInput(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(new[] { 4, 49, 32 }); // [batch, 7x7 window, dim]
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static double SumAbsDiff(Tensor<float> a, Tensor<float> b)
    {
        double s = 0; for (int i = 0; i < a.Length; i++) s += System.Math.Abs((double)a[i] - (double)b[i]); return s;
    }

    [Fact]
    public void DropPath_AtInference_IsDeterministicIdentity()
    {
        var block = new SwinTransformerBlockLayer<float>(dim: 32, numHeads: 4, windowSize: 7, shiftSize: 0, mlpRatio: 4, dropPathRate: 0.5) { RandomSeed = 123 };
        var x = RandInput(7);
        block.SetTrainingMode(false);
        var a = block.Forward(x);
        var b = block.Forward(x);
        // Eval mode performs no dropping → two passes are bit-identical (no RNG path taken).
        Assert.Equal(0.0, SumAbsDiff(a, b), 6);
    }

    [Fact]
    public void DropPath_DuringTraining_ChangesOutput()
    {
        var block = new SwinTransformerBlockLayer<float>(dim: 32, numHeads: 4, windowSize: 7, shiftSize: 0, mlpRatio: 4, dropPathRate: 0.5) { RandomSeed = 123 };
        var x = RandInput(7);
        block.SetTrainingMode(false);
        var evalOut = block.Forward(x);
        block.SetTrainingMode(true);
        var trainOut = block.Forward(x);
        // With rate 0.5 over batch=4, at least one sample's residual branch is dropped/scaled, so the
        // training output must differ from the clean eval output (and stay finite).
        Assert.True(SumAbsDiff(evalOut, trainOut) > 1e-3, "DropPath should perturb the training output.");
        for (int i = 0; i < trainOut.Length; i++)
            // net471 has no float.IsFinite — use the !NaN && !Infinity form (both exist there).
            Assert.True(!float.IsNaN(trainOut[i]) && !float.IsInfinity(trainOut[i]), "DropPath output must be finite.");
    }

    [Fact]
    public void DropPath_RateZero_IsExactNoOp()
    {
        var block = new SwinTransformerBlockLayer<float>(dim: 32, numHeads: 4, windowSize: 7, shiftSize: 0, mlpRatio: 4, dropPathRate: 0.0) { RandomSeed = 123 };
        // Force the unfused (train/eval-identical) forward so the ONLY thing that could differ between
        // eval and train is drop-path. Without this, DenseLayer's fused-activation eval optimization
        // (GELU in the MLP) reorders the rounding by ~1e-8/element — a deliberate inference speedup,
        // not drop-path. With it, eval and train are bit-identical, so rate-0 drop-path is provably a no-op.
        block.SetDeterministicForward(true);
        var x = RandInput(7);
        block.SetTrainingMode(false);
        var evalOut = block.Forward(x);
        block.SetTrainingMode(true);
        var trainOut = block.Forward(x);
        Assert.Equal(0.0, SumAbsDiff(evalOut, trainOut), 6);
    }
}
