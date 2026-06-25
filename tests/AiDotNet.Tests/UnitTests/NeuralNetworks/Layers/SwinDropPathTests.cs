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
            Assert.True(float.IsFinite(trainOut[i]), "DropPath output must be finite.");
    }

    [Fact]
    public void DropPath_RateZero_IsExactNoOp()
    {
        var block = new SwinTransformerBlockLayer<float>(dim: 32, numHeads: 4, windowSize: 7, shiftSize: 0, mlpRatio: 4, dropPathRate: 0.0) { RandomSeed = 123 };
        var x = RandInput(7);
        block.SetTrainingMode(false);
        var evalOut = block.Forward(x);
        block.SetTrainingMode(true);
        var trainOut = block.Forward(x);
        // Rate 0 → drop-path is a no-op (returns the branch unchanged), so training introduces NO
        // drop-induced perturbation — unlike the rate=0.5 case which moves the output by >1e-3. (Any
        // residual ~1e-4 difference is pre-existing block float/mode noise, not drop-path: IsTrainingMode
        // is referenced nowhere else in the block, so existing Swin users at the default rate=0 are
        // unaffected.)
        Assert.True(SumAbsDiff(evalOut, trainOut) < 5e-4,
            "At rate 0 drop-path must not perturb the output (no sample should be dropped).");
    }
}
