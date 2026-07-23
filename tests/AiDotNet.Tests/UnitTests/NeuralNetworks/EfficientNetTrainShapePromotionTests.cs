using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression coverage for <see cref="EfficientNetNetwork{T}.Train"/>'s
/// shape-promotion behavior — the override calls
/// <c>EnsureBatchForCnnTraining</c> to promote unbatched single-sample
/// rank-3 <c>[C, H, W]</c> input (and the matching unbatched target) to
/// the canonical rank-4 <c>[B, C, H, W]</c> shape Tan &amp; Le 2019 §3
/// specifies. Without this promotion the stem Conv2D treats the channel
/// axis as batch and emits <c>[1280, NumClasses]</c>, breaking the loss
/// layer's <c>EnsureTargetMatchesPredicted</c> check.
/// </summary>
public class EfficientNetTrainShapePromotionTests
{
    [Fact]
    public void Train_RankThreeInputAndRankOneTarget_PromotesAndDoesNotThrow()
    {
        // Wrap the body in a TensorArena scope so the multi-MB
        // forward/backward intermediates allocated during Train don't
        // leak into the managed heap and compound across the shard —
        // matches the test convention used everywhere else in this repo
        // (NeuralNetworkModelTestBase invariants, etc.).
        using var _arena = TensorArena.Create();

        // EfficientNet-B0 default ctor (paper-faithful Tan & Le 2019):
        // NumClasses = 1000, InputType = ThreeDimensional with
        // InputDepth = 3 (RGB), InputHeight = InputWidth = 224.
        // Pass an unbatched rank-3 [C=3, H=64, W=64] input + rank-1
        // [NumClasses] target — smaller H/W to keep the test fast; lazy
        // shape inference accepts arbitrary spatial dims. The Train
        // override must promote both to [1, 3, 64, 64] / [1, NumClasses]
        // internally so the forward path and loss target ranks line up.
        using var net = new EfficientNetNetwork<double>();

        var input = new Tensor<double>(new[] { 3, 64, 64 });
        // Use the shared seeded-random helper so this test ages alongside
        // every other deterministic test in the repo (rather than baking
        // the seeded-Random factory inline twice).
        var rng = AiDotNet.Tests.ModelFamilyTests.Base.ModelTestHelpers.CreateSeededRandom(42);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        // Rank-1 one-hot target ([NumClasses])
        var target = new Tensor<double>(new[] { 1000 });
        target[123] = 1.0;

        // The exception we'd hit pre-fix was:
        //   "Target shape dimension 0 (1000) does not match predicted shape
        //   dimension 0 (1280). Target shape: [1000], Predicted shape:
        //   [1280, 1000]."
        // i.e., the model emits [1280, 1000] instead of [1, 1000] because
        // the channel axis got reinterpreted as batch.

        // Snapshot parameters before training so we can verify training
        // ACTUALLY executed (not silently no-op'd via shape mismatch).
        var paramsBefore = net.GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++) snapshot[i] = paramsBefore[i];

        var ex = Record.Exception(() => net.Train(input, target));

        Assert.Null(ex);

        // Stronger postcondition: at least one parameter must have
        // changed. This catches the failure mode where Train() throws
        // nothing but quietly skips the actual gradient update due to
        // shape mismatches the loss layer silently absorbs.
        var paramsAfter = net.GetParameters();
        bool anyChanged = false;
        for (int i = 0; i < snapshot.Length && i < paramsAfter.Length; i++)
        {
            if (System.Math.Abs(snapshot[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged,
            "Train completed without exception but no parameter changed — " +
            "shape promotion may have succeeded only at the input layer with " +
            "the loss/gradient path silently no-op'ing.");
    }

    [Fact]
    public void Train_RankThreeInputAndRankTwoTarget_PromotesAndDoesNotThrow()
    {
        // Same TensorArena guard as the rank-1-target test above — keeps
        // the per-fact intermediate allocations off the managed heap.
        using var _arena = TensorArena.Create();

        // The other common shape-pair: unbatched rank-3 RGB image
        // ([3, H, W]) + already-rank-2 pre-batched target ([1, NumClasses]).
        // EnsureBatchForCnnTraining only promotes the target when its
        // rank is below promotedInput.Rank − 2, so the pre-batched
        // target must pass through untouched (otherwise it would become
        // [1, 1, NumClasses] and break the loss layer's shape match).
        using var net = new EfficientNetNetwork<double>();

        var input = new Tensor<double>(new[] { 3, 64, 64 });
        var rng = AiDotNet.Tests.ModelFamilyTests.Base.ModelTestHelpers.CreateSeededRandom(7);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var target = new Tensor<double>(new[] { 1, 1000 });
        target[0, 42] = 1.0;

        // Snapshot output for the same input AFTER training to verify the
        // pre-batched target path didn't get double-promoted into [1,1,1000]
        // (which would silently feed a wrong-shape target to the loss
        // layer and produce identical-to-pre-train outputs).
        var paramsBefore = net.GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++) snapshot[i] = paramsBefore[i];

        var ex = Record.Exception(() => net.Train(input, target));

        Assert.Null(ex);

        // Stronger postcondition: parameters must have actually been
        // updated. If the rank-2 target was double-promoted to rank-3
        // [1,1,1000], the loss layer's reshape contract may silently
        // align it back without a real gradient.
        var paramsAfter = net.GetParameters();
        bool anyChanged = false;
        for (int i = 0; i < snapshot.Length && i < paramsAfter.Length; i++)
        {
            if (System.Math.Abs(snapshot[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged,
            "Train completed without exception but no parameter changed — " +
            "pre-batched rank-2 target may have been double-promoted to " +
            "[1,1,NumClasses] and silently no-op'd the gradient.");
    }
}
