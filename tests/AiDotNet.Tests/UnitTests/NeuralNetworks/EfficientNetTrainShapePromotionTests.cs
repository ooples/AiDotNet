using AiDotNet.NeuralNetworks;
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
        var rng = new Random(42);
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
        var ex = Record.Exception(() => net.Train(input, target));

        Assert.Null(ex);
    }

    [Fact]
    public void Train_RankThreeInputAndRankTwoTarget_PromotesAndDoesNotThrow()
    {
        // The other common shape-pair: unbatched rank-3 RGB image
        // ([3, H, W]) + already-rank-2 pre-batched target ([1, NumClasses]).
        // EnsureBatchForCnnTraining only promotes the target when its
        // rank is below promotedInput.Rank − 2, so the pre-batched
        // target must pass through untouched (otherwise it would become
        // [1, 1, NumClasses] and break the loss layer's shape match).
        using var net = new EfficientNetNetwork<double>();

        var input = new Tensor<double>(new[] { 3, 64, 64 });
        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var target = new Tensor<double>(new[] { 1, 1000 });
        target[0, 42] = 1.0;

        var ex = Record.Exception(() => net.Train(input, target));

        Assert.Null(ex);
    }
}
