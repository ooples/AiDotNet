using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

/// <summary>
/// Verifies <see cref="CrossEntropyWithLogitsLoss{T}.ComputeTapeLoss"/> accepts
/// BOTH PyTorch-style target formats:
/// <list type="bullet">
/// <item>soft / one-hot targets: <c>target.Shape == predicted.Shape</c></item>
/// <item>class-index targets:    <c>target.Shape == predicted.Shape[:-1]</c></item>
/// </list>
/// The class-index branch was added in this PR after PR #1404's blanket
/// <c>CrossEntropyLoss → CrossEntropyWithLogitsLoss</c> swap caused TinyBERTNER
/// to throw <c>ArgumentException: Tensors with shapes [N] and [N, C] cannot be
/// broadcast</c> — the model passes class-index targets, the loss previously
/// only handled soft / one-hot.
/// </summary>
public class CrossEntropyWithLogitsLossTapeTargetTests
{
    [Fact]
    public void ComputeTapeLoss_OneHotTarget_MatchesClassIndexTarget()
    {
        var loss = new CrossEntropyWithLogitsLoss<double>();

        // 4 samples × 3 classes; class index = argmax of these one-hot rows
        var predicted = new Tensor<double>(new[] { 4, 3 });
        predicted[0, 0] = 0.1; predicted[0, 1] = 0.2; predicted[0, 2] = 0.7;
        predicted[1, 0] = 0.9; predicted[1, 1] = 0.05; predicted[1, 2] = 0.05;
        predicted[2, 0] = 0.3; predicted[2, 1] = 0.5; predicted[2, 2] = 0.2;
        predicted[3, 0] = 0.2; predicted[3, 1] = 0.3; predicted[3, 2] = 0.5;

        // Form (a): one-hot targets [4, 3]
        var oneHot = new Tensor<double>(new[] { 4, 3 });
        oneHot[0, 2] = 1.0;
        oneHot[1, 0] = 1.0;
        oneHot[2, 1] = 1.0;
        oneHot[3, 2] = 1.0;

        // Form (b): class-index targets [4]
        var classIdx = new Tensor<double>(new[] { 4 });
        classIdx[0] = 2.0;
        classIdx[1] = 0.0;
        classIdx[2] = 1.0;
        classIdx[3] = 2.0;

        var lossOneHot = loss.ComputeTapeLoss(predicted, oneHot);
        var lossClassIdx = loss.ComputeTapeLoss(predicted, classIdx);

        // Both must produce the same scalar loss — class indices are just a
        // compact representation of the same one-hot targets.
        Assert.Equal(lossOneHot.Length, lossClassIdx.Length);
        for (int i = 0; i < lossOneHot.Length; i++)
        {
            Assert.True(
                System.Math.Abs(lossOneHot.Data.Span[i] - lossClassIdx.Data.Span[i]) < 1e-10,
                $"Loss differs at [{i}]: one-hot={lossOneHot.Data.Span[i]}, " +
                $"class-idx={lossClassIdx.Data.Span[i]}");
        }
    }

    [Fact]
    public void ComputeTapeLoss_ClassIndexTarget_DoesNotThrowOnRankMismatch()
    {
        // Regression for TinyBERTNER: predicted [256, 9], target [256].
        // Before the class-index branch this threw
        //   ArgumentException: Tensors with shapes [256] and [256, 9] cannot
        //   be broadcast (dimension 1 sizes 256 vs 9).
        var loss = new CrossEntropyWithLogitsLoss<double>();
        var predicted = new Tensor<double>(new[] { 256, 9 });
        var classIdx = new Tensor<double>(new[] { 256 });
        for (int i = 0; i < 256; i++) classIdx[i] = i % 9; // valid indices

        var result = loss.ComputeTapeLoss(predicted, classIdx);
        Assert.NotNull(result);
        // Loss should be finite (predicted is all zeros → uniform logits →
        // log_softmax = -log(9) per slot; -mean(target_onehot · log_softmax)
        // = log(9) ≈ 2.197).
        Assert.False(double.IsNaN(result.Data.Span[0]));
        Assert.False(double.IsInfinity(result.Data.Span[0]));
    }

    [Fact]
    public void ComputeTapeLoss_OutOfRangeClassIndex_IsTreatedAsIgnore()
    {
        // PyTorch convention: out-of-range indices (e.g. -1 sentinel) zero
        // their one-hot row, contributing zero gradient. The loss should
        // still be finite — the rows are effectively skipped.
        var loss = new CrossEntropyWithLogitsLoss<double>();
        var predicted = new Tensor<double>(new[] { 4, 3 });
        predicted[0, 0] = 0.5; predicted[0, 1] = 0.2; predicted[0, 2] = 0.3;
        predicted[1, 0] = 0.1; predicted[1, 1] = 0.6; predicted[1, 2] = 0.3;
        predicted[2, 0] = 0.4; predicted[2, 1] = 0.1; predicted[2, 2] = 0.5;
        predicted[3, 0] = 0.3; predicted[3, 1] = 0.3; predicted[3, 2] = 0.4;

        var classIdx = new Tensor<double>(new[] { 4 });
        classIdx[0] = 0;
        classIdx[1] = -1; // ignore sentinel
        classIdx[2] = 2;
        classIdx[3] = 99; // out-of-range — also treated as ignore

        var result = loss.ComputeTapeLoss(predicted, classIdx);
        Assert.False(double.IsNaN(result.Data.Span[0]));
        Assert.False(double.IsInfinity(result.Data.Span[0]));
    }
}
