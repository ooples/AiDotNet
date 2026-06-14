using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
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
    public void ComputeTapeLoss_LastAxisOneHotTarget_GradientIsSoftmaxMinusTarget()
    {
        var loss = new CrossEntropyWithLogitsLoss<double>();
        var predicted = new Tensor<double>(new[] { 1, 3 });
        predicted[0, 0] = 0.0;
        predicted[0, 1] = 0.0;
        predicted[0, 2] = 0.0;

        var oneHot = new Tensor<double>(new[] { 1, 3 });
        oneHot[0, 0] = 1.0;

        using var tape = new GradientTape<double>();
        var lossTensor = loss.ComputeTapeLoss(predicted, oneHot);
        var gradients = tape.ComputeGradients(lossTensor, new[] { predicted });

        Assert.True(gradients.TryGetValue(predicted, out var grad), "No gradient returned for logits.");
        Assert.True(System.Math.Abs(grad[0, 0] - (-2.0 / 3.0)) < 1e-10,
            $"Target class gradient should be softmax-target = -2/3; actual={grad[0, 0]}");
        Assert.True(System.Math.Abs(grad[0, 1] - (1.0 / 3.0)) < 1e-10,
            $"Non-target class gradient should be softmax-target = 1/3; actual={grad[0, 1]}");
        Assert.True(System.Math.Abs(grad[0, 2] - (1.0 / 3.0)) < 1e-10,
            $"Non-target class gradient should be softmax-target = 1/3; actual={grad[0, 2]}");
    }

    [Fact]
    public void ComputeTapeLoss_NchwClassIndexTarget_UsesChannelAxis()
    {
        var loss = new CrossEntropyWithLogitsLoss<double>();
        var predicted = new Tensor<double>(new[] { 1, 3, 2, 2 });

        // Four pixels, three class logits per pixel in standard PyTorch
        // segmentation layout [B, C, H, W].
        SetPixel(predicted, 0, 0, 2.0, 0.0, -1.0);
        SetPixel(predicted, 0, 1, -1.0, 3.0, 0.0);
        SetPixel(predicted, 1, 0, 0.0, -2.0, 4.0);
        SetPixel(predicted, 1, 1, 1.0, 0.5, -0.5);

        var classIdx = new Tensor<double>(new[] { 1, 2, 2 });
        classIdx[0, 0, 0] = 0;
        classIdx[0, 0, 1] = 1;
        classIdx[0, 1, 0] = 2;
        classIdx[0, 1, 1] = 0;

        var result = loss.ComputeTapeLoss(predicted, classIdx);

        double expected =
            PixelCrossEntropy(2.0, 0.0, -1.0, 0) +
            PixelCrossEntropy(-1.0, 3.0, 0.0, 1) +
            PixelCrossEntropy(0.0, -2.0, 4.0, 2) +
            PixelCrossEntropy(1.0, 0.5, -0.5, 0);
        expected /= 4.0;

        Assert.True(
            System.Math.Abs(result.Data.Span[0] - expected) < 1e-10,
            $"NCHW CE mismatch: actual={result.Data.Span[0]}, expected={expected}");
    }

    [Fact]
    public void ComputeTapeLoss_NchwOneHotTarget_MatchesClassIndexTarget()
    {
        var loss = new CrossEntropyWithLogitsLoss<double>();
        var predicted = new Tensor<double>(new[] { 1, 3, 2, 2 });
        SetPixel(predicted, 0, 0, 2.0, 0.0, -1.0);
        SetPixel(predicted, 0, 1, -1.0, 3.0, 0.0);
        SetPixel(predicted, 1, 0, 0.0, -2.0, 4.0);
        SetPixel(predicted, 1, 1, 1.0, 0.5, -0.5);

        var classIdx = new Tensor<double>(new[] { 1, 2, 2 });
        classIdx[0, 0, 0] = 0;
        classIdx[0, 0, 1] = 1;
        classIdx[0, 1, 0] = 2;
        classIdx[0, 1, 1] = 0;

        var oneHot = new Tensor<double>(new[] { 1, 3, 2, 2 });
        oneHot[0, 0, 0, 0] = 1.0;
        oneHot[0, 1, 0, 1] = 1.0;
        oneHot[0, 2, 1, 0] = 1.0;
        oneHot[0, 0, 1, 1] = 1.0;

        var classIdxLoss = loss.ComputeTapeLoss(predicted, classIdx);
        var oneHotLoss = loss.ComputeTapeLoss(predicted, oneHot);

        Assert.True(
            System.Math.Abs(classIdxLoss.Data.Span[0] - oneHotLoss.Data.Span[0]) < 1e-10,
            $"NCHW one-hot target should match class-index target: oneHot={oneHotLoss.Data.Span[0]}, classIdx={classIdxLoss.Data.Span[0]}");
    }

    [Fact]
    public void ComputeTapeLoss_NchwClassIndexTarget_GradientUsesChannelAxis()
    {
        var loss = new CrossEntropyWithLogitsLoss<double>();
        var predicted = new Tensor<double>(new[] { 1, 3, 1, 1 });
        predicted[0, 0, 0, 0] = 0.0;
        predicted[0, 1, 0, 0] = 0.0;
        predicted[0, 2, 0, 0] = 0.0;

        var classIdx = new Tensor<double>(new[] { 1, 1, 1 });
        classIdx[0, 0, 0] = 0;

        using var tape = new GradientTape<double>();
        var lossTensor = loss.ComputeTapeLoss(predicted, classIdx);
        var gradients = tape.ComputeGradients(lossTensor, new[] { predicted });

        Assert.True(gradients.TryGetValue(predicted, out var grad), "No gradient returned for logits.");
        Assert.True(System.Math.Abs(grad[0, 0, 0, 0] - (-2.0 / 3.0)) < 1e-10,
            $"Target class gradient should be softmax-target = -2/3; actual={grad[0, 0, 0, 0]}");
        Assert.True(System.Math.Abs(grad[0, 1, 0, 0] - (1.0 / 3.0)) < 1e-10,
            $"Non-target class gradient should be softmax-target = 1/3; actual={grad[0, 1, 0, 0]}");
        Assert.True(System.Math.Abs(grad[0, 2, 0, 0] - (1.0 / 3.0)) < 1e-10,
            $"Non-target class gradient should be softmax-target = 1/3; actual={grad[0, 2, 0, 0]}");
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

    private static void SetPixel(Tensor<double> tensor, int row, int col, double c0, double c1, double c2)
    {
        tensor[0, 0, row, col] = c0;
        tensor[0, 1, row, col] = c1;
        tensor[0, 2, row, col] = c2;
    }

    private static double PixelCrossEntropy(double c0, double c1, double c2, int targetClass)
    {
        double max = System.Math.Max(c0, System.Math.Max(c1, c2));
        double logSumExp = max + System.Math.Log(
            System.Math.Exp(c0 - max) +
            System.Math.Exp(c1 - max) +
            System.Math.Exp(c2 - max));
        double targetLogit = targetClass switch
        {
            0 => c0,
            1 => c1,
            _ => c2
        };
        return -targetLogit + logSumExp;
    }
}
