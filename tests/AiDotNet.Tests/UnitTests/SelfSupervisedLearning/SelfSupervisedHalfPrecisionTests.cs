using System;
using AiDotNet.SelfSupervisedLearning.Evaluation;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SelfSupervisedLearning;

/// <summary>
/// Half-precision references for self-supervised reductions where master counted or accumulated
/// outside T. Half sums of ones stop at 2048, so counts must be taken in int.
/// </summary>
public class SelfSupervisedHalfPrecisionTests
{
    [Fact]
    public void MaePerSampleLoss_Half_DividesByTheTrueMaskedPatchCount()
    {
        // One sample, 3000 masked one-dimensional patches; only patch 0 differs (by 2, loss 4).
        // Master: 4 / 3000. A Half count of the 3000 masked patches would stop at 2048.
        const int patches = 3000;
        var reconstructed = new Tensor<Half>(new[] { 1, patches, 1 });
        reconstructed[0, 0, 0] = (Half)2.0f;
        var original = new Tensor<Half>(new[] { 1, patches, 1 });
        var mask = new Tensor<Half>(new[] { 1, patches });
        for (int p = 0; p < patches; p++)
        {
            mask[0, p] = (Half)1.0f;
        }

        var loss = new MAEReconstructionLoss<Half>(normalize: true, perPatchNormalization: false);
        Half[] perSample = loss.ComputePerSampleLoss(reconstructed, original, mask);

        Assert.Single(perSample);
        Assert.Equal(4.0 / patches, (double)perSample[0], 5);
    }

    [Fact]
    public void DinoLoss_Half_ZeroStudentProbabilityStaysFinite()
    {
        // The student puts probability exp(-200) = 0 on the teacher's class. The 1e-8 guard of
        // log(p + 1e-8) rounds to 0 in Half, so the loss was -log(0) = +inf. With Half's smallest
        // normal (2^-14) as the guard the loss is -log(2^-14) = 14 ln 2.
        var student = new Tensor<Half>(new[] { 1, 2 });
        student[0, 0] = (Half)0.0f;
        student[0, 1] = (Half)(-20.0f);
        var teacher = new Tensor<Half>(new[] { 1, 2 });
        teacher[0, 0] = (Half)(-20.0f);
        teacher[0, 1] = (Half)0.0f;

        double loss = (double)new DINOLoss<Half>(2).ComputeLoss(student, teacher, updateCenter: false);

        Assert.Equal(14 * Math.Log(2), loss, 2);
    }

    [Fact]
    public void CosineSimilarity_Half_ZeroRowIsZeroNotNaN()
    {
        // Row 0 of z1 is zero: sqrt(0 + 1e-8) was sqrt(0) = 0 in Half and 0 / 0 made the mean NaN.
        // With a representable guard the zero row contributes 0 and row 1 contributes 1: mean 0.5.
        var z1 = new Tensor<Half>(new[] { 2, 2 });
        z1[1, 0] = (Half)1.0f;
        var z2 = new Tensor<Half>(new[] { 2, 2 });
        z2[0, 0] = (Half)1.0f;
        z2[1, 0] = (Half)1.0f;

        double similarity = (double)new SelfSupervisedLearningMetrics<Half>().ComputeCosineSimilarity(z1, z2);

        Assert.Equal(0.5, similarity, 2);
    }

    [Fact]
    public void Epsilon_Half_UnderflowingGuardUsesSmallestNormal()
    {
        Assert.Equal(Math.Pow(2, -14), (double)ObjectiveOps.Epsilon<Half>(1e-8));
        Assert.Equal(1e-8, ObjectiveOps.Epsilon<double>(1e-8));
    }

    [Fact]
    public void BilinearCropMatrix_Half_KeepsTheSourceFraction()
    {
        // size 1500, crop 500, offset 1000: output 1 reads source 1000 + 500 / 1500 = 1000.333..., so it weighs
        // pixel 1000 by 2/3 and pixel 1001 by 1/3. Computed in Half the position rounded to 1000.5 (Half resolves
        // 0.5 there) and both weights were 0.5.
        var method = typeof(AiDotNet.SelfSupervisedLearning.SelfSupervisedLearningAugmentationPolicies<Half>)
            .GetMethod("BilinearCropMatrix", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        Assert.NotNull(method);
        var matrix = method?.Invoke(null, new object[] { 1500, 500, 1000 }) as Tensor<Half>;
        Assert.NotNull(matrix);

        Assert.Equal(2.0 / 3.0, (double)(matrix?[1, 1000] ?? Half.NaN), 3);
        Assert.Equal(1.0 / 3.0, (double)(matrix?[1, 1001] ?? Half.NaN), 3);
    }

    [Fact]
    public void PretrainerFeatureStdDev_Half_SquaresBeyondHalfRange()
    {
        // Rows 300 and -300: unbiased variance (90000 + 90000) / 1, std 424.26. Each square overflows in Half.
        var matrix = new Tensor<Half>(new[] { 2, 1 });
        matrix[0, 0] = (Half)300.0f;
        matrix[1, 0] = (Half)(-300.0f);

        double std = AiDotNet.SelfSupervisedLearning.SelfSupervisedLearningPretrainer.FeatureStdDev(matrix);

        Assert.Equal(Math.Sqrt(180000.0), std, 6);
    }

    [Fact]
    public void PretrainerRidgeR2_Half_NormalEquationsBeyondHalfRange()
    {
        // y = 2x + 1 on x = 100..400 fits almost exactly (ridge 1e-3). FᵀF holds Σx² = 300000, past Half's 65504.
        var features = new Tensor<Half>(new[] { 4, 1 });
        var targets = new Tensor<Half>(new[] { 4, 1 });
        for (int i = 0; i < 4; i++)
        {
            features[i, 0] = (Half)(100.0f * (i + 1));
            targets[i, 0] = (Half)(200.0f * (i + 1) + 1.0f);
        }

        double r2 = AiDotNet.SelfSupervisedLearning.SelfSupervisedLearningPretrainer.RidgeR2(features, targets);

        Assert.InRange(r2, 0.999, 1.0);
    }
}
