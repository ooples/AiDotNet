using System;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

public class SparseTopKTailBucketDistillationStrategyTests
{
    [Fact]
    public void TopKEqualsVocab_MatchesStandardResponseDistillation()
    {
        const int batch = 3;
        const int vocab = 12;
        var rng = RandomHelper.CreateSeededRandom(42);
        var student = new Matrix<double>(batch, vocab);
        var teacher = new Matrix<double>(batch, vocab);
        var labels = new Matrix<double>(batch, vocab);
        for (int r = 0; r < batch; r++)
        {
            labels[r, (r * 5 + 1) % vocab] = 1.0;
            for (int c = 0; c < vocab; c++)
            {
                student[r, c] = (rng.NextDouble() - 0.5) * 3.0;
                teacher[r, c] = (rng.NextDouble() - 0.5) * 3.0;
            }
        }

        var sparseFull = new SparseTopKTailBucketDistillationStrategy<double>(topK: vocab, temperature: 2.0, alpha: 0.4);
        var standard = new DistillationLoss<double>(temperature: 2.0, alpha: 0.4);

        double lossSparse = sparseFull.ComputeLoss(student, teacher, labels);
        double lossStd = standard.ComputeLoss(student, teacher, labels);
        Assert.Equal(lossStd, lossSparse, precision: 6);

        var gradSparse = sparseFull.ComputeGradient(student, teacher, labels);
        var gradStd = standard.ComputeGradient(student, teacher, labels);
        for (int r = 0; r < batch; r++)
        {
            for (int c = 0; c < vocab; c++)
            {
                Assert.Equal(gradStd[r, c], gradSparse[r, c], precision: 6);
            }
        }
    }

    [Fact]
    public void StudentMatchingTeacherUpToShift_HasZeroSoftLossAndZeroTailGradient()
    {
        const int batch = 2;
        const int vocab = 64;
        var rng = RandomHelper.CreateSeededRandom(101);
        var teacher = new Matrix<double>(batch, vocab);
        var student = new Matrix<double>(batch, vocab);
        for (int r = 0; r < batch; r++)
        {
            double shift = (r + 1) * 3.7;
            for (int c = 0; c < vocab; c++)
            {
                double v = (rng.NextDouble() - 0.5) * 1.5;
                teacher[r, c] = v;
                student[r, c] = v + shift;
            }
        }

        var strategy = (SparseTopKTailBucketDistillationStrategy<double>)
            DistillationStrategyFactory<double>.CreateSparseTopKTailBucketStrategy(topK: 6, temperature: 2.0, alpha: 0.0);

        double loss = strategy.ComputeLoss(student, teacher, trueLabelsBatch: null);
        Assert.True(Math.Abs(loss) < 1e-11, $"Expected ~0 soft KL loss at shifted teacher logits, got {loss:E4}");

        var grad = strategy.ComputeGradient(student, teacher, trueLabelsBatch: null);
        for (int r = 0; r < batch; r++)
        {
            for (int c = 0; c < vocab; c++)
            {
                Assert.True(Math.Abs(grad[r, c]) < 1e-11, $"Non-zero gradient {grad[r, c]:E4} at [{r},{c}] (tail starvation regression)");
            }
        }
    }

    [Fact]
    public void ComputeGradient_MatchesCentralFiniteDifference_WithSmallTopKAndTailMass()
    {
        const int batch = 2;
        const int vocab = 24;
        var rng = RandomHelper.CreateSeededRandom(2026);
        var student = new Matrix<double>(batch, vocab);
        var teacher = new Matrix<double>(batch, vocab);
        var labels = new Matrix<double>(batch, vocab);
        for (int r = 0; r < batch; r++)
        {
            labels[r, (r * 7 + 3) % vocab] = 1.0;
            for (int c = 0; c < vocab; c++)
            {
                student[r, c] = (rng.NextDouble() - 0.5) * 2.4;
                teacher[r, c] = (rng.NextDouble() - 0.5) * 2.0;
            }
        }

        var strategy = new SparseTopKTailBucketDistillationStrategy<double>(topK: 4, temperature: 2.0, alpha: 0.35);
        strategy.ExtractTopKTargets(teacher, out var topIdx, out var topProbs);
        var anaGrad = strategy.ComputeGradientFromSparseTargets(student, topIdx, topProbs, labels);

        const double h = 1e-6;
        for (int r = 0; r < batch; r++)
        {
            for (int c = 0; c < vocab; c++)
            {
                double orig = student[r, c];
                student[r, c] = orig + h;
                double lp = strategy.ComputeLossFromSparseTargets(student, topIdx, topProbs, labels);
                student[r, c] = orig - h;
                double lm = strategy.ComputeLossFromSparseTargets(student, topIdx, topProbs, labels);
                student[r, c] = orig;

                double num = (lp - lm) / (2.0 * h);
                Assert.True(Math.Abs(num - anaGrad[r, c]) < 1e-7,
                    $"Grad mismatch at [{r},{c}]: num={num:E8}, ana={anaGrad[r, c]:E8}");
            }
        }
    }

    [Theory]
    [InlineData(40.0, 0.0, 0.25, -0.25)]
    [InlineData(-40.0, 0.0, -0.75, 0.75)]
    public void ExtremeStudentLogits_PreserveExactLogDomainLossAndGradient(
        double selectedLogit,
        double tailLogit,
        double expectedSelectedGrad,
        double expectedTailGrad)
    {
        var strategy = new SparseTopKTailBucketDistillationStrategy<double>(topK: 1, temperature: 1.0, alpha: 0.0);
        var student = new Matrix<double>(1, 2);
        student[0, 0] = selectedLogit;
        student[0, 1] = tailLogit;

        var topIdx = new Tensor<int>(new[] { 1, 1 });
        topIdx[0, 0] = 0;
        var topProbs = new Matrix<double>(1, 1);
        topProbs[0, 0] = 0.75;

        var anaGrad = strategy.ComputeGradientFromSparseTargets(student, topIdx, topProbs);
        Assert.Equal(expectedSelectedGrad, anaGrad[0, 0], precision: 9);
        Assert.Equal(expectedTailGrad, anaGrad[0, 1], precision: 9);

        const double h = 1e-6;
        for (int c = 0; c < 2; c++)
        {
            double orig = student[0, c];
            student[0, c] = orig + h;
            double lp = strategy.ComputeLossFromSparseTargets(student, topIdx, topProbs);
            student[0, c] = orig - h;
            double lm = strategy.ComputeLossFromSparseTargets(student, topIdx, topProbs);
            student[0, c] = orig;

            double num = (lp - lm) / (2.0 * h);
            Assert.True(Math.Abs(num - anaGrad[0, c]) < 1e-6,
                $"Extreme-logit grad mismatch at col={c}: num={num:E9}, ana={anaGrad[0, c]:E9}");
        }
    }

    [Fact]
    public void DuplicateRowClassIndices_ThrowsArgumentException()
    {
        var strategy = new SparseTopKTailBucketDistillationStrategy<double>(topK: 2, temperature: 1.0, alpha: 0.0);
        var student = new Matrix<double>(1, 3);
        var topIdx = new Tensor<int>(new[] { 1, 2 });
        topIdx[0, 0] = 0;
        topIdx[0, 1] = 0;
        var topProbs = new Matrix<double>(1, 2);
        topProbs[0, 0] = 0.4;
        topProbs[0, 1] = 0.3;

        Assert.Throws<ArgumentException>(() => strategy.ComputeLossFromSparseTargets(student, topIdx, topProbs));
        Assert.Throws<ArgumentException>(() => strategy.ComputeGradientFromSparseTargets(student, topIdx, topProbs));
    }

    [Theory]
    [InlineData(double.NaN, 0.2)]
    [InlineData(double.PositiveInfinity, 0.2)]
    [InlineData(double.NegativeInfinity, 0.2)]
    [InlineData(-0.05, 0.2)]
    [InlineData(1.05, 0.0)]
    [InlineData(0.6, 0.5)]
    public void InvalidSparseProbabilities_ThrowsArgumentException(double p0, double p1)
    {
        var strategy = new SparseTopKTailBucketDistillationStrategy<double>(topK: 2, temperature: 1.0, alpha: 0.0);
        var student = new Matrix<double>(1, 4);
        var topIdx = new Tensor<int>(new[] { 1, 2 });
        topIdx[0, 0] = 0;
        topIdx[0, 1] = 2;
        var topProbs = new Matrix<double>(1, 2);
        topProbs[0, 0] = p0;
        topProbs[0, 1] = p1;

        Assert.Throws<ArgumentException>(() => strategy.ComputeLossFromSparseTargets(student, topIdx, topProbs));
        Assert.Throws<ArgumentException>(() => strategy.ComputeGradientFromSparseTargets(student, topIdx, topProbs));
    }

    [Fact]
    public void FloatPrecisionExtractTopKTargets_PassesRowSumValidation()
    {
        var rng = RandomHelper.CreateSeededRandom(777);
        var student = new Matrix<float>(4, 128);
        var teacher = new Matrix<float>(4, 128);
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 128; c++)
            {
                student[r, c] = (float)((rng.NextDouble() - 0.5) * 4.0);
                teacher[r, c] = (float)((rng.NextDouble() - 0.5) * 4.0);
            }
        }

        var strategy = new SparseTopKTailBucketDistillationStrategy<float>(topK: 64, temperature: 1.0, alpha: 0.0);
        float loss = strategy.ComputeLoss(student, teacher);
        var grad = strategy.ComputeGradient(student, teacher);

        Assert.False(float.IsNaN(loss) || float.IsInfinity(loss));
        Assert.Equal(4, grad.Rows);
        Assert.Equal(128, grad.Columns);
    }
}
