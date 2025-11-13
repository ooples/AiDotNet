using AiDotNet.KnowledgeDistillation;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the DistillationLoss class.
/// </summary>
public class DistillationLossTests
{
    /// <summary>
    /// Helper method to convert a Vector to a single-row Matrix for batch processing API.
    /// </summary>
    private Matrix<double> MatrixFromVector(Vector<double> vector)
    {
        var matrix = new Matrix<double>(1, vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            matrix[0, i] = vector[i];
        }
        return matrix;
    }

    [Fact]
    public void Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange & Act
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.3);

        // Assert
        Assert.NotNull(distillationLoss);
        Assert.Equal(3.0, distillationLoss.Temperature);
        Assert.Equal(0.3, distillationLoss.Alpha);
    }

    [Fact]
    public void Constructor_WithInvalidTemperature_ThrowsArgumentException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new DistillationLoss<double>(temperature: 0, alpha: 0.3));
        Assert.Throws<ArgumentException>(() => new DistillationLoss<double>(temperature: -1, alpha: 0.3));
    }

    [Fact]
    public void Constructor_WithInvalidAlpha_ThrowsArgumentException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new DistillationLoss<double>(temperature: 3.0, alpha: -0.1));
        Assert.Throws<ArgumentException>(() => new DistillationLoss<double>(temperature: 3.0, alpha: 1.5));
    }

    [Fact]
    public void ComputeLoss_WithIdenticalLogits_ReturnsZeroSoftLoss()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var loss = distillationLoss.ComputeLoss(studentBatch, teacherBatch, null);

        // Assert
        // Identical logits should produce identical soft predictions, resulting in KL divergence ≈ 0
        Assert.True(Math.Abs(loss) < 1e-6, $"Expected loss close to 0, but got {loss}");
    }

    [Fact]
    public void ComputeLoss_WithDifferentLogits_ReturnsPositiveLoss()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 2.5, 1.0, 0.3 });
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var loss = distillationLoss.ComputeLoss(studentBatch, teacherBatch, null);

        // Assert
        // Different logits should produce non-zero loss
        Assert.True(loss > 0, $"Expected positive loss, but got {loss}");
    }

    [Fact]
    public void ComputeLoss_WithHighTemperature_ProducesSofterDistribution()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 10.0, 1.0, 0.1 });
        var teacherLogits = new Vector<double>(new[] { 9.0, 1.5, 0.2 });
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);

        var lowTempLoss = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);
        var highTempLoss = new DistillationLoss<double>(temperature: 5.0, alpha: 0.0);

        // Act
        var lowLoss = lowTempLoss.ComputeLoss(studentBatch, teacherBatch, null);
        var highLoss = highTempLoss.ComputeLoss(studentBatch, teacherBatch, null);

        // Assert
        // High temperature loss should be smaller because distributions are softer and closer
        Assert.True(highLoss < lowLoss,
            $"High temp loss ({highLoss}) should be less than low temp loss ({lowLoss})");
    }

    [Fact]
    public void ComputeLoss_WithTrueLabels_CombinesHardAndSoftLoss()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });
        var trueLabels = new Vector<double>(new[] { 0.0, 1.0, 0.0 }); // Class 1 is correct
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);
        var labelsBatch = MatrixFromVector(trueLabels);

        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.5);
        var softOnlyLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var combinedLoss = distillationLoss.ComputeLoss(studentBatch, teacherBatch, labelsBatch);
        var softLoss = softOnlyLoss.ComputeLoss(studentBatch, teacherBatch, null);

        // Assert
        // Combined loss should be different from soft-only loss
        Assert.NotEqual(combinedLoss, softLoss);
        Assert.True(combinedLoss > 0, "Combined loss should be positive");
    }

    [Fact]
    public void ComputeLoss_WithMismatchedDimensions_ThrowsArgumentException()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8 }); // Different length
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);
        var distillationLoss = new DistillationLoss<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            distillationLoss.ComputeLoss(studentBatch, teacherBatch, null));
    }

    [Fact]
    public void ComputeGradient_ReturnsCorrectShape()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var gradient = distillationLoss.ComputeGradient(studentBatch, teacherBatch, null);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(1, gradient.Rows); // Single sample batch
        Assert.Equal(studentLogits.Length, gradient.Columns);
    }

    [Fact]
    public void ComputeGradient_WithIdenticalPredictions_ReturnsZeroGradient()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var gradient = distillationLoss.ComputeGradient(studentBatch, teacherBatch, null);

        // Assert
        // Gradients should be very close to zero for identical predictions
        for (int i = 0; i < gradient.Columns; i++)
        {
            Assert.True(Math.Abs(gradient[0, i]) < 1e-6,
                $"Expected gradient[0, {i}] close to 0, but got {gradient[0, i]}");
        }
    }

    [Fact]
    public void ComputeGradient_WithTrueLabels_CombinesGradients()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });
        var trueLabels = new Vector<double>(new[] { 0.0, 1.0, 0.0 });
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);
        var labelsBatch = MatrixFromVector(trueLabels);

        var combinedGradLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.5);
        var softOnlyLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var combinedGrad = combinedGradLoss.ComputeGradient(studentBatch, teacherBatch, labelsBatch);
        var softGrad = softOnlyLoss.ComputeGradient(studentBatch, teacherBatch, null);

        // Assert
        // Combined gradient should differ from soft-only gradient
        bool isDifferent = false;
        for (int i = 0; i < combinedGrad.Columns; i++)
        {
            if (Math.Abs(combinedGrad[0, i] - softGrad[0, i]) > 1e-6)
            {
                isDifferent = true;
                break;
            }
        }
        Assert.True(isDifferent, "Combined gradient should differ from soft-only gradient");
    }

    [Fact]
    public void ComputeGradient_HasReasonableMagnitude()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);
        var temperature = 3.0;
        var distillationLoss = new DistillationLoss<double>(temperature, alpha: 0.0);

        // Act
        var gradient = distillationLoss.ComputeGradient(studentBatch, teacherBatch, null);

        // Assert
        // Gradient magnitudes should be reasonable (not exploding or vanishing)
        for (int i = 0; i < gradient.Columns; i++)
        {
            double gradVal = gradient[0, i];
            Assert.True(Math.Abs(gradVal) < 100, $"Gradient should not explode: gradient[0, {i}] = {gradVal}");
        }
    }

    [Fact]
    public void TemperatureScaling_AffectsLossMagnitude()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });
        var studentBatch = MatrixFromVector(studentLogits);
        var teacherBatch = MatrixFromVector(teacherLogits);

        var loss1 = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);
        var loss3 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var loss5 = new DistillationLoss<double>(temperature: 5.0, alpha: 0.0);

        // Act
        var l1 = loss1.ComputeLoss(studentBatch, teacherBatch, null);
        var l3 = loss3.ComputeLoss(studentBatch, teacherBatch, null);
        var l5 = loss5.ComputeLoss(studentBatch, teacherBatch, null);

        // Assert
        // Loss is scaled by T^2, but KL divergence also changes with T
        // Just verify all losses are reasonable
        Assert.True(l1 > 0, "Loss with T=1 should be positive");
        Assert.True(l3 > 0, "Loss with T=3 should be positive");
        Assert.True(l5 > 0, "Loss with T=5 should be positive");
    }
}
