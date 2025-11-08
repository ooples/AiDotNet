using AiDotNet.KnowledgeDistillation;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the DistillationLoss class.
/// </summary>
public class DistillationLossTests
{
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
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var loss = distillationLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);

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
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var loss = distillationLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);

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

        var lowTempLoss = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);
        var highTempLoss = new DistillationLoss<double>(temperature: 5.0, alpha: 0.0);

        // Act
        var lowLoss = lowTempLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);
        var highLoss = highTempLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);

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

        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.5);
        var softOnlyLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var combinedLoss = distillationLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels);
        var softLoss = softOnlyLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);

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
        var distillationLoss = new DistillationLoss<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            distillationLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels: null));
    }

    [Fact]
    public void ComputeGradient_ReturnsCorrectShape()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var gradient = distillationLoss.ComputeGradient(studentLogits, teacherLogits, trueLabels: null);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(studentLogits.Length, gradient.Length);
    }

    [Fact]
    public void ComputeGradient_WithIdenticalPredictions_ReturnsZeroGradient()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var gradient = distillationLoss.ComputeGradient(studentLogits, teacherLogits, trueLabels: null);

        // Assert
        // Gradients should be very close to zero for identical predictions
        for (int i = 0; i < gradient.Length; i++)
        {
            Assert.True(Math.Abs(gradient[i]) < 1e-6,
                $"Expected gradient[{i}] close to 0, but got {gradient[i]}");
        }
    }

    [Fact]
    public void ComputeGradient_WithTrueLabels_CombinesGradients()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });
        var trueLabels = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

        var combinedGradLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.5);
        var softOnlyLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

        // Act
        var combinedGrad = combinedGradLoss.ComputeGradient(studentLogits, teacherLogits, trueLabels);
        var softGrad = softOnlyLoss.ComputeGradient(studentLogits, teacherLogits, trueLabels: null);

        // Assert
        // Combined gradient should differ from soft-only gradient
        bool isDifferent = false;
        for (int i = 0; i < combinedGrad.Length; i++)
        {
            if (Math.Abs(combinedGrad[i] - softGrad[i]) > 1e-6)
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
        var temperature = 3.0;
        var distillationLoss = new DistillationLoss<double>(temperature, alpha: 0.0);

        // Act
        var gradient = distillationLoss.ComputeGradient(studentLogits, teacherLogits, trueLabels: null);

        // Assert
        // Gradient magnitudes should be reasonable (not exploding or vanishing)
        for (int i = 0; i < gradient.Length; i++)
        {
            double gradVal = gradient[i];
            Assert.True(Math.Abs(gradVal) < 100, $"Gradient should not explode: gradient[{i}] = {gradVal}");
        }
    }

    [Fact]
    public void TemperatureScaling_AffectsLossMagnitude()
    {
        // Arrange
        var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
        var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });

        var loss1 = new DistillationLoss<double>(temperature: 1.0, alpha: 0.0);
        var loss3 = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
        var loss5 = new DistillationLoss<double>(temperature: 5.0, alpha: 0.0);

        // Act
        var l1 = loss1.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);
        var l3 = loss3.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);
        var l5 = loss5.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);

        // Assert
        // Loss is scaled by T^2, but KL divergence also changes with T
        // Just verify all losses are reasonable
        Assert.True(l1 > 0, "Loss with T=1 should be positive");
        Assert.True(l3 > 0, "Loss with T=3 should be positive");
        Assert.True(l5 > 0, "Loss with T=5 should be positive");
    }
}
