namespace AiDotNet.Tests.FederatedLearning;

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.FederatedLearning.Privacy;
using Xunit;

/// <summary>
/// Unit tests for Gaussian Differential Privacy mechanism.
/// </summary>
public class GaussianDifferentialPrivacyTests
{
    [Fact]
    public void Constructor_WithValidClipNorm_InitializesSuccessfully()
    {
        // Arrange & Act
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0);

        // Assert
        Assert.NotNull(dp);
        Assert.Equal(0.0, dp.GetPrivacyBudgetConsumed());
    }

    [Fact]
    public void Constructor_WithNegativeClipNorm_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new GaussianDifferentialPrivacy<double>(clipNorm: -1.0));
    }

    [Fact]
    public void ApplyPrivacy_WithValidParameters_AddsNoiseToModel()
    {
        // Arrange
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 10.0, randomSeed: 42);

        var originalModel = new Dictionary<string, double[]>
        {
            ["layer1"] = new double[] { 1.0, 2.0, 3.0 }
        };

        // Act
        var noisyModel = dp.ApplyPrivacy(originalModel, epsilon: 1.0, delta: 1e-5);

        // Assert
        Assert.NotNull(noisyModel);
        Assert.Contains("layer1", noisyModel.Keys);

        // Model should be different due to noise
        bool hasNoise = false;
        for (int i = 0; i < originalModel["layer1"].Length; i++)
        {
            if (Math.Abs(noisyModel["layer1"][i] - originalModel["layer1"][i]) > 0.0001)
            {
                hasNoise = true;
                break;
            }
        }
        Assert.True(hasNoise, "Noise should have been added to the model");
    }

    [Fact]
    public void ApplyPrivacy_UpdatesPrivacyBudget()
    {
        // Arrange
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0);

        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new double[] { 1.0 }
        };

        // Act
        dp.ApplyPrivacy(model, epsilon: 0.5, delta: 1e-5);

        // Assert
        Assert.Equal(0.5, dp.GetPrivacyBudgetConsumed());

        // Apply privacy again
        dp.ApplyPrivacy(model, epsilon: 0.3, delta: 1e-5);
        Assert.Equal(0.8, dp.GetPrivacyBudgetConsumed(), precision: 5);
    }

    [Fact]
    public void ApplyPrivacy_WithZeroEpsilon_ThrowsArgumentException()
    {
        // Arrange
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0);

        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new double[] { 1.0 }
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: 0.0, delta: 1e-5));
    }

    [Fact]
    public void ApplyPrivacy_WithInvalidDelta_ThrowsArgumentException()
    {
        // Arrange
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0);

        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new double[] { 1.0 }
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: 1.0, delta: 0.0));
        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: 1.0, delta: 1.0));
    }

    [Fact]
    public void ResetPrivacyBudget_ResetsToZero()
    {
        // Arrange
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0);

        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new double[] { 1.0 }
        };

        dp.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);
        Assert.Equal(1.0, dp.GetPrivacyBudgetConsumed());

        // Act
        dp.ResetPrivacyBudget();

        // Assert
        Assert.Equal(0.0, dp.GetPrivacyBudgetConsumed());
    }

    [Fact]
    public void GetMechanismName_ReturnsCorrectName()
    {
        // Arrange
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 2.5);

        // Act
        var name = dp.GetMechanismName();

        // Assert
        Assert.Contains("Gaussian DP", name);
        Assert.Contains("2.5", name);
    }

    [Fact]
    public void ApplyPrivacy_WithSameSeed_ProducesSameNoise()
    {
        // Arrange
        var dp1 = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 123);
        var dp2 = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 123);

        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new double[] { 1.0, 2.0, 3.0 }
        };

        // Act
        var noisyModel1 = dp1.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);
        var noisyModel2 = dp2.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);

        // Assert
        for (int i = 0; i < noisyModel1["layer1"].Length; i++)
        {
            Assert.Equal(noisyModel1["layer1"][i], noisyModel2["layer1"][i], precision: 10);
        }
    }

    [Fact]
    public void ApplyPrivacy_PerformsGradientClipping()
    {
        // Arrange
        var clipNorm = 1.0;
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: clipNorm, randomSeed: 42);

        // Create model with large norm (sqrt(100 + 100 + 100) = ~17.3)
        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new double[] { 10.0, 10.0, 10.0 }
        };

        // Act
        var clippedModel = dp.ApplyPrivacy(model, epsilon: 10.0, delta: 1e-5);

        // Assert
        // Calculate L2 norm of clipped model
        double sumSquares = clippedModel["layer1"].Sum(x => x * x);
        double norm = Math.Sqrt(sumSquares);

        // Norm should be approximately clipNorm (plus some noise variance)
        // With high epsilon (10.0), noise is minimal, so norm should be close to clipNorm
        Assert.True(norm < clipNorm * 2.0, $"Norm {norm} should be reasonably close to clip norm {clipNorm}");
    }
}
