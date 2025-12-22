using AiDotNet.ContinualLearning.Config;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// Unit tests for the ContinualLearnerConfig class.
/// </summary>
public class ContinualLearnerConfigTests
{
    [Fact]
    public void Constructor_DefaultParameters_InitializesSuccessfully()
    {
        // Act
        var config = new ContinualLearnerConfig<double>();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.IsValid());
    }

    [Fact]
    public void PropertyInitialization_CustomParameters_InitializesSuccessfully()
    {
        // Arrange & Act
        var config = new ContinualLearnerConfig<double>
        {
            LearningRate = 0.001,
            EpochsPerTask = 10,
            BatchSize = 32,
            MemorySize = 1000,
            EwcLambda = 1000.0
        };

        // Assert
        Assert.NotNull(config);
        Assert.Equal(0.001, config.LearningRate);
        Assert.Equal(10, config.EpochsPerTask);
        Assert.Equal(32, config.BatchSize);
        Assert.Equal(1000, config.MemorySize);
        Assert.Equal(1000.0, config.EwcLambda);
        Assert.True(config.IsValid());
    }

    [Fact]
    public void IsValid_WithNegativeLearningRate_ReturnsFalse()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>
        {
            LearningRate = -0.001,
            EpochsPerTask = 10,
            BatchSize = 32,
            MemorySize = 1000
        };

        // Act & Assert
        Assert.False(config.IsValid());
    }

    [Fact]
    public void IsValid_WithZeroEpochs_ReturnsFalse()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>
        {
            LearningRate = 0.001,
            EpochsPerTask = 0,
            BatchSize = 32,
            MemorySize = 1000
        };

        // Act & Assert
        Assert.False(config.IsValid());
    }

    [Fact]
    public void IsValid_WithNegativeBatchSize_ReturnsFalse()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>
        {
            LearningRate = 0.001,
            EpochsPerTask = 10,
            BatchSize = -1,
            MemorySize = 1000
        };

        // Act & Assert
        Assert.False(config.IsValid());
    }

    [Fact]
    public void GetEffectiveLearningRate_NullValue_ReturnsDefault()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>();

        // Act
        var effectiveLR = config.GetEffectiveLearningRate();

        // Assert
        Assert.Equal(0.001, effectiveLR);
    }

    [Fact]
    public void GetEffectiveBatchSize_NullValue_ReturnsDefault()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>();

        // Act
        var effectiveBatchSize = config.GetEffectiveBatchSize();

        // Assert
        Assert.Equal(32, effectiveBatchSize);
    }

    [Fact]
    public void GetEffectiveEpochsPerTask_NullValue_ReturnsDefault()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>();

        // Act
        var effectiveEpochs = config.GetEffectiveEpochsPerTask();

        // Assert
        Assert.Equal(10, effectiveEpochs);
    }

    [Fact]
    public void GetEffectiveEwcLambda_NullValue_ReturnsDefault()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>();

        // Act
        var effectiveLambda = config.GetEffectiveEwcLambda();

        // Assert
        Assert.Equal(1000.0, effectiveLambda);
    }

    [Fact]
    public void ForEwc_CreatesEwcOptimizedConfig()
    {
        // Act
        var config = ContinualLearnerConfig<double>.ForEwc(lambda: 5000.0, fisherSamples: 500);

        // Assert
        Assert.NotNull(config);
        Assert.Equal(5000.0, config.EwcLambda);
        Assert.Equal(500, config.FisherSamples);
        Assert.True(config.IsValid());
    }

    [Fact]
    public void ForLwf_CreatesLwfOptimizedConfig()
    {
        // Act
        var config = ContinualLearnerConfig<double>.ForLwf(temperature: 4.0, weight: 2.0);

        // Assert
        Assert.NotNull(config);
        Assert.Equal(4.0, config.DistillationTemperature);
        Assert.Equal(2.0, config.DistillationWeight);
        Assert.True(config.IsValid());
    }

    [Fact]
    public void ForGem_CreatesGemOptimizedConfig()
    {
        // Act
        var config = ContinualLearnerConfig<double>.ForGem(memoryStrength: 0.8, memorySize: 2000);

        // Assert
        Assert.NotNull(config);
        Assert.Equal(0.8, config.GemMemoryStrength);
        Assert.Equal(2000, config.MemorySize);
        Assert.True(config.IsValid());
    }

    [Fact]
    public void IsValid_WithInvalidGemMemoryStrength_ReturnsFalse()
    {
        // Arrange - GEM memory strength must be between 0 and 1
        var config = new ContinualLearnerConfig<double>
        {
            GemMemoryStrength = 1.5
        };

        // Act & Assert
        Assert.False(config.IsValid());
    }

    [Fact]
    public void IsValid_WithInvalidPackNetPruneRatio_ReturnsFalse()
    {
        // Arrange - PackNet prune ratio must be > 0 and < 1
        var config = new ContinualLearnerConfig<double>
        {
            PackNetPruneRatio = 1.0
        };

        // Act & Assert
        Assert.False(config.IsValid());
    }
}
