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
    public void Constructor_SetsDefaultLearningRate()
    {
        // Arrange & Act
        var config = new ContinualLearnerConfig<double>();

        // Assert - default is 0.001 (industry standard for Adam optimizer)
        Assert.Equal(0.001, config.LearningRate);
    }

    [Fact]
    public void Constructor_SetsDefaultBatchSize()
    {
        // Arrange & Act
        var config = new ContinualLearnerConfig<double>();

        // Assert - default is 32 (balance of speed and gradient noise)
        Assert.Equal(32, config.BatchSize);
    }

    [Fact]
    public void Constructor_SetsDefaultEpochsPerTask()
    {
        // Arrange & Act
        var config = new ContinualLearnerConfig<double>();

        // Assert - default is 10
        Assert.Equal(10, config.EpochsPerTask);
    }

    [Fact]
    public void Constructor_SetsDefaultEwcLambda()
    {
        // Arrange & Act
        var config = new ContinualLearnerConfig<double>();

        // Assert - default is 1000.0 (based on Kirkpatrick et al. 2017)
        Assert.Equal(1000.0, config.EwcLambda);
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
