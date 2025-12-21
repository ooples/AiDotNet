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
    public void Constructor_CustomParameters_InitializesSuccessfully()
    {
        // Arrange & Act
        var config = new ContinualLearnerConfig<double>(
            learningRate: 0.001,
            epochsPerTask: 10,
            batchSize: 32,
            memorySize: 1000,
            regularizationStrength: 1000.0);

        // Assert
        Assert.NotNull(config);
        Assert.Equal(0.001, config.LearningRate);
        Assert.Equal(10, config.EpochsPerTask);
        Assert.Equal(32, config.BatchSize);
        Assert.Equal(1000, config.MemorySize);
        Assert.Equal(1000.0, config.RegularizationStrength);
        Assert.True(config.IsValid());
    }

    [Fact]
    public void IsValid_WithNegativeLearningRate_ReturnsFalse()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>(
            learningRate: -0.001,
            epochsPerTask: 10,
            batchSize: 32,
            memorySize: 1000,
            regularizationStrength: 1000.0);

        // Act & Assert
        Assert.False(config.IsValid());
    }

    [Fact]
    public void IsValid_WithZeroEpochs_ReturnsFalse()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>(
            learningRate: 0.001,
            epochsPerTask: 0,
            batchSize: 32,
            memorySize: 1000,
            regularizationStrength: 1000.0);

        // Act & Assert
        Assert.False(config.IsValid());
    }

    [Fact]
    public void IsValid_WithNegativeBatchSize_ReturnsFalse()
    {
        // Arrange
        var config = new ContinualLearnerConfig<double>(
            learningRate: 0.001,
            epochsPerTask: 10,
            batchSize: -1,
            memorySize: 1000,
            regularizationStrength: 1000.0);

        // Act & Assert
        Assert.False(config.IsValid());
    }
}
