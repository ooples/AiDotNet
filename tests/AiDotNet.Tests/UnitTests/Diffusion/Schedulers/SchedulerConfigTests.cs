using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// Unit tests for <see cref="SchedulerConfig{T}"/>.
/// </summary>
public class SchedulerConfigTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithDefaultValues_SetsCorrectDefaults()
    {
        // Arrange & Act
        var config = new SchedulerConfig<double>(
            trainTimesteps: 1000,
            betaStart: 0.0001,
            betaEnd: 0.02);

        // Assert
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(0.0001, config.BetaStart);
        Assert.Equal(0.02, config.BetaEnd);
        Assert.Equal(BetaSchedule.Linear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.Epsilon, config.PredictionType);
        Assert.False(config.ClipSample);
    }

    [Fact]
    public void Constructor_WithCustomValues_SetsAllProperties()
    {
        // Arrange & Act
        var config = new SchedulerConfig<double>(
            trainTimesteps: 500,
            betaStart: 0.001,
            betaEnd: 0.05,
            betaSchedule: BetaSchedule.ScaledLinear,
            predictionType: DiffusionPredictionType.VPrediction,
            clipSample: true);

        // Assert
        Assert.Equal(500, config.TrainTimesteps);
        Assert.Equal(0.001, config.BetaStart);
        Assert.Equal(0.05, config.BetaEnd);
        Assert.Equal(BetaSchedule.ScaledLinear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.VPrediction, config.PredictionType);
        Assert.True(config.ClipSample);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-100)]
    public void Constructor_WithInvalidTrainTimesteps_ThrowsArgumentOutOfRangeException(int invalidTimesteps)
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SchedulerConfig<double>(
                trainTimesteps: invalidTimesteps,
                betaStart: 0.0001,
                betaEnd: 0.02));
    }

    [Fact]
    public void Constructor_WithZeroBetaStart_DoesNotThrow()
    {
        // Arrange & Act - The implementation doesn't validate beta values
        // so we test that it accepts any beta values without throwing
        var config = new SchedulerConfig<double>(
            trainTimesteps: 1000,
            betaStart: 0.0,
            betaEnd: 0.02);

        // Assert
        Assert.Equal(0.0, config.BetaStart);
    }

    [Fact]
    public void Constructor_WithBetaEndLessThanBetaStart_DoesNotThrow()
    {
        // Arrange & Act - The implementation doesn't validate beta ordering
        var config = new SchedulerConfig<double>(
            trainTimesteps: 1000,
            betaStart: 0.02,
            betaEnd: 0.0001);

        // Assert
        Assert.Equal(0.02, config.BetaStart);
        Assert.Equal(0.0001, config.BetaEnd);
    }

    #endregion

    #region Factory Method Tests

    [Fact]
    public void CreateDefault_ReturnsStandardDDPMConfig()
    {
        // Arrange & Act
        var config = SchedulerConfig<double>.CreateDefault();

        // Assert
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(0.0001, config.BetaStart);
        Assert.Equal(0.02, config.BetaEnd);
        Assert.Equal(BetaSchedule.Linear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.Epsilon, config.PredictionType);
    }

    [Fact]
    public void CreateStableDiffusion_ReturnsStableDiffusionConfig()
    {
        // Arrange & Act
        var config = SchedulerConfig<double>.CreateStableDiffusion();

        // Assert
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(0.00085, config.BetaStart);
        Assert.Equal(0.012, config.BetaEnd);
        Assert.Equal(BetaSchedule.ScaledLinear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.Epsilon, config.PredictionType);
    }

    #endregion

    #region Generic Type Tests

    [Fact]
    public void Constructor_WithFloatType_WorksCorrectly()
    {
        // Arrange & Act
        var config = new SchedulerConfig<float>(
            trainTimesteps: 1000,
            betaStart: 0.0001f,
            betaEnd: 0.02f);

        // Assert
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(0.0001f, config.BetaStart);
        Assert.Equal(0.02f, config.BetaEnd);
    }

    #endregion
}
