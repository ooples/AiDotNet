using AiDotNet.ActiveLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActiveLearning;

/// <summary>
/// Unit tests for the Uncertainty Sampling query strategy.
/// </summary>
public class UncertaintySamplingTests
{
    [Fact(Timeout = 60000)]
    public void Constructor_LeastConfidence_InitializesSuccessfully()
    {
        // Act
        var strategy = new UncertaintySampling<double>(
            UncertaintySampling<double>.UncertaintyMeasure.LeastConfidence);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-LeastConfidence", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public void Constructor_MarginSampling_InitializesSuccessfully()
    {
        // Act
        var strategy = new UncertaintySampling<double>(
            UncertaintySampling<double>.UncertaintyMeasure.MarginSampling);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-MarginSampling", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public void Constructor_Entropy_InitializesSuccessfully()
    {
        // Act
        var strategy = new UncertaintySampling<double>(
            UncertaintySampling<double>.UncertaintyMeasure.Entropy);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-Entropy", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public void Constructor_DefaultParameter_UsesEntropy()
    {
        // Act
        var strategy = new UncertaintySampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-Entropy", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public void UseBatchDiversity_DefaultsFalse()
    {
        // Act
        var strategy = new UncertaintySampling<double>();

        // Assert
        Assert.False(strategy.UseBatchDiversity);
    }

    [Fact(Timeout = 60000)]
    public void UseBatchDiversity_CanBeSet()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();

        // Act
        strategy.UseBatchDiversity = true;

        // Assert
        Assert.True(strategy.UseBatchDiversity);
    }

    [Fact(Timeout = 60000)]
    public void GetSelectionStatistics_ReturnsEmptyInitially()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();

        // Act
        var stats = strategy.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Contains("MinScore", stats.Keys);
        Assert.Contains("MaxScore", stats.Keys);
        Assert.Contains("MeanScore", stats.Keys);
    }
}
