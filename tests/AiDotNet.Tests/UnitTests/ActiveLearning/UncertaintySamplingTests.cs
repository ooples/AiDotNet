using AiDotNet.ActiveLearning.QueryStrategies;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActiveLearning;

/// <summary>
/// Unit tests for the Uncertainty Sampling query strategy.
/// </summary>
public class UncertaintySamplingTests
{
    [Fact]
    public void Constructor_LeastConfidence_InitializesSuccessfully()
    {
        // Act
        var strategy = new UncertaintySampling<double, Matrix<double>, Vector<double>>(
            UncertaintySampling<double, Matrix<double>, Vector<double>>.UncertaintyMeasure.LeastConfidence);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-LeastConfidence", strategy.Name);
    }

    [Fact]
    public void Constructor_Margin_InitializesSuccessfully()
    {
        // Act
        var strategy = new UncertaintySampling<double, Matrix<double>, Vector<double>>(
            UncertaintySampling<double, Matrix<double>, Vector<double>>.UncertaintyMeasure.Margin);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-Margin", strategy.Name);
    }

    [Fact]
    public void Constructor_Entropy_InitializesSuccessfully()
    {
        // Act
        var strategy = new UncertaintySampling<double, Matrix<double>, Vector<double>>(
            UncertaintySampling<double, Matrix<double>, Vector<double>>.UncertaintyMeasure.Entropy);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-Entropy", strategy.Name);
    }

    [Fact]
    public void Constructor_DefaultParameter_UsesLeastConfidence()
    {
        // Act
        var strategy = new UncertaintySampling<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-LeastConfidence", strategy.Name);
    }
}
