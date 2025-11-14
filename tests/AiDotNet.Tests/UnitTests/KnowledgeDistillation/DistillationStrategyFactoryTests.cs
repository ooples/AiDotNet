using AiDotNet.Enums;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the DistillationStrategyFactory class.
/// </summary>
public class DistillationStrategyFactoryTests
{
    [Theory]
    [InlineData(DistillationStrategyType.ResponseBased)]
    //     [InlineData(DistillationStrategyType.FeatureBased)]
    [InlineData(DistillationStrategyType.AttentionBased)]
    [InlineData(DistillationStrategyType.RelationBased)]
    [InlineData(DistillationStrategyType.ContrastiveBased)]
    [InlineData(DistillationStrategyType.SimilarityPreserving)]
    [InlineData(DistillationStrategyType.ProbabilisticTransfer)]
    [InlineData(DistillationStrategyType.VariationalInformation)]
    [InlineData(DistillationStrategyType.FactorTransfer)]
    [InlineData(DistillationStrategyType.NeuronSelectivity)]
    [InlineData(DistillationStrategyType.Hybrid)]
    public void CreateStrategy_WithAllStrategyTypes_ReturnsValidStrategy(DistillationStrategyType strategyType)
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(strategyType);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal(3.0, strategy.Temperature); // Default temperature
        Assert.Equal(0.3, strategy.Alpha); // Default alpha
    }

    [Fact]
    public void CreateStrategy_ResponseBased_ReturnsDistillationLoss()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.ResponseBased,
            temperature: 4.0,
            alpha: 0.5);

        // Assert
        Assert.IsType<DistillationLoss<double>>(strategy);
        Assert.Equal(4.0, strategy.Temperature);
        Assert.Equal(0.5, strategy.Alpha);
    }

    [Fact]
    public void CreateStrategy_FeatureBased_ThrowsNotSupportedException()
    {
        // Arrange & Act & Assert
        var exception = Assert.Throws<NotSupportedException>(() =>
            DistillationStrategyFactory<double>.CreateStrategy(
                DistillationStrategyType.FeatureBased,
                featureWeight: 0.7));

        Assert.Contains("FeatureBased", exception.Message);
    }

    [Fact]
    public void CreateStrategy_AttentionBased_ReturnsAttentionDistillationStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.AttentionBased,
            attentionWeight: 0.6);

        // Assert
        Assert.IsType<AttentionDistillationStrategy<double>>(strategy);
    }

    [Fact]
    public void CreateStrategy_Hybrid_ReturnsHybridStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.Hybrid);

        // Assert
        Assert.IsType<HybridDistillationStrategy<double>>(strategy);
    }

    [Fact]
    public void CreateStrategy_Probabilistic_ReturnsProbabilisticStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.ProbabilisticTransfer);

        // Assert
        Assert.IsType<ProbabilisticDistillationStrategy<double>>(strategy);
    }

    [Fact]
    public void CreateStrategy_Variational_ReturnsVariationalStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.VariationalInformation);

        // Assert
        Assert.IsType<VariationalDistillationStrategy<double>>(strategy);
    }

    [Fact]
    public void CreateStrategy_FactorTransfer_ReturnsFactorTransferStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.FactorTransfer);

        // Assert
        Assert.IsType<FactorTransferDistillationStrategy<double>>(strategy);
    }

    [Fact]
    public void CreateStrategy_NeuronSelectivity_ReturnsNeuronSelectivityStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.NeuronSelectivity);

        // Assert
        Assert.IsType<NeuronSelectivityDistillationStrategy<double>>(strategy);
    }

    [Fact]
    public void CreateStrategy_WithCustomParameters_AppliesParameters()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.ResponseBased,
            temperature: 5.0,
            alpha: 0.8);

        // Assert
        Assert.Equal(5.0, strategy.Temperature);
        Assert.Equal(0.8, strategy.Alpha);
    }

    [Fact]
    public void FluentBuilder_ConfiguresStrategyCorrectly()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>
            .Configure(DistillationStrategyType.ResponseBased)
            .WithTemperature(4.5)
            .WithAlpha(0.4)
            .Build();

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal(4.5, strategy.Temperature);
        Assert.Equal(0.4, strategy.Alpha);
    }

    [Fact]
    public void FluentBuilder_FeatureBased_ConfiguresFeatureWeight()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>
            .Configure(DistillationStrategyType.FeatureBased)
            .WithFeatureWeight(0.8)
            .Build();

        // Assert
        Assert.IsType<FeatureDistillationStrategy<double>>(strategy);
    }

    [Fact]
    public void FluentBuilder_AttentionBased_ConfiguresAttentionWeight()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>
            .Configure(DistillationStrategyType.AttentionBased)
            .WithAttentionWeight(0.7)
            .Build();

        // Assert
        Assert.IsType<AttentionDistillationStrategy<double>>(strategy);
    }

    // [Fact]
    // public void FluentBuilder_Contrastive_ConfiguresLossType()
    // {
    //     // Arrange & Act
    //     var strategy = DistillationStrategyFactory<double>
    //         .Configure(DistillationStrategyType.ContrastiveBased)
    //         .WithContrastiveLossType(ContrastiveLossType.TripletLoss)
    //         .Build();

    //     // Assert
    //     Assert.IsType<ContrastiveDistillationStrategy<double>>(strategy);
    // }
}
