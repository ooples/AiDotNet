using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.LinearAlgebra;
using Xunit;
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the DistillationStrategyFactory class. The strategy IS the parameter:
/// each named strategy is a public factory method; null resolves to the response-based default.
/// </summary>
public class DistillationStrategyFactoryTests
{
    public static TheoryData<Func<IDistillationStrategy<double>>> AllStrategyFactories() => new()
    {
        () => DistillationStrategyFactory<double>.CreateResponseBasedStrategy(),
        () => DistillationStrategyFactory<double>.CreateAttentionBasedStrategy(),
        () => DistillationStrategyFactory<double>.CreateRelationBasedStrategy(),
        () => DistillationStrategyFactory<double>.CreateContrastiveStrategy(),
        () => DistillationStrategyFactory<double>.CreateSimilarityPreservingStrategy(),
        () => DistillationStrategyFactory<double>.CreateProbabilisticStrategy(),
        () => DistillationStrategyFactory<double>.CreateVariationalStrategy(),
        () => DistillationStrategyFactory<double>.CreateFactorTransferStrategy(),
        () => DistillationStrategyFactory<double>.CreateNeuronSelectivityStrategy(),
        () => DistillationStrategyFactory<double>.CreateHybridStrategy(),
    };

    [Theory]
    [MemberData(nameof(AllStrategyFactories))]
    public async Task CreateStrategy_WithAllStrategyTypes_ReturnsValidStrategy(Func<IDistillationStrategy<double>> factory)
    {
        // Act
        var strategy = factory();

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal(3.0, strategy.Temperature); // Default temperature
        Assert.Equal(0.3, strategy.Alpha); // Default alpha
    }

    [Fact(Timeout = 60000)]
    public async Task ResolveStrategy_Null_ReturnsResponseBasedDefault()
    {
        // Arrange & Act: null strategy resolves to the response-based (Hinton) default
        var strategy = DistillationStrategyFactory<double>.ResolveStrategy(null, temperature: 4.0, alpha: 0.5);

        // Assert
        Assert.IsType<DistillationLoss<double>>(strategy);
        Assert.Equal(4.0, strategy.Temperature);
        Assert.Equal(0.5, strategy.Alpha);
    }

    [Fact(Timeout = 60000)]
    public async Task ResolveStrategy_NonNull_ReturnsSuppliedStrategy()
    {
        // Arrange
        var supplied = DistillationStrategyFactory<double>.CreateAttentionBasedStrategy();

        // Act
        var resolved = DistillationStrategyFactory<double>.ResolveStrategy(supplied);

        // Assert
        Assert.Same(supplied, resolved);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_ResponseBased_ReturnsDistillationLoss()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateResponseBasedStrategy(
            temperature: 4.0,
            alpha: 0.5);

        // Assert
        Assert.IsType<DistillationLoss<double>>(strategy);
        Assert.Equal(4.0, strategy.Temperature);
        Assert.Equal(0.5, strategy.Alpha);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_FeatureBased_ThrowsNotSupportedException()
    {
        // Arrange & Act & Assert
        var exception = Assert.Throws<NotSupportedException>(() =>
            DistillationStrategyFactory<double>.CreateFeatureBasedStrategy(featureWeight: 0.7));

        Assert.Contains("FeatureDistillationStrategy", exception.Message);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_AttentionBased_ReturnsAttentionDistillationStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateAttentionBasedStrategy(
            attentionWeight: 0.6);

        // Assert
        Assert.IsType<AttentionDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_Hybrid_ReturnsHybridStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateHybridStrategy();

        // Assert
        Assert.IsType<HybridDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_Probabilistic_ReturnsProbabilisticStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateProbabilisticStrategy();

        // Assert
        Assert.IsType<ProbabilisticDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_Variational_ReturnsVariationalStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateVariationalStrategy();

        // Assert
        Assert.IsType<VariationalDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_FactorTransfer_ReturnsFactorTransferStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateFactorTransferStrategy();

        // Assert
        Assert.IsType<FactorTransferDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_NeuronSelectivity_ReturnsNeuronSelectivityStrategy()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateNeuronSelectivityStrategy();

        // Assert
        Assert.IsType<NeuronSelectivityDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateStrategy_WithCustomParameters_AppliesParameters()
    {
        // Arrange & Act
        var strategy = DistillationStrategyFactory<double>.CreateResponseBasedStrategy(
            temperature: 5.0,
            alpha: 0.8);

        // Assert
        Assert.Equal(5.0, strategy.Temperature);
        Assert.Equal(0.8, strategy.Alpha);
    }
}
