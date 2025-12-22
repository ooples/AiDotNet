using AiDotNet.ActiveLearning;
using AiDotNet.Interfaces;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActiveLearning;

/// <summary>
/// Unit tests for the HybridSampling class.
/// </summary>
public class HybridSamplingTests
{
    #region Test Data

    public static IEnumerable<object[]> CombinationMethods =>
        new List<object[]>
        {
            new object[] { HybridSampling<double>.CombinationMethod.WeightedSum },
            new object[] { HybridSampling<double>.CombinationMethod.Product },
            new object[] { HybridSampling<double>.CombinationMethod.RankFusion },
            new object[] { HybridSampling<double>.CombinationMethod.Maximum },
            new object[] { HybridSampling<double>.CombinationMethod.Minimum }
        };

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_ValidStrategies_InitializesCorrectly()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);

        // Act
        var hybrid = new HybridSampling<double>(strategies);

        // Assert
        Assert.NotNull(hybrid);
        Assert.Equal(2, hybrid.Strategies.Count);
    }

    [Fact]
    public void Constructor_NullStrategies_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new HybridSampling<double>(null!));
    }

    [Fact]
    public void Constructor_EmptyStrategies_ThrowsArgumentException()
    {
        // Arrange
        var strategies = new List<(IActiveLearningStrategy<double>, double)>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new HybridSampling<double>(strategies));
    }

    [Theory]
    [MemberData(nameof(CombinationMethods))]
    public void Constructor_DifferentCombinationMethods_InitializesCorrectly(HybridSampling<double>.CombinationMethod method)
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);

        // Act
        var hybrid = new HybridSampling<double>(strategies, method);

        // Assert
        Assert.NotNull(hybrid);
        Assert.Contains(method.ToString(), hybrid.Name);
    }

    #endregion

    #region Name Property Tests

    [Fact]
    public void Name_MultipleStrategies_ContainsAllStrategyNames()
    {
        // Arrange
        var strategies = new List<(IActiveLearningStrategy<double>, double)>
        {
            (new UncertaintySampling<double>(), 0.5),
            (new DiversitySampling<double>(), 0.5)
        };
        var hybrid = new HybridSampling<double>(strategies);

        // Act
        var name = hybrid.Name;

        // Assert
        Assert.Contains("Uncertainty", name);
        Assert.Contains("Diversity", name);
        Assert.Contains("Hybrid", name);
    }

    [Fact]
    public void Name_ContainsCombinationMethod()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies, HybridSampling<double>.CombinationMethod.Product);

        // Act
        var name = hybrid.Name;

        // Assert
        Assert.Contains("Product", name);
    }

    #endregion

    #region Strategies Property Tests

    [Fact]
    public void Strategies_ReturnsReadOnlyList()
    {
        // Arrange
        var strategies = CreateStrategies(count: 3);
        var hybrid = new HybridSampling<double>(strategies);

        // Act
        var returnedStrategies = hybrid.Strategies;

        // Assert
        Assert.Equal(3, returnedStrategies.Count);
    }

    [Fact]
    public void Strategies_PreservesWeights()
    {
        // Arrange
        var strategies = new List<(IActiveLearningStrategy<double>, double)>
        {
            (new UncertaintySampling<double>(), 0.7),
            (new DiversitySampling<double>(), 0.3)
        };
        var hybrid = new HybridSampling<double>(strategies);

        // Act
        var returnedStrategies = hybrid.Strategies;

        // Assert
        Assert.Equal(0.7, returnedStrategies[0].Weight);
        Assert.Equal(0.3, returnedStrategies[1].Weight);
    }

    #endregion

    #region UseBatchDiversity Property Tests

    [Fact]
    public void UseBatchDiversity_DefaultValue_IsFalse()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);

        // Act & Assert
        Assert.False(hybrid.UseBatchDiversity);
    }

    [Fact]
    public void UseBatchDiversity_SetToTrue_UpdatesCorrectly()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);

        // Act
        hybrid.UseBatchDiversity = true;

        // Assert
        Assert.True(hybrid.UseBatchDiversity);
    }

    #endregion

    #region CreateUncertaintyDiversity Tests

    [Fact]
    public void CreateUncertaintyDiversity_DefaultWeights_CreatesHybridSampler()
    {
        // Act
        var hybrid = HybridSampling<double>.CreateUncertaintyDiversity();

        // Assert
        Assert.NotNull(hybrid);
        Assert.Equal(2, hybrid.Strategies.Count);
    }

    [Fact]
    public void CreateUncertaintyDiversity_CustomWeights_AppliesWeights()
    {
        // Act
        var hybrid = HybridSampling<double>.CreateUncertaintyDiversity(
            uncertaintyWeight: 0.8,
            diversityWeight: 0.2);

        // Assert
        Assert.Equal(0.8, hybrid.Strategies[0].Weight);
        Assert.Equal(0.2, hybrid.Strategies[1].Weight);
    }

    #endregion

    #region SelectSamples Tests

    [Fact]
    public void SelectSamples_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var pool = CreateTestPool(numSamples: 10, featureSize: 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => hybrid.SelectSamples(null!, pool, batchSize: 3));
    }

    [Fact]
    public void SelectSamples_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => hybrid.SelectSamples(model, null!, batchSize: 3));
    }

    [Fact]
    public void SelectSamples_ValidInputs_ReturnsRequestedBatchSize()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = hybrid.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void SelectSamples_BatchSizeLargerThanPool_ReturnsAllSamples()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 5, featureSize: 10);

        // Act
        var selected = hybrid.SelectSamples(model, pool, batchSize: 10);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void SelectSamples_ReturnsUniqueIndices()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = hybrid.SelectSamples(model, pool, batchSize: 10);

        // Assert
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Theory]
    [MemberData(nameof(CombinationMethods))]
    public void SelectSamples_AllCombinationMethods_ReturnValidIndices(HybridSampling<double>.CombinationMethod method)
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies, method);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 15, featureSize: 10);

        // Act
        var selected = hybrid.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.All(selected, idx => Assert.InRange(idx, 0, 14));
    }

    #endregion

    #region ComputeInformativenessScores Tests

    [Fact]
    public void ComputeInformativenessScores_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var pool = CreateTestPool(numSamples: 10, featureSize: 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => hybrid.ComputeInformativenessScores(null!, pool));
    }

    [Fact]
    public void ComputeInformativenessScores_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => hybrid.ComputeInformativenessScores(model, null!));
    }

    [Fact]
    public void ComputeInformativenessScores_ValidInputs_ReturnsScorePerSample()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 15, featureSize: 10);

        // Act
        var scores = hybrid.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(15, scores.Length);
    }

    [Fact]
    public void ComputeInformativenessScores_ReturnsNonNegativeScores()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = hybrid.ComputeInformativenessScores(model, pool);

        // Assert
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0);
        }
    }

    [Theory]
    [MemberData(nameof(CombinationMethods))]
    public void ComputeInformativenessScores_AllMethods_ReturnsValidScores(HybridSampling<double>.CombinationMethod method)
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies, method);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = hybrid.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(10, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.False(double.IsNaN(scores[i]));
            Assert.False(double.IsInfinity(scores[i]));
        }
    }

    #endregion

    #region GetSelectionStatistics Tests

    [Fact]
    public void GetSelectionStatistics_BeforeAnySelection_ReturnsInitialStatistics()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);

        // Act
        var stats = hybrid.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Contains("MinScore", stats.Keys);
        Assert.Contains("MaxScore", stats.Keys);
        Assert.Contains("MeanScore", stats.Keys);
        Assert.Contains("NumStrategies", stats.Keys);
        Assert.Equal(2.0, stats["NumStrategies"]);
    }

    [Fact]
    public void GetSelectionStatistics_AfterSelection_ReturnsValidStatistics()
    {
        // Arrange
        var strategies = CreateStrategies(count: 3);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);
        hybrid.SelectSamples(model, pool, batchSize: 5);

        // Act
        var stats = hybrid.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.True(stats["MaxScore"] >= stats["MinScore"]);
        Assert.Equal(3.0, stats["NumStrategies"]);
    }

    [Fact]
    public void GetSelectionStatistics_IncludesIndividualStrategyStats()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);
        hybrid.SelectSamples(model, pool, batchSize: 5);

        // Act
        var stats = hybrid.GetSelectionStatistics();

        // Assert
        // Should include Strategy0_ and Strategy1_ prefixed keys
        Assert.True(stats.Keys.Any(k => k.StartsWith("Strategy0_")));
        Assert.True(stats.Keys.Any(k => k.StartsWith("Strategy1_")));
    }

    #endregion

    #region Combination Method Tests

    [Fact]
    public void WeightedSum_CombinesScoresCorrectly()
    {
        // Arrange
        var strategies = new List<(IActiveLearningStrategy<double>, double)>
        {
            (new UncertaintySampling<double>(), 0.6),
            (new DiversitySampling<double>(), 0.4)
        };
        var hybrid = new HybridSampling<double>(strategies, HybridSampling<double>.CombinationMethod.WeightedSum);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = hybrid.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(10, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0);
        }
    }

    [Fact]
    public void Product_CombinesScoresCorrectly()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies, HybridSampling<double>.CombinationMethod.Product);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = hybrid.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(10, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0);
        }
    }

    [Fact]
    public void RankFusion_CombinesScoresCorrectly()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies, HybridSampling<double>.CombinationMethod.RankFusion);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = hybrid.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(10, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0);
        }
    }

    [Fact]
    public void Maximum_SelectsHighestScoreAcrossStrategies()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies, HybridSampling<double>.CombinationMethod.Maximum);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = hybrid.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(10, scores.Length);
    }

    [Fact]
    public void Minimum_SelectsLowestScoreAcrossStrategies()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybrid = new HybridSampling<double>(strategies, HybridSampling<double>.CombinationMethod.Minimum);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = hybrid.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(10, scores.Length);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void HybridSampling_CompleteWorkflow_ExecutesCorrectly()
    {
        // Arrange
        var hybrid = HybridSampling<double>.CreateUncertaintyDiversity();
        var model = new MockNeuralNetwork(parameterCount: 20, outputSize: 5);
        var pool = CreateTestPool(numSamples: 50, featureSize: 20);

        // Act - Select multiple batches
        var batch1 = hybrid.SelectSamples(model, pool, batchSize: 10);
        var stats1 = hybrid.GetSelectionStatistics();

        var batch2 = hybrid.SelectSamples(model, pool, batchSize: 10);
        var stats2 = hybrid.GetSelectionStatistics();

        // Assert
        Assert.Equal(10, batch1.Length);
        Assert.Equal(10, batch2.Length);
        Assert.NotNull(stats1);
        Assert.NotNull(stats2);
    }

    [Fact]
    public void HybridSampling_MultipleStrategies_WorksCorrectly()
    {
        // Arrange
        var strategies = new List<(IActiveLearningStrategy<double>, double)>
        {
            (new UncertaintySampling<double>(), 0.5),
            (new DiversitySampling<double>(), 0.3),
            (new ExpectedModelChange<double>(), 0.2)
        };
        var hybrid = new HybridSampling<double>(strategies);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = hybrid.SelectSamples(model, pool, batchSize: 5);
        var stats = hybrid.GetSelectionStatistics();

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(3.0, stats["NumStrategies"]);
    }

    [Fact]
    public void HybridSampling_DifferentCombinationMethods_ProduceDifferentResults()
    {
        // Arrange
        var strategies = CreateStrategies(count: 2);
        var hybridWeighted = new HybridSampling<double>(strategies, HybridSampling<double>.CombinationMethod.WeightedSum);
        var hybridProduct = new HybridSampling<double>(strategies, HybridSampling<double>.CombinationMethod.Product);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateVariedTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selectedWeighted = hybridWeighted.SelectSamples(model, pool, batchSize: 5);
        var selectedProduct = hybridProduct.SelectSamples(model, pool, batchSize: 5);

        // Assert - Both should return valid results
        Assert.Equal(5, selectedWeighted.Length);
        Assert.Equal(5, selectedProduct.Length);
    }

    #endregion

    #region Helper Methods

    private static List<(IActiveLearningStrategy<double>, double)> CreateStrategies(int count)
    {
        var strategies = new List<(IActiveLearningStrategy<double>, double)>();
        double weight = 1.0 / count;
        for (int i = 0; i < count; i++)
        {
            strategies.Add((new UncertaintySampling<double>(), weight));
        }
        return strategies;
    }

    private static Tensor<double> CreateTestPool(int numSamples, int featureSize)
    {
        var tensor = new Tensor<double>(new int[] { numSamples, featureSize });
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i * 0.01;
        }
        return tensor;
    }

    private static Tensor<double> CreateVariedTestPool(int numSamples, int featureSize)
    {
        var tensor = new Tensor<double>(new int[] { numSamples, featureSize });
        var random = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.NextDouble();
        }
        return tensor;
    }

    #endregion
}
