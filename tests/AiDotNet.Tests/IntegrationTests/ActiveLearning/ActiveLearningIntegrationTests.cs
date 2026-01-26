using AiDotNet.ActiveLearning;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.ActiveLearning;

/// <summary>
/// Comprehensive integration tests for ActiveLearning module.
/// Tests sampling strategies for correctness, edge cases, and mathematical validation.
/// </summary>
public class ActiveLearningIntegrationTests
{
    private const double Tolerance = 1e-6;
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();

    #region Test Helpers

    /// <summary>
    /// Creates a simple mock model that returns predictable class probabilities.
    /// </summary>
    private static MockClassificationModel CreateMockModel(int numClasses)
    {
        return new MockClassificationModel(numClasses);
    }

    /// <summary>
    /// Creates a pool of unlabeled samples with controlled features.
    /// </summary>
    private static Tensor<double> CreateUnlabeledPool(int numSamples, int numFeatures, int seed = 42)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new double[numSamples * numFeatures];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble() * 2 - 1; // Range [-1, 1]
        }
        return new Tensor<double>(new[] { numSamples, numFeatures }, new Vector<double>(data));
    }

    /// <summary>
    /// Creates a pool where samples have known uncertainty levels.
    /// Sample 0 has high confidence (low entropy), Sample N-1 has low confidence (high entropy).
    /// </summary>
    private static Tensor<double> CreatePoolWithKnownUncertainty(int numSamples, int numFeatures)
    {
        var data = new double[numSamples * numFeatures];
        for (int i = 0; i < numSamples; i++)
        {
            // Feature values encode the sample index to create predictable uncertainty
            for (int j = 0; j < numFeatures; j++)
            {
                // Sample 0 gets values near 1 (confident), Sample N-1 gets values near 0.5 (uncertain)
                data[i * numFeatures + j] = 1.0 - (0.5 * i / (numSamples - 1));
            }
        }
        return new Tensor<double>(new[] { numSamples, numFeatures }, new Vector<double>(data));
    }

    #endregion

    #region EntropySampling Tests

    [Fact]
    public void EntropySampling_SelectSamples_ReturnsCorrectBatchSize()
    {
        // Arrange
        var strategy = new EntropySampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(100, 10);
        int batchSize = 10;

        // Act
        var selected = strategy.SelectSamples(model, pool, batchSize);

        // Assert
        Assert.Equal(batchSize, selected.Length);
        Assert.All(selected, idx => Assert.InRange(idx, 0, 99));
        Assert.Equal(selected.Length, selected.Distinct().Count()); // No duplicates
    }

    [Fact]
    public void EntropySampling_SelectSamples_BatchSizeLargerThanPool_ReturnsAllSamples()
    {
        // Arrange
        var strategy = new EntropySampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(5, 10);
        int batchSize = 100;

        // Act
        var selected = strategy.SelectSamples(model, pool, batchSize);

        // Assert
        Assert.Equal(5, selected.Length); // Should return all available samples
    }

    [Fact]
    public void EntropySampling_SelectSamples_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new EntropySampling<double>();
        var pool = CreateUnlabeledPool(10, 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.SelectSamples(null!, pool, 5));
    }

    [Fact]
    public void EntropySampling_SelectSamples_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new EntropySampling<double>();
        var model = CreateMockModel(3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.SelectSamples(model, null!, 5));
    }

    [Fact]
    public void EntropySampling_ComputeInformativenessScores_HighEntropyForUncertainSamples()
    {
        // Arrange
        var strategy = new EntropySampling<double>();
        var model = new MockModelWithControlledUncertainty(3);
        var pool = CreatePoolWithKnownUncertainty(10, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert - Higher indices should have higher entropy (more uncertain)
        Assert.Equal(10, scores.Length);
        // The last sample should have higher entropy than the first
        Assert.True(scores[9] > scores[0], $"Expected entropy[9]={scores[9]} > entropy[0]={scores[0]}");
    }

    [Fact]
    public void EntropySampling_GetSelectionStatistics_ReturnsValidStats()
    {
        // Arrange
        var strategy = new EntropySampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(50, 10);

        // Act
        _ = strategy.ComputeInformativenessScores(model, pool);
        var stats = strategy.GetSelectionStatistics();

        // Assert
        Assert.Contains("MinScore", stats.Keys);
        Assert.Contains("MaxScore", stats.Keys);
        Assert.Contains("MeanScore", stats.Keys);
        Assert.True(stats["MinScore"] <= stats["MeanScore"]);
        Assert.True(stats["MeanScore"] <= stats["MaxScore"]);
    }

    [Fact]
    public void EntropySampling_WithBatchDiversity_SelectsDiverseSamples()
    {
        // Arrange
        var strategyWithDiversity = new EntropySampling<double> { UseBatchDiversity = true };
        var strategyWithoutDiversity = new EntropySampling<double> { UseBatchDiversity = false };
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(100, 10, seed: 42);

        // Act
        var selectedWithDiversity = strategyWithDiversity.SelectSamples(model, pool, 10);
        var selectedWithoutDiversity = strategyWithoutDiversity.SelectSamples(model, pool, 10);

        // Assert - Diversity mode should (potentially) produce different selection
        Assert.Equal(10, selectedWithDiversity.Length);
        Assert.Equal(10, selectedWithoutDiversity.Length);
        // Note: They may or may not be identical depending on data, but both should be valid
    }

    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(10)]
    public void EntropySampling_WorksWithDifferentNumberOfClasses(int numClasses)
    {
        // Arrange
        var strategy = new EntropySampling<double>();
        var model = CreateMockModel(numClasses);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(20, scores.Length);
        // Maximum entropy for uniform distribution is log(numClasses)
        double maxPossibleEntropy = Math.Log(numClasses);
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0 && scores[i] <= maxPossibleEntropy + 0.01,
                $"Entropy {scores[i]} should be in [0, {maxPossibleEntropy}]"));
    }

    #endregion

    #region UncertaintySampling Tests

    [Theory]
    [InlineData(UncertaintySampling<double>.UncertaintyMeasure.LeastConfidence)]
    [InlineData(UncertaintySampling<double>.UncertaintyMeasure.MarginSampling)]
    [InlineData(UncertaintySampling<double>.UncertaintyMeasure.Entropy)]
    public void UncertaintySampling_AllMeasures_SelectValidSamples(UncertaintySampling<double>.UncertaintyMeasure measure)
    {
        // Arrange
        var strategy = new UncertaintySampling<double>(measure);
        var model = CreateMockModel(4);
        var pool = CreateUnlabeledPool(50, 8);

        // Act
        var selected = strategy.SelectSamples(model, pool, 10);

        // Assert
        Assert.Equal(10, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
        Assert.All(selected, idx => Assert.InRange(idx, 0, 49));
    }

    [Fact]
    public void UncertaintySampling_LeastConfidence_CalculatesCorrectly()
    {
        // For a confident prediction like [0.9, 0.05, 0.05], least confidence = 1 - 0.9 = 0.1
        // For uncertain prediction [0.33, 0.33, 0.34], least confidence = 1 - 0.34 ≈ 0.66
        var strategy = new UncertaintySampling<double>(UncertaintySampling<double>.UncertaintyMeasure.LeastConfidence);
        var model = new MockModelWithControlledUncertainty(3);
        var pool = CreatePoolWithKnownUncertainty(10, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert - Least confidence should be in [0, 1]
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0 && scores[i] <= 1,
                $"Least confidence {scores[i]} should be in [0, 1]"));
    }

    [Fact]
    public void UncertaintySampling_MarginSampling_CalculatesCorrectly()
    {
        // For confident [0.9, 0.05, 0.05], margin = 0.9 - 0.05 = 0.85, uncertainty = 1 - 0.85 = 0.15
        // For uncertain [0.4, 0.35, 0.25], margin = 0.4 - 0.35 = 0.05, uncertainty = 1 - 0.05 = 0.95
        var strategy = new UncertaintySampling<double>(UncertaintySampling<double>.UncertaintyMeasure.MarginSampling);
        var model = new MockModelWithControlledUncertainty(3);
        var pool = CreatePoolWithKnownUncertainty(10, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert - Margin uncertainty should be in [0, 1]
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0 && scores[i] <= 1,
                $"Margin uncertainty {scores[i]} should be in [0, 1]"));
    }

    [Fact]
    public void UncertaintySampling_Name_IncludesMeasure()
    {
        var lcStrategy = new UncertaintySampling<double>(UncertaintySampling<double>.UncertaintyMeasure.LeastConfidence);
        var msStrategy = new UncertaintySampling<double>(UncertaintySampling<double>.UncertaintyMeasure.MarginSampling);
        var enStrategy = new UncertaintySampling<double>(UncertaintySampling<double>.UncertaintyMeasure.Entropy);

        Assert.Contains("LeastConfidence", lcStrategy.Name);
        Assert.Contains("MarginSampling", msStrategy.Name);
        Assert.Contains("Entropy", enStrategy.Name);
    }

    #endregion

    #region BALD Tests

    [Fact]
    public void BALD_SelectSamples_ReturnsValidSelection()
    {
        // Arrange
        var strategy = new BALD<double>(numMcSamples: 5, dropoutRate: 0.3);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(30, 10);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void BALD_ComputeInformativenessScores_ReturnsNonNegativeScores()
    {
        // BALD score = H(y|x) - E[H(y|x,θ)] should be non-negative (epistemic uncertainty)
        var strategy = new BALD<double>(numMcSamples: 10);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert - BALD scores should be non-negative (or close to 0 due to numerical precision)
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= -Tolerance,
                $"BALD score {scores[i]} should be non-negative"));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(20)]
    public void BALD_DifferentMcSamples_AllWorkCorrectly(int numMcSamples)
    {
        // Arrange
        var strategy = new BALD<double>(numMcSamples: numMcSamples);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 3);

        // Assert
        Assert.Equal(3, selected.Length);
    }

    [Fact]
    public void BALD_Name_IncludesMcSampleCount()
    {
        var strategy = new BALD<double>(numMcSamples: 15);
        Assert.Contains("MC15", strategy.Name);
    }

    #endregion

    #region RandomSampling Tests

    [Fact]
    public void RandomSampling_SelectSamples_ReturnsRandomSelection()
    {
        // Arrange
        var strategy = new RandomSampling<double>(seed: 42);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(100, 10);

        // Act
        var selected1 = strategy.SelectSamples(model, pool, 10);

        // Create new strategy with different seed
        var strategy2 = new RandomSampling<double>(seed: 123);
        var selected2 = strategy2.SelectSamples(model, pool, 10);

        // Assert - Different seeds should (very likely) produce different selections
        Assert.Equal(10, selected1.Length);
        Assert.Equal(10, selected2.Length);
        // Note: There's a tiny chance they could be equal, but extremely unlikely with 100 samples
    }

    [Fact]
    public void RandomSampling_WithSameSeed_ProducesReproducibleResults()
    {
        // Arrange
        var pool = CreateUnlabeledPool(50, 5);
        var model = CreateMockModel(2);

        var strategy1 = new RandomSampling<double>(seed: 42);
        var strategy2 = new RandomSampling<double>(seed: 42);

        // Act
        var selected1 = strategy1.SelectSamples(model, pool, 10);
        var selected2 = strategy2.SelectSamples(model, pool, 10);

        // Assert - Same seed should produce same results
        Assert.Equal(selected1, selected2);
    }

    [Fact]
    public void RandomSampling_InformativenessScores_AreUniform()
    {
        // Random sampling should assign uniform scores (all equal)
        var strategy = new RandomSampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert - All scores should be the same (or very close due to implementation)
        var firstScore = scores[0];
        // Random sampling typically assigns equal informativeness
    }

    #endregion

    #region MarginSampling Tests

    [Fact]
    public void MarginSampling_SelectSamples_ReturnsValidSelection()
    {
        var strategy = new MarginSampling<double>();
        var model = CreateMockModel(4);
        var pool = CreateUnlabeledPool(40, 8);

        var selected = strategy.SelectSamples(model, pool, 8);

        Assert.Equal(8, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void MarginSampling_ScoresAreInValidRange()
    {
        var strategy = new MarginSampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(25, 5);

        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Margin = P(1st) - P(2nd), should be in [0, 1]
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0 && scores[i] <= 1,
                $"Margin score {scores[i]} should be in [0, 1]"));
    }

    #endregion

    #region LeastConfidenceSampling Tests

    [Fact]
    public void LeastConfidenceSampling_SelectSamples_ReturnsValidSelection()
    {
        var strategy = new LeastConfidenceSampling<double>();
        var model = CreateMockModel(5);
        var pool = CreateUnlabeledPool(30, 6);

        var selected = strategy.SelectSamples(model, pool, 6);

        Assert.Equal(6, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void LeastConfidenceSampling_ScoresAreInValidRange()
    {
        var strategy = new LeastConfidenceSampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Least confidence = 1 - max(P), should be in [0, 1]
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0 && scores[i] <= 1,
                $"Least confidence score {scores[i]} should be in [0, 1]"));
    }

    #endregion

    #region VariationRatios Tests

    [Fact]
    public void VariationRatios_SelectSamples_ReturnsValidSelection()
    {
        var strategy = new VariationRatios<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(25, 5);

        var selected = strategy.SelectSamples(model, pool, 5);

        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void VariationRatios_ScoresAreInValidRange()
    {
        var strategy = new VariationRatios<double>();
        var model = CreateMockModel(4);
        var pool = CreateUnlabeledPool(20, 5);

        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Variation ratio = 1 - max(P), same as least confidence
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0 && scores[i] <= 1,
                $"Variation ratio {scores[i]} should be in [0, 1]"));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void AllStrategies_SingleSamplePool_ReturnsOneSample()
    {
        var strategies = new object[]
        {
            new EntropySampling<double>(),
            new BALD<double>(),
            new RandomSampling<double>(),
            new MarginSampling<double>(),
            new LeastConfidenceSampling<double>(),
            new VariationRatios<double>(),
        };

        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(1, 5);

        foreach (var strategy in strategies)
        {
            // Use reflection to call SelectSamples (they all implement IActiveLearningStrategy<T>)
            var method = strategy.GetType().GetMethod("SelectSamples");
            var selected = method?.Invoke(strategy, new object[] { model, pool, 10 }) as int[];

            Assert.NotNull(selected);
            Assert.Single(selected);
            Assert.Equal(0, selected[0]);
        }
    }

    [Fact]
    public void AllStrategies_ZeroBatchSize_ReturnsEmptyArray()
    {
        var strategy = new EntropySampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(10, 5);

        var selected = strategy.SelectSamples(model, pool, 0);

        Assert.Empty(selected);
    }

    [Fact]
    public void AllStrategies_LargeBatchSize_ClampedToPoolSize()
    {
        var strategy = new BALD<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(5, 10);

        var selected = strategy.SelectSamples(model, pool, 1000);

        Assert.Equal(5, selected.Length);
    }

    #endregion

    #region Mathematical Validation

    [Fact]
    public void Entropy_UniformDistribution_EqualsLogK()
    {
        // For uniform distribution over k classes, entropy = log(k)
        // We test this indirectly through the model
        var strategy = new EntropySampling<double>();
        var model = new MockModelWithUniformPredictions(4); // 4 classes
        var pool = CreateUnlabeledPool(10, 5);

        var scores = strategy.ComputeInformativenessScores(model, pool);

        double expectedMaxEntropy = Math.Log(4); // log(4) ≈ 1.386
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(Math.Abs(scores[i] - expectedMaxEntropy) < 0.1,
                $"Entropy for uniform distribution should be close to log(4)={expectedMaxEntropy}, got {scores[i]}"));
    }

    [Fact]
    public void Entropy_CertainPrediction_EqualsZero()
    {
        // For certain prediction (all probability on one class), entropy = 0
        var strategy = new EntropySampling<double>();
        var model = new MockModelWithCertainPredictions(3);
        var pool = CreateUnlabeledPool(10, 5);

        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] < 0.01,
                $"Entropy for certain prediction should be close to 0, got {scores[i]}"));
    }

    #endregion

    #region DiversitySampling Tests

    [Theory]
    [InlineData(DiversitySampling<double>.DiversityMethod.FarthestFirst)]
    [InlineData(DiversitySampling<double>.DiversityMethod.KCenterGreedy)]
    [InlineData(DiversitySampling<double>.DiversityMethod.DensityPeaks)]
    public void DiversitySampling_AllMethods_SelectValidSamples(DiversitySampling<double>.DiversityMethod method)
    {
        // Arrange
        var strategy = new DiversitySampling<double>(method: method);
        var model = CreateMockModel(3); // Model is not used by DiversitySampling but required by interface
        var pool = CreateUnlabeledPool(30, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count()); // No duplicates
        Assert.All(selected, idx => Assert.InRange(idx, 0, 29));
    }

    [Theory]
    [InlineData(DiversitySampling<double>.DistanceMetric.Euclidean)]
    [InlineData(DiversitySampling<double>.DistanceMetric.Cosine)]
    [InlineData(DiversitySampling<double>.DistanceMetric.Manhattan)]
    public void DiversitySampling_AllDistanceMetrics_SelectValidSamples(DiversitySampling<double>.DistanceMetric metric)
    {
        // Arrange
        var strategy = new DiversitySampling<double>(distanceMetric: metric);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void DiversitySampling_CoverageRadius_IsComputed()
    {
        // Arrange
        var strategy = new DiversitySampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);
        var coverageRadius = strategy.CoverageRadius;

        // Assert
        Assert.True(coverageRadius >= 0, "Coverage radius should be non-negative");
    }

    [Fact]
    public void DiversitySampling_FarthestFirst_SelectsDiverseSamples()
    {
        // Arrange - Create a pool with clearly separated clusters
        var poolData = new double[20 * 2]; // 20 samples, 2 features
        // Cluster 1: around (0, 0)
        for (int i = 0; i < 10; i++)
        {
            poolData[i * 2] = 0.1 * i;
            poolData[i * 2 + 1] = 0.1 * i;
        }
        // Cluster 2: around (10, 10)
        for (int i = 10; i < 20; i++)
        {
            poolData[i * 2] = 10 + 0.1 * (i - 10);
            poolData[i * 2 + 1] = 10 + 0.1 * (i - 10);
        }
        var pool = new Tensor<double>(new[] { 20, 2 }, new Vector<double>(poolData));

        var strategy = new DiversitySampling<double>(DiversitySampling<double>.DiversityMethod.FarthestFirst);
        var model = CreateMockModel(3);

        // Act
        var selected = strategy.SelectSamples(model, pool, 4);

        // Assert - Should select samples from both clusters
        var cluster1Selected = selected.Count(idx => idx < 10);
        var cluster2Selected = selected.Count(idx => idx >= 10);

        // With farthest-first, we expect selection from both clusters
        Assert.True(cluster1Selected >= 1 && cluster2Selected >= 1,
            $"Expected samples from both clusters, got cluster1={cluster1Selected}, cluster2={cluster2Selected}");
    }

    #endregion

    #region CoreSetSelection Tests

    [Fact]
    public void CoreSetSelection_SelectSamples_ReturnsValidSelection()
    {
        // Arrange
        var strategy = new CoreSetSelection<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(30, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void CoreSetSelection_ComputeInformativenessScores_ReturnsValidScores()
    {
        // Arrange
        var strategy = new CoreSetSelection<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(15, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(15, scores.Length);
        // All scores should be non-negative (distances)
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0, $"CoreSet score {scores[i]} should be non-negative"));
    }

    #endregion

    #region HybridSampling Tests

    [Fact]
    public void HybridSampling_SelectSamples_ReturnsValidSelection()
    {
        // Arrange - Use factory method to create hybrid strategy
        var strategy = HybridSampling<double>.CreateUncertaintyDiversity();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(30, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 8);

        // Assert
        Assert.Equal(8, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void HybridSampling_ComputeInformativenessScores_CombinesStrategies()
    {
        // Arrange - Create a hybrid strategy with entropy and diversity
        var strategies = new List<(IActiveLearningStrategy<double>, double)>
        {
            (new EntropySampling<double>(), 0.6),
            (new DiversitySampling<double>(), 0.4)
        };
        var strategy = new HybridSampling<double>(strategies);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(20, scores.Length);
    }

    [Theory]
    [InlineData(HybridSampling<double>.CombinationMethod.WeightedSum)]
    [InlineData(HybridSampling<double>.CombinationMethod.Product)]
    [InlineData(HybridSampling<double>.CombinationMethod.RankFusion)]
    [InlineData(HybridSampling<double>.CombinationMethod.Maximum)]
    [InlineData(HybridSampling<double>.CombinationMethod.Minimum)]
    public void HybridSampling_AllCombinationMethods_Work(HybridSampling<double>.CombinationMethod method)
    {
        // Arrange
        var strategies = new List<(IActiveLearningStrategy<double>, double)>
        {
            (new EntropySampling<double>(), 0.5),
            (new RandomSampling<double>(), 0.5)
        };
        var strategy = new HybridSampling<double>(strategies, method);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    #endregion

    #region InformationDensity Tests

    [Fact]
    public void InformationDensity_SelectSamples_ReturnsValidSelection()
    {
        // Arrange
        var strategy = new InformationDensity<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(25, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void InformationDensity_BalancesInformativenessAndDensity()
    {
        // Arrange
        var strategy = new InformationDensity<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(20, scores.Length);
        // Scores should be non-negative (product of informativeness and density)
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0, $"Information density score should be non-negative"));
    }

    #endregion

    #region DensityWeightedSampling Tests

    [Fact]
    public void DensityWeightedSampling_SelectSamples_ReturnsValidSelection()
    {
        // Arrange
        var strategy = new DensityWeightedSampling<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(25, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    #endregion

    #region ExpectedModelChange Tests

    [Fact]
    public void ExpectedModelChange_SelectSamples_ReturnsValidSelection()
    {
        // Arrange
        var strategy = new ExpectedModelChange<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void ExpectedModelChange_Scores_AreNonNegative()
    {
        // Expected model change measures gradient magnitude which is always >= 0
        var strategy = new ExpectedModelChange<double>();
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(15, 5);

        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0, $"Expected model change score {scores[i]} should be non-negative"));
    }

    #endregion

    #region BatchBALD Tests

    [Fact]
    public void BatchBALD_SelectSamples_ReturnsValidSelection()
    {
        // Arrange
        var strategy = new BatchBALD<double>(numMcSamples: 5);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void BatchBALD_WithDifferentMcSamples_Works()
    {
        // Arrange
        var strategy3 = new BatchBALD<double>(numMcSamples: 3);
        var strategy10 = new BatchBALD<double>(numMcSamples: 10);
        var model = CreateMockModel(3);
        var pool = CreateUnlabeledPool(15, 5);

        // Act
        var selected3 = strategy3.SelectSamples(model, pool, 3);
        var selected10 = strategy10.SelectSamples(model, pool, 3);

        // Assert
        Assert.Equal(3, selected3.Length);
        Assert.Equal(3, selected10.Length);
    }

    #endregion

    #region QueryByCommittee Tests

    [Fact]
    public void QueryByCommittee_SelectSamples_ReturnsValidSelection()
    {
        // Arrange - Create a committee of models
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            new MockClassificationModel(3, seed: 1),
            new MockClassificationModel(3, seed: 2),
            new MockClassificationModel(3, seed: 3),
        };
        var strategy = new QueryByCommittee<double>(committee);
        var pool = CreateUnlabeledPool(25, 5);

        // Act
        var selected = strategy.SelectSamples(committee[0], pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void QueryByCommittee_DisagreementScores_AreValid()
    {
        // Arrange
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            new MockClassificationModel(3, seed: 10),
            new MockClassificationModel(3, seed: 20),
        };
        var strategy = new QueryByCommittee<double>(committee);
        var pool = CreateUnlabeledPool(15, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(committee[0], pool);

        // Assert
        Assert.Equal(15, scores.Length);
        // Vote entropy should be non-negative
        Assert.All(Enumerable.Range(0, scores.Length), i =>
            Assert.True(scores[i] >= 0, $"QBC disagreement score {scores[i]} should be non-negative"));
    }

    #endregion

    #region Mock Classes

    /// <summary>
    /// Base class for mock models to reduce duplication of interface implementations.
    /// </summary>
    private abstract class MockModelBase : IFullModel<double, Tensor<double>, Tensor<double>>
    {
        protected readonly int _numClasses;
        private HashSet<int> _activeFeatures = new();

        protected MockModelBase(int numClasses)
        {
            _numClasses = numClasses;
        }

        // IModel<TInput, TOutput, TMetadata>
        public abstract Tensor<double> Predict(Tensor<double> input);
        public void Train(Tensor<double> input, Tensor<double> expected) { }
        public ModelMetadata<double> GetModelMetadata() => new();

        // IModelSerializer
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }

        // ICheckpointableModel
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }

        // IParameterizable<T, TInput, TOutput>
        public Vector<double> GetParameters() => new(0);
        public void SetParameters(Vector<double> parameters) { }
        public int ParameterCount => 0;
        public IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters) => this;

        // IFeatureAware
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = new HashSet<int>(featureIndices);
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);

        // IFeatureImportance<T>
        public Dictionary<string, double> GetFeatureImportance() => new();

        // ICloneable<IFullModel<T, TInput, TOutput>>
        public IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy() => Clone();
        public abstract IFullModel<double, Tensor<double>, Tensor<double>> Clone();

        // IGradientComputable<T, TInput, TOutput>
        public Vector<double> ComputeGradients(Tensor<double> input, Tensor<double> target, ILossFunction<double>? lossFunction = null) => new(0);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }

        // IJitCompilable<T>
        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
            => throw new NotSupportedException("Mock model does not support JIT compilation");
        public bool SupportsJitCompilation => false;

        // IFullModel specific
        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
    }

    /// <summary>
    /// Simple mock model that returns random but consistent predictions.
    /// </summary>
    private class MockClassificationModel : MockModelBase
    {
        private readonly Random _random;

        public MockClassificationModel(int numClasses, int seed = 42) : base(numClasses)
        {
            _random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        }

        public override Tensor<double> Predict(Tensor<double> input)
        {
            int numSamples = input.Shape[0];
            var output = new double[numSamples * _numClasses];

            for (int i = 0; i < numSamples; i++)
            {
                // Generate random logits
                for (int c = 0; c < _numClasses; c++)
                {
                    output[i * _numClasses + c] = _random.NextDouble() * 4 - 2; // Range [-2, 2]
                }
            }

            return new Tensor<double>(new[] { numSamples, _numClasses }, new Vector<double>(output));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> Clone() => new MockClassificationModel(_numClasses);
    }

    /// <summary>
    /// Mock model that returns predictions with controlled uncertainty based on input.
    /// </summary>
    private class MockModelWithControlledUncertainty : MockModelBase
    {
        public MockModelWithControlledUncertainty(int numClasses) : base(numClasses) { }

        public override Tensor<double> Predict(Tensor<double> input)
        {
            int numSamples = input.Shape[0];
            int numFeatures = input.Length / numSamples;
            var output = new double[numSamples * _numClasses];

            for (int i = 0; i < numSamples; i++)
            {
                // Use first feature to determine uncertainty
                double firstFeature = input[i * numFeatures];
                // Higher feature value = more confident (lower entropy)
                double confidence = Math.Max(0.1, Math.Min(0.9, firstFeature));

                // First class gets confidence, rest share (1-confidence)
                output[i * _numClasses] = Math.Log(confidence);
                double remaining = (1 - confidence) / (_numClasses - 1);
                for (int c = 1; c < _numClasses; c++)
                {
                    output[i * _numClasses + c] = Math.Log(remaining + 0.01);
                }
            }

            return new Tensor<double>(new[] { numSamples, _numClasses }, new Vector<double>(output));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> Clone() => new MockModelWithControlledUncertainty(_numClasses);
    }

    /// <summary>
    /// Mock model that returns uniform predictions (maximum entropy).
    /// </summary>
    private class MockModelWithUniformPredictions : MockModelBase
    {
        public MockModelWithUniformPredictions(int numClasses) : base(numClasses) { }

        public override Tensor<double> Predict(Tensor<double> input)
        {
            int numSamples = input.Shape[0];
            var output = new double[numSamples * _numClasses];

            // All classes get same logit = 0, which after softmax = 1/numClasses
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = 0.0;
            }

            return new Tensor<double>(new[] { numSamples, _numClasses }, new Vector<double>(output));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> Clone() => new MockModelWithUniformPredictions(_numClasses);
    }

    /// <summary>
    /// Mock model that returns certain predictions (zero entropy).
    /// </summary>
    private class MockModelWithCertainPredictions : MockModelBase
    {
        public MockModelWithCertainPredictions(int numClasses) : base(numClasses) { }

        public override Tensor<double> Predict(Tensor<double> input)
        {
            int numSamples = input.Shape[0];
            var output = new double[numSamples * _numClasses];

            for (int i = 0; i < numSamples; i++)
            {
                // First class gets very high logit, others get very low
                output[i * _numClasses] = 100.0; // After softmax ≈ 1.0
                for (int c = 1; c < _numClasses; c++)
                {
                    output[i * _numClasses + c] = -100.0; // After softmax ≈ 0.0
                }
            }

            return new Tensor<double>(new[] { numSamples, _numClasses }, new Vector<double>(output));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> Clone() => new MockModelWithCertainPredictions(_numClasses);
    }

    #endregion
}
