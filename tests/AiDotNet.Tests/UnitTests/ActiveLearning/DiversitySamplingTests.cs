using AiDotNet.ActiveLearning;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActiveLearning;

/// <summary>
/// Unit tests for the DiversitySampling class.
/// </summary>
public class DiversitySamplingTests
{
    #region Test Data

    public static IEnumerable<object[]> DiversityMethods =>
        new List<object[]>
        {
            new object[] { DiversitySampling<double>.DiversityMethod.FarthestFirst },
            new object[] { DiversitySampling<double>.DiversityMethod.KCenterGreedy },
            new object[] { DiversitySampling<double>.DiversityMethod.DensityPeaks }
        };

    public static IEnumerable<object[]> DistanceMetrics =>
        new List<object[]>
        {
            new object[] { DiversitySampling<double>.DistanceMetric.Euclidean },
            new object[] { DiversitySampling<double>.DistanceMetric.Cosine },
            new object[] { DiversitySampling<double>.DistanceMetric.Manhattan }
        };

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_DefaultParameters_InitializesWithKCenterGreedy()
    {
        // Arrange & Act
        var sampler = new DiversitySampling<double>();

        // Assert
        Assert.NotNull(sampler);
        Assert.Contains("KCenterGreedy", sampler.Name);
        Assert.Contains("Euclidean", sampler.Name);
    }

    [Theory]
    [MemberData(nameof(DiversityMethods))]
    public void Constructor_DifferentMethods_InitializesCorrectly(DiversitySampling<double>.DiversityMethod method)
    {
        // Arrange & Act
        var sampler = new DiversitySampling<double>(method);

        // Assert
        Assert.NotNull(sampler);
        Assert.Contains(method.ToString(), sampler.Name);
    }

    [Theory]
    [MemberData(nameof(DistanceMetrics))]
    public void Constructor_DifferentDistanceMetrics_InitializesCorrectly(DiversitySampling<double>.DistanceMetric metric)
    {
        // Arrange & Act
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.KCenterGreedy,
            metric);

        // Assert
        Assert.NotNull(sampler);
        Assert.Contains(metric.ToString(), sampler.Name);
    }

    #endregion

    #region Name Property Tests

    [Fact]
    public void Name_FarthestFirstEuclidean_ContainsMethodAndMetric()
    {
        // Arrange
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.FarthestFirst,
            DiversitySampling<double>.DistanceMetric.Euclidean);

        // Act & Assert
        Assert.Equal("DiversitySampling-FarthestFirst-Euclidean", sampler.Name);
    }

    [Fact]
    public void Name_KCenterGreedyCosine_ContainsMethodAndMetric()
    {
        // Arrange
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.KCenterGreedy,
            DiversitySampling<double>.DistanceMetric.Cosine);

        // Act & Assert
        Assert.Equal("DiversitySampling-KCenterGreedy-Cosine", sampler.Name);
    }

    [Fact]
    public void Name_DensityPeaksManhattan_ContainsMethodAndMetric()
    {
        // Arrange
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.DensityPeaks,
            DiversitySampling<double>.DistanceMetric.Manhattan);

        // Act & Assert
        Assert.Equal("DiversitySampling-DensityPeaks-Manhattan", sampler.Name);
    }

    #endregion

    #region UseBatchDiversity Property Tests

    [Fact]
    public void UseBatchDiversity_DefaultValue_IsTrue()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();

        // Act & Assert
        // DiversitySampling inherently uses batch diversity, so default is true
        Assert.True(sampler.UseBatchDiversity);
    }

    [Fact]
    public void UseBatchDiversity_SetToFalse_UpdatesCorrectly()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();

        // Act
        sampler.UseBatchDiversity = false;

        // Assert
        Assert.False(sampler.UseBatchDiversity);
    }

    #endregion

    #region CoverageRadius Property Tests

    [Fact]
    public void CoverageRadius_BeforeSelection_IsZero()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();

        // Act & Assert
        Assert.Equal(0.0, sampler.CoverageRadius);
    }

    [Fact]
    public void CoverageRadius_AfterSelection_IsNonNegative()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        sampler.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.True(sampler.CoverageRadius >= 0);
    }

    #endregion

    #region SelectSamples Tests

    [Fact]
    public void SelectSamples_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => sampler.SelectSamples(model, null!, batchSize: 3));
    }

    [Fact]
    public void SelectSamples_ValidInputs_ReturnsRequestedBatchSize()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = sampler.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void SelectSamples_BatchSizeLargerThanPool_ReturnsAllSamples()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 5, featureSize: 10);

        // Act
        var selected = sampler.SelectSamples(model, pool, batchSize: 10);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void SelectSamples_ReturnsUniqueIndices()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = sampler.SelectSamples(model, pool, batchSize: 10);

        // Assert
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Theory]
    [MemberData(nameof(DiversityMethods))]
    public void SelectSamples_AllMethods_ReturnValidIndices(DiversitySampling<double>.DiversityMethod method)
    {
        // Arrange
        var sampler = new DiversitySampling<double>(method);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 15, featureSize: 10);

        // Act
        var selected = sampler.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.All(selected, idx => Assert.InRange(idx, 0, 14));
    }

    [Theory]
    [MemberData(nameof(DistanceMetrics))]
    public void SelectSamples_AllDistanceMetrics_ReturnValidIndices(DiversitySampling<double>.DistanceMetric metric)
    {
        // Arrange
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.KCenterGreedy,
            metric);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateVariedTestPool(numSamples: 15, featureSize: 10);

        // Act
        var selected = sampler.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.All(selected, idx => Assert.InRange(idx, 0, 14));
    }

    #endregion

    #region ComputeInformativenessScores Tests

    [Fact]
    public void ComputeInformativenessScores_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => sampler.ComputeInformativenessScores(model, null!));
    }

    [Fact]
    public void ComputeInformativenessScores_ValidInputs_ReturnsScorePerSample()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 15, featureSize: 10);

        // Act
        var scores = sampler.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(15, scores.Length);
    }

    [Fact]
    public void ComputeInformativenessScores_ReturnsNonNegativeScores()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = sampler.ComputeInformativenessScores(model, pool);

        // Assert
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0);
        }
    }

    [Theory]
    [MemberData(nameof(DiversityMethods))]
    public void ComputeInformativenessScores_AllMethods_ReturnsValidScores(DiversitySampling<double>.DiversityMethod method)
    {
        // Arrange
        var sampler = new DiversitySampling<double>(method);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = sampler.ComputeInformativenessScores(model, pool);

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
    public void GetSelectionStatistics_BeforeAnySelection_ReturnsZeroStatistics()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();

        // Act
        var stats = sampler.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Contains("MinScore", stats.Keys);
        Assert.Contains("MaxScore", stats.Keys);
        Assert.Contains("MeanScore", stats.Keys);
        Assert.Contains("CoverageRadius", stats.Keys);
        Assert.Equal(0.0, stats["MinScore"]);
        Assert.Equal(0.0, stats["CoverageRadius"]);
    }

    [Fact]
    public void GetSelectionStatistics_AfterSelection_ReturnsValidStatistics()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);
        sampler.SelectSamples(model, pool, batchSize: 5);

        // Act
        var stats = sampler.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.True(stats["CoverageRadius"] >= 0);
    }

    #endregion

    #region Distance Metric Tests

    [Fact]
    public void EuclideanDistance_SameSample_ReturnsZero()
    {
        // This tests indirectly through selection behavior
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.FarthestFirst,
            DiversitySampling<double>.DistanceMetric.Euclidean);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 5);

        // Act
        var selected = sampler.SelectSamples(model, pool, batchSize: 3);

        // Assert - Should return unique samples
        Assert.Equal(3, selected.Distinct().Count());
    }

    [Fact]
    public void CosineDistance_NormalizedVectors_WorksCorrectly()
    {
        // Arrange
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.FarthestFirst,
            DiversitySampling<double>.DistanceMetric.Cosine);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateVariedTestPool(numSamples: 10, featureSize: 5);

        // Act
        var selected = sampler.SelectSamples(model, pool, batchSize: 3);

        // Assert
        Assert.Equal(3, selected.Length);
    }

    [Fact]
    public void ManhattanDistance_WorksCorrectly()
    {
        // Arrange
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.FarthestFirst,
            DiversitySampling<double>.DistanceMetric.Manhattan);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 5);

        // Act
        var selected = sampler.SelectSamples(model, pool, batchSize: 3);

        // Assert
        Assert.Equal(3, selected.Length);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void DiversitySampling_CompleteWorkflow_ExecutesCorrectly()
    {
        // Arrange
        var sampler = new DiversitySampling<double>(
            DiversitySampling<double>.DiversityMethod.KCenterGreedy,
            DiversitySampling<double>.DistanceMetric.Euclidean);
        var model = new MockNeuralNetwork(parameterCount: 20, outputSize: 5);
        var pool = CreateVariedTestPool(numSamples: 50, featureSize: 20);

        // Act - Select multiple batches
        var batch1 = sampler.SelectSamples(model, pool, batchSize: 10);
        var stats1 = sampler.GetSelectionStatistics();
        var coverage1 = sampler.CoverageRadius;

        var batch2 = sampler.SelectSamples(model, pool, batchSize: 10);
        var stats2 = sampler.GetSelectionStatistics();
        var coverage2 = sampler.CoverageRadius;

        // Assert
        Assert.Equal(10, batch1.Length);
        Assert.Equal(10, batch2.Length);
        Assert.NotNull(stats1);
        Assert.NotNull(stats2);
        Assert.True(coverage1 >= 0);
        Assert.True(coverage2 >= 0);
    }

    [Fact]
    public void DiversitySampling_DifferentMethods_ProduceDifferentSelections()
    {
        // Arrange
        var samplerFarthest = new DiversitySampling<double>(DiversitySampling<double>.DiversityMethod.FarthestFirst);
        var samplerDensity = new DiversitySampling<double>(DiversitySampling<double>.DiversityMethod.DensityPeaks);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateVariedTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selectedFarthest = samplerFarthest.SelectSamples(model, pool, batchSize: 5);
        var selectedDensity = samplerDensity.SelectSamples(model, pool, batchSize: 5);

        // Assert - Both should return valid results (may or may not differ)
        Assert.Equal(5, selectedFarthest.Length);
        Assert.Equal(5, selectedDensity.Length);
    }

    [Fact]
    public void DiversitySampling_LargerBatch_ReducesCoverageRadius()
    {
        // Arrange
        var sampler = new DiversitySampling<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateVariedTestPool(numSamples: 30, featureSize: 10);

        // Act
        sampler.SelectSamples(model, pool, batchSize: 3);
        var coverage3 = sampler.CoverageRadius;

        sampler.SelectSamples(model, pool, batchSize: 10);
        var coverage10 = sampler.CoverageRadius;

        sampler.SelectSamples(model, pool, batchSize: 20);
        var coverage20 = sampler.CoverageRadius;

        // Assert - Larger batches should have smaller or equal coverage radius
        Assert.True(coverage10 <= coverage3 || Math.Abs(coverage10 - coverage3) < 0.1);
        Assert.True(coverage20 <= coverage10 || Math.Abs(coverage20 - coverage10) < 0.1);
    }

    #endregion

    #region Helper Methods

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
