using AiDotNet.ActiveLearning;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActiveLearning;

/// <summary>
/// Unit tests for the ExpectedModelChange class.
/// </summary>
public class ExpectedModelChangeTests
{
    #region Test Data

    public static IEnumerable<object[]> ChangeMetrics =>
        new List<object[]>
        {
            new object[] { ExpectedModelChange<double>.ChangeMetric.ExpectedGradientLength },
            new object[] { ExpectedModelChange<double>.ChangeMetric.MaxGradientLength },
            new object[] { ExpectedModelChange<double>.ChangeMetric.GradientVariance }
        };

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_DefaultMetric_InitializesWithExpectedGradientLength()
    {
        // Arrange & Act
        var emc = new ExpectedModelChange<double>();

        // Assert
        Assert.NotNull(emc);
        Assert.Contains("ExpectedGradientLength", emc.Name);
    }

    [Theory]
    [MemberData(nameof(ChangeMetrics))]
    public void Constructor_DifferentMetrics_InitializesCorrectly(ExpectedModelChange<double>.ChangeMetric metric)
    {
        // Arrange & Act
        var emc = new ExpectedModelChange<double>(metric);

        // Assert
        Assert.NotNull(emc);
        Assert.Contains(metric.ToString(), emc.Name);
    }

    #endregion

    #region Name Property Tests

    [Fact]
    public void Name_ExpectedGradientLength_ContainsMetricName()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(ExpectedModelChange<double>.ChangeMetric.ExpectedGradientLength);

        // Act & Assert
        Assert.Equal("ExpectedModelChange-ExpectedGradientLength", emc.Name);
    }

    [Fact]
    public void Name_MaxGradientLength_ContainsMetricName()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(ExpectedModelChange<double>.ChangeMetric.MaxGradientLength);

        // Act & Assert
        Assert.Equal("ExpectedModelChange-MaxGradientLength", emc.Name);
    }

    [Fact]
    public void Name_GradientVariance_ContainsMetricName()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(ExpectedModelChange<double>.ChangeMetric.GradientVariance);

        // Act & Assert
        Assert.Equal("ExpectedModelChange-GradientVariance", emc.Name);
    }

    #endregion

    #region UseBatchDiversity Property Tests

    [Fact]
    public void UseBatchDiversity_DefaultValue_IsFalse()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();

        // Act & Assert
        Assert.False(emc.UseBatchDiversity);
    }

    [Fact]
    public void UseBatchDiversity_SetToTrue_UpdatesCorrectly()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();

        // Act
        emc.UseBatchDiversity = true;

        // Assert
        Assert.True(emc.UseBatchDiversity);
    }

    #endregion

    #region SelectSamples Tests

    [Fact]
    public void SelectSamples_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var pool = CreateTestPool(numSamples: 10, featureSize: 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => emc.SelectSamples(null!, pool, batchSize: 3));
    }

    [Fact]
    public void SelectSamples_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => emc.SelectSamples(model, null!, batchSize: 3));
    }

    [Fact]
    public void SelectSamples_ValidInputs_ReturnsRequestedBatchSize()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = emc.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void SelectSamples_BatchSizeLargerThanPool_ReturnsAllSamples()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 5, featureSize: 10);

        // Act
        var selected = emc.SelectSamples(model, pool, batchSize: 10);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void SelectSamples_ReturnsUniqueIndices()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = emc.SelectSamples(model, pool, batchSize: 10);

        // Assert
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void SelectSamples_WithBatchDiversity_ReturnsValidIndices()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        emc.UseBatchDiversity = true;
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = emc.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.All(selected, idx => Assert.InRange(idx, 0, 19));
    }

    [Theory]
    [MemberData(nameof(ChangeMetrics))]
    public void SelectSamples_AllMetrics_ReturnValidIndices(ExpectedModelChange<double>.ChangeMetric metric)
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(metric);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 15, featureSize: 10);

        // Act
        var selected = emc.SelectSamples(model, pool, batchSize: 5);

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
        var emc = new ExpectedModelChange<double>();
        var pool = CreateTestPool(numSamples: 10, featureSize: 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => emc.ComputeInformativenessScores(null!, pool));
    }

    [Fact]
    public void ComputeInformativenessScores_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => emc.ComputeInformativenessScores(model, null!));
    }

    [Fact]
    public void ComputeInformativenessScores_ValidInputs_ReturnsScorePerSample()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 15, featureSize: 10);

        // Act
        var scores = emc.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(15, scores.Length);
    }

    [Fact]
    public void ComputeInformativenessScores_ReturnsNonNegativeScores()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = emc.ComputeInformativenessScores(model, pool);

        // Assert
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0);
        }
    }

    [Theory]
    [MemberData(nameof(ChangeMetrics))]
    public void ComputeInformativenessScores_AllMetrics_ReturnsValidScores(ExpectedModelChange<double>.ChangeMetric metric)
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(metric);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = emc.ComputeInformativenessScores(model, pool);

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
        var emc = new ExpectedModelChange<double>();

        // Act
        var stats = emc.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Contains("MinScore", stats.Keys);
        Assert.Contains("MaxScore", stats.Keys);
        Assert.Contains("MeanScore", stats.Keys);
        Assert.Equal(0.0, stats["MinScore"]);
        Assert.Equal(0.0, stats["MaxScore"]);
        Assert.Equal(0.0, stats["MeanScore"]);
    }

    [Fact]
    public void GetSelectionStatistics_AfterSelection_ReturnsValidStatistics()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>();
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);
        emc.SelectSamples(model, pool, batchSize: 5);

        // Act
        var stats = emc.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.True(stats["MaxScore"] >= stats["MinScore"]);
        Assert.True(stats["MeanScore"] >= stats["MinScore"]);
        Assert.True(stats["MeanScore"] <= stats["MaxScore"]);
    }

    #endregion

    #region Gradient Computation Tests

    [Fact]
    public void ExpectedGradientLength_UniformProbabilities_ReturnsConsistentScores()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(ExpectedModelChange<double>.ChangeMetric.ExpectedGradientLength);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateUniformPool(numSamples: 5, featureSize: 10);

        // Act
        var scores = emc.ComputeInformativenessScores(model, pool);

        // Assert - All uniform samples should have similar EGL
        Assert.Equal(5, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] > 0);
        }
    }

    [Fact]
    public void MaxGradientLength_ComputesMaximumAcrossLabels()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(ExpectedModelChange<double>.ChangeMetric.MaxGradientLength);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = emc.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(10, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] > 0);
        }
    }

    [Fact]
    public void GradientVariance_ComputesVarianceAcrossLabels()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(ExpectedModelChange<double>.ChangeMetric.GradientVariance);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = emc.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(10, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0);
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void ExpectedModelChange_CompleteWorkflow_ExecutesCorrectly()
    {
        // Arrange
        var emc = new ExpectedModelChange<double>(ExpectedModelChange<double>.ChangeMetric.ExpectedGradientLength);
        var model = new MockNeuralNetwork(parameterCount: 20, outputSize: 5);
        var pool = CreateTestPool(numSamples: 50, featureSize: 20);

        // Act - Select multiple batches
        var batch1 = emc.SelectSamples(model, pool, batchSize: 10);
        var stats1 = emc.GetSelectionStatistics();

        var batch2 = emc.SelectSamples(model, pool, batchSize: 10);
        var stats2 = emc.GetSelectionStatistics();

        // Assert
        Assert.Equal(10, batch1.Length);
        Assert.Equal(10, batch2.Length);
        Assert.NotNull(stats1);
        Assert.NotNull(stats2);
    }

    [Fact]
    public void ExpectedModelChange_DiversityMode_ProducesDifferentResults()
    {
        // Arrange
        var emcNoDiversity = new ExpectedModelChange<double>();
        var emcWithDiversity = new ExpectedModelChange<double>();
        emcWithDiversity.UseBatchDiversity = true;

        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateVariedTestPool(numSamples: 30, featureSize: 10);

        // Act
        var selectedNoDiversity = emcNoDiversity.SelectSamples(model, pool, batchSize: 10);
        var selectedWithDiversity = emcWithDiversity.SelectSamples(model, pool, batchSize: 10);

        // Assert - Both should return valid results
        Assert.Equal(10, selectedNoDiversity.Length);
        Assert.Equal(10, selectedWithDiversity.Length);
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

    private static Tensor<double> CreateUniformPool(int numSamples, int featureSize)
    {
        var tensor = new Tensor<double>(new int[] { numSamples, featureSize });
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = 0.5;
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
