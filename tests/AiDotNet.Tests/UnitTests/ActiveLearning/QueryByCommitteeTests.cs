using AiDotNet.ActiveLearning;
using AiDotNet.Interfaces;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActiveLearning;

/// <summary>
/// Unit tests for the QueryByCommittee class.
/// </summary>
public class QueryByCommitteeTests
{
    #region Test Data

    public static IEnumerable<object[]> DisagreementMeasures =>
        new List<object[]>
        {
            new object[] { QueryByCommittee<double>.DisagreementMeasure.VoteEntropy },
            new object[] { QueryByCommittee<double>.DisagreementMeasure.KLDivergence },
            new object[] { QueryByCommittee<double>.DisagreementMeasure.PredictionVariance }
        };

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_ValidCommittee_InitializesCorrectly()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);

        // Act
        var qbc = new QueryByCommittee<double>(committee);

        // Assert
        Assert.NotNull(qbc);
        Assert.Equal(3, qbc.Committee.Count);
    }

    [Fact]
    public void Constructor_NullCommittee_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryByCommittee<double>(null!));
    }

    [Fact]
    public void Constructor_SingleMember_ThrowsArgumentException()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new QueryByCommittee<double>(committee));
    }

    [Fact]
    public void Constructor_EmptyCommittee_ThrowsArgumentException()
    {
        // Arrange
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new QueryByCommittee<double>(committee));
    }

    [Theory]
    [MemberData(nameof(DisagreementMeasures))]
    public void Constructor_DifferentMeasures_InitializesCorrectly(QueryByCommittee<double>.DisagreementMeasure measure)
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);

        // Act
        var qbc = new QueryByCommittee<double>(committee, measure);

        // Assert
        Assert.NotNull(qbc);
        Assert.Contains(measure.ToString(), qbc.Name);
    }

    #endregion

    #region Name Property Tests

    [Fact]
    public void Name_VoteEntropy_ContainsMeasureName()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee, QueryByCommittee<double>.DisagreementMeasure.VoteEntropy);

        // Act & Assert
        Assert.Equal("QueryByCommittee-VoteEntropy", qbc.Name);
    }

    [Fact]
    public void Name_KLDivergence_ContainsMeasureName()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee, QueryByCommittee<double>.DisagreementMeasure.KLDivergence);

        // Act & Assert
        Assert.Equal("QueryByCommittee-KLDivergence", qbc.Name);
    }

    [Fact]
    public void Name_PredictionVariance_ContainsMeasureName()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee, QueryByCommittee<double>.DisagreementMeasure.PredictionVariance);

        // Act & Assert
        Assert.Equal("QueryByCommittee-PredictionVariance", qbc.Name);
    }

    #endregion

    #region Committee Property Tests

    [Fact]
    public void Committee_ReturnsReadOnlyList()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 5);
        var qbc = new QueryByCommittee<double>(committee);

        // Act
        var returnedCommittee = qbc.Committee;

        // Assert
        Assert.Equal(5, returnedCommittee.Count);
    }

    #endregion

    #region UseBatchDiversity Property Tests

    [Fact]
    public void UseBatchDiversity_DefaultValue_IsFalse()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);

        // Act & Assert
        Assert.False(qbc.UseBatchDiversity);
    }

    [Fact]
    public void UseBatchDiversity_SetToTrue_UpdatesCorrectly()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);

        // Act
        qbc.UseBatchDiversity = true;

        // Assert
        Assert.True(qbc.UseBatchDiversity);
    }

    #endregion

    #region SelectSamples Tests

    [Fact]
    public void SelectSamples_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => qbc.SelectSamples(model, null!, batchSize: 3));
    }

    [Fact]
    public void SelectSamples_ValidInputs_ReturnsRequestedBatchSize()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = qbc.SelectSamples(model, pool, batchSize: 5);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void SelectSamples_BatchSizeLargerThanPool_ReturnsAllSamples()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 5, featureSize: 10);

        // Act
        var selected = qbc.SelectSamples(model, pool, batchSize: 10);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void SelectSamples_ReturnsUniqueIndices()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 4);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = qbc.SelectSamples(model, pool, batchSize: 10);

        // Assert
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Theory]
    [MemberData(nameof(DisagreementMeasures))]
    public void SelectSamples_AllMeasures_ReturnValidIndices(QueryByCommittee<double>.DisagreementMeasure measure)
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee, measure);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 15, featureSize: 10);

        // Act
        var selected = qbc.SelectSamples(model, pool, batchSize: 5);

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
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => qbc.ComputeInformativenessScores(model, null!));
    }

    [Fact]
    public void ComputeInformativenessScores_ValidInputs_ReturnsScorePerSample()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 15, featureSize: 10);

        // Act
        var scores = qbc.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(15, scores.Length);
    }

    [Fact]
    public void ComputeInformativenessScores_ReturnsNonNegativeScores()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = qbc.ComputeInformativenessScores(model, pool);

        // Assert
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= 0);
        }
    }

    [Theory]
    [MemberData(nameof(DisagreementMeasures))]
    public void ComputeInformativenessScores_AllMeasures_ReturnsValidScores(QueryByCommittee<double>.DisagreementMeasure measure)
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 4);
        var qbc = new QueryByCommittee<double>(committee, measure);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);

        // Act
        var scores = qbc.ComputeInformativenessScores(model, pool);

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
        var committee = CreateCommittee(memberCount: 3);
        var qbc = new QueryByCommittee<double>(committee);

        // Act
        var stats = qbc.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Contains("MinScore", stats.Keys);
        Assert.Contains("MaxScore", stats.Keys);
        Assert.Contains("MeanScore", stats.Keys);
        Assert.Contains("CommitteeSize", stats.Keys);
        Assert.Equal(0.0, stats["MinScore"]);
        Assert.Equal(3.0, stats["CommitteeSize"]);
    }

    [Fact]
    public void GetSelectionStatistics_AfterSelection_ReturnsValidStatistics()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 5);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 10, featureSize: 10);
        qbc.SelectSamples(model, pool, batchSize: 5);

        // Act
        var stats = qbc.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.True(stats["MaxScore"] >= stats["MinScore"]);
        Assert.Equal(5.0, stats["CommitteeSize"]);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void QueryByCommittee_CompleteWorkflow_ExecutesCorrectly()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 5);
        var qbc = new QueryByCommittee<double>(committee, QueryByCommittee<double>.DisagreementMeasure.VoteEntropy);
        var model = new MockNeuralNetwork(parameterCount: 20, outputSize: 5);
        var pool = CreateTestPool(numSamples: 50, featureSize: 20);

        // Act - Select multiple batches
        var batch1 = qbc.SelectSamples(model, pool, batchSize: 10);
        var stats1 = qbc.GetSelectionStatistics();

        var batch2 = qbc.SelectSamples(model, pool, batchSize: 10);
        var stats2 = qbc.GetSelectionStatistics();

        // Assert
        Assert.Equal(10, batch1.Length);
        Assert.Equal(10, batch2.Length);
        Assert.NotNull(stats1);
        Assert.NotNull(stats2);
        Assert.Equal(5.0, stats1["CommitteeSize"]);
    }

    [Fact]
    public void QueryByCommittee_LargeCommittee_WorksCorrectly()
    {
        // Arrange
        var committee = CreateCommittee(memberCount: 10);
        var qbc = new QueryByCommittee<double>(committee);
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var pool = CreateTestPool(numSamples: 20, featureSize: 10);

        // Act
        var selected = qbc.SelectSamples(model, pool, batchSize: 5);
        var stats = qbc.GetSelectionStatistics();

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(10.0, stats["CommitteeSize"]);
    }

    #endregion

    #region Helper Methods

    private static List<IFullModel<double, Tensor<double>, Tensor<double>>> CreateCommittee(int memberCount)
    {
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>();
        for (int i = 0; i < memberCount; i++)
        {
            committee.Add(new MockNeuralNetwork(parameterCount: 10, outputSize: 3));
        }
        return committee;
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

    #endregion
}
