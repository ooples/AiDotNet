using AiDotNet.ActiveLearning;
using AiDotNet.Interfaces;
using Moq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActiveLearning;

/// <summary>
/// Unit tests for the Query-by-Committee strategy.
/// </summary>
public class QueryByCommitteeTests
{
    private static List<IFullModel<double, Tensor<double>, Tensor<double>>> CreateMockCommittee(int size)
    {
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>();
        for (int i = 0; i < size; i++)
        {
            var mockModel = new Mock<IFullModel<double, Tensor<double>, Tensor<double>>>();
            committee.Add(mockModel.Object);
        }
        return committee;
    }

    [Fact]
    public void Constructor_ValidCommitteeSize_InitializesSuccessfully()
    {
        // Arrange
        var committee = CreateMockCommittee(5);

        // Act
        var strategy = new QueryByCommittee<double>(committee);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("QueryByCommittee-VoteEntropy", strategy.Name);
        Assert.Equal(5, strategy.Committee.Count);
    }

    [Fact]
    public void Constructor_CommitteeSizeLessThan2_ThrowsArgumentException()
    {
        // Arrange
        var committee = CreateMockCommittee(1);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new QueryByCommittee<double>(committee));

        Assert.Contains("at least 2", exception.Message);
    }

    [Fact]
    public void Constructor_NullCommittee_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new QueryByCommittee<double>(null!));
    }

    [Fact]
    public void Constructor_WithDifferentMeasures_InitializesCorrectly()
    {
        // Arrange
        var committee = CreateMockCommittee(3);

        // Act
        var voteEntropy = new QueryByCommittee<double>(
            committee,
            QueryByCommittee<double>.DisagreementMeasure.VoteEntropy);

        var klDivergence = new QueryByCommittee<double>(
            committee,
            QueryByCommittee<double>.DisagreementMeasure.KLDivergence);

        var predictionVariance = new QueryByCommittee<double>(
            committee,
            QueryByCommittee<double>.DisagreementMeasure.PredictionVariance);

        // Assert
        Assert.Equal("QueryByCommittee-VoteEntropy", voteEntropy.Name);
        Assert.Equal("QueryByCommittee-KLDivergence", klDivergence.Name);
        Assert.Equal("QueryByCommittee-PredictionVariance", predictionVariance.Name);
    }

    [Fact]
    public void UseBatchDiversity_DefaultsFalse()
    {
        // Arrange
        var committee = CreateMockCommittee(3);

        // Act
        var strategy = new QueryByCommittee<double>(committee);

        // Assert
        Assert.False(strategy.UseBatchDiversity);
    }

    [Fact]
    public void UseBatchDiversity_CanBeSet()
    {
        // Arrange
        var committee = CreateMockCommittee(3);
        var strategy = new QueryByCommittee<double>(committee);

        // Act
        strategy.UseBatchDiversity = true;

        // Assert
        Assert.True(strategy.UseBatchDiversity);
    }

    [Fact]
    public void GetSelectionStatistics_ContainsCommitteeSize()
    {
        // Arrange
        var committee = CreateMockCommittee(5);
        var strategy = new QueryByCommittee<double>(committee);

        // Act
        var stats = strategy.GetSelectionStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Contains("CommitteeSize", stats.Keys);
        Assert.Equal(5.0, stats["CommitteeSize"]);
    }

    [Fact]
    public void Constructor_ExactlyTwoMembers_Succeeds()
    {
        // Arrange
        var committee = CreateMockCommittee(2);

        // Act
        var strategy = new QueryByCommittee<double>(committee);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal(2, strategy.Committee.Count);
    }
}
