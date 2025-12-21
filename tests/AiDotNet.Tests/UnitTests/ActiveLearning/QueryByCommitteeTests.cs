using AiDotNet.ActiveLearning.QueryStrategies;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActiveLearning;

/// <summary>
/// Unit tests for the Query-by-Committee strategy.
/// </summary>
public class QueryByCommitteeTests
{
    [Fact]
    public void Constructor_ValidCommitteeSize_InitializesSuccessfully()
    {
        // Act
        var strategy = new QueryByCommittee<double, Matrix<double>, Vector<double>>(
            committeeSize: 5);

        // Assert
        Assert.NotNull(strategy);
        Assert.Equal("QueryByCommittee-VoteEntropy", strategy.Name);
    }

    [Fact]
    public void Constructor_CommitteeSizeLessThan2_ThrowsArgumentException()
    {
        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new QueryByCommittee<double, Matrix<double>, Vector<double>>(committeeSize: 1));

        Assert.Contains("at least 2", exception.Message);
    }

    [Fact]
    public void Constructor_WithDifferentMeasures_InitializesCorrectly()
    {
        // Arrange & Act
        var voteEntropy = new QueryByCommittee<double, Matrix<double>, Vector<double>>(
            committeeSize: 3,
            measure: QueryByCommittee<double, Matrix<double>, Vector<double>>.DisagreementMeasure.VoteEntropy);

        var consensusEntropy = new QueryByCommittee<double, Matrix<double>, Vector<double>>(
            committeeSize: 3,
            measure: QueryByCommittee<double, Matrix<double>, Vector<double>>.DisagreementMeasure.ConsensusEntropy);

        var klDivergence = new QueryByCommittee<double, Matrix<double>, Vector<double>>(
            committeeSize: 3,
            measure: QueryByCommittee<double, Matrix<double>, Vector<double>>.DisagreementMeasure.KLDivergence);

        // Assert
        Assert.Equal("QueryByCommittee-VoteEntropy", voteEntropy.Name);
        Assert.Equal("QueryByCommittee-ConsensusEntropy", consensusEntropy.Name);
        Assert.Equal("QueryByCommittee-KLDivergence", klDivergence.Name);
    }
}
