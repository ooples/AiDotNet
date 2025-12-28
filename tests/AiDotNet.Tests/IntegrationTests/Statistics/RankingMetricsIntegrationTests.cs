using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for ranking metrics (NDCG, MRR, MAP).
/// Ground truth values verified against standard implementations.
/// </summary>
public class RankingMetricsIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region Mean Reciprocal Rank (MRR) Tests

    [Fact]
    public void MRR_FirstItemRelevant_ReturnsOne()
    {
        // Actual: relevant item has actual=1, others have actual=0
        // Predicted: highest predicted score on the relevant item
        var actual = Vector<double>.FromArray([1.0, 0.0, 0.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.5, 0.3, 0.2, 0.1]);

        var result = StatisticsHelper<double>.CalculateMeanReciprocalRank(actual, predicted);

        // First item is relevant (sorted by predicted: first is highest)
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MRR_SecondItemRelevant_ReturnsHalf()
    {
        // Relevant item is second when sorted by predicted scores
        var actual = Vector<double>.FromArray([0.0, 1.0, 0.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.3, 0.2, 0.1]);

        var result = StatisticsHelper<double>.CalculateMeanReciprocalRank(actual, predicted);

        // MRR = 1/2 because relevant item is at position 2
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void MRR_ThirdItemRelevant_ReturnsOneThird()
    {
        var actual = Vector<double>.FromArray([0.0, 0.0, 1.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.2, 0.1]);

        var result = StatisticsHelper<double>.CalculateMeanReciprocalRank(actual, predicted);

        // MRR = 1/3
        Assert.Equal(1.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void MRR_LastItemRelevant_ReturnsReciprocal()
    {
        var actual = Vector<double>.FromArray([0.0, 0.0, 0.0, 0.0, 1.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var result = StatisticsHelper<double>.CalculateMeanReciprocalRank(actual, predicted);

        // MRR = 1/5
        Assert.Equal(0.2, result, Tolerance);
    }

    [Fact]
    public void MRR_NoRelevantItems_ReturnsZero()
    {
        var actual = Vector<double>.FromArray([0.0, 0.0, 0.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var result = StatisticsHelper<double>.CalculateMeanReciprocalRank(actual, predicted);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MRR_MultipleRelevantItems_ReturnsFirstPosition()
    {
        // Multiple relevant items - MRR only considers the first one
        var actual = Vector<double>.FromArray([0.0, 1.0, 1.0, 1.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var result = StatisticsHelper<double>.CalculateMeanReciprocalRank(actual, predicted);

        // First relevant item is at position 2 (index 1 when sorted by predicted)
        Assert.Equal(0.5, result, Tolerance);
    }

    #endregion

    #region Mean Average Precision (MAP) Tests

    [Fact]
    public void MAP_AllRelevantAtTop_ReturnsOne()
    {
        // All relevant items are at the top
        var actual = Vector<double>.FromArray([1.0, 1.0, 0.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.3, 0.2, 0.1]);

        var result = StatisticsHelper<double>.CalculateMeanAveragePrecision(actual, predicted, 5);

        // Precision@1 = 1/1 = 1.0, Precision@2 = 2/2 = 1.0
        // MAP = (1.0 + 1.0) / 2 = 1.0
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MAP_SingleRelevant_ReturnsReciprocalRank()
    {
        // Single relevant item
        var actual = Vector<double>.FromArray([0.0, 1.0, 0.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.3, 0.2, 0.1]);

        var result = StatisticsHelper<double>.CalculateMeanAveragePrecision(actual, predicted, 5);

        // Precision@2 = 1/2 = 0.5, only 1 relevant item
        // MAP = 0.5 / 1 = 0.5
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void MAP_AlternatingRelevance_ReturnsExpectedValue()
    {
        // Relevant items at positions 1, 3, 5 when sorted by predicted
        var actual = Vector<double>.FromArray([1.0, 0.0, 1.0, 0.0, 1.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var result = StatisticsHelper<double>.CalculateMeanAveragePrecision(actual, predicted, 5);

        // Precision@1 = 1/1 = 1.0 (relevant at pos 1)
        // Precision@3 = 2/3 = 0.667 (relevant at pos 3)
        // Precision@5 = 3/5 = 0.6 (relevant at pos 5)
        // MAP = (1.0 + 0.667 + 0.6) / 3 = 0.756
        Assert.Equal(0.756, result, 0.01);
    }

    [Fact]
    public void MAP_KLessThanLength_UsesOnlyTopK()
    {
        var actual = Vector<double>.FromArray([1.0, 0.0, 1.0, 0.0, 1.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var resultK3 = StatisticsHelper<double>.CalculateMeanAveragePrecision(actual, predicted, 3);

        // With k=3, only top 3 items considered
        // Precision@1 = 1/1 = 1.0 (relevant at pos 1)
        // Precision@3 = 2/3 = 0.667 (relevant at pos 3)
        // MAP = (1.0 + 0.667) / 2 = 0.833
        Assert.Equal(0.833, resultK3, 0.01);
    }

    #endregion

    #region NDCG Tests

    [Fact]
    public void NDCG_PerfectRanking_ReturnsOne()
    {
        // Items are already in perfect relevance order
        var actual = Vector<double>.FromArray([3.0, 2.0, 1.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var result = StatisticsHelper<double>.CalculateNDCG(actual, predicted, 5);

        // Perfect ranking should give NDCG = 1.0
        Assert.Equal(1.0, result, 0.01);
    }

    [Fact]
    public void NDCG_ReverseRanking_LowerThanPerfect()
    {
        // Items are in reverse relevance order (worst ranking)
        var actual = Vector<double>.FromArray([0.0, 0.0, 1.0, 2.0, 3.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var result = StatisticsHelper<double>.CalculateNDCG(actual, predicted, 5);

        // Reverse ranking should give lower NDCG than perfect
        Assert.True(result < 1.0, "Reverse ranking should have NDCG < 1");
        Assert.True(result > 0.0, "NDCG should be positive");
    }

    [Fact]
    public void NDCG_KLessThanLength_UsesOnlyTopK()
    {
        var actual = Vector<double>.FromArray([3.0, 2.0, 1.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var resultK3 = StatisticsHelper<double>.CalculateNDCG(actual, predicted, 3);
        var resultK5 = StatisticsHelper<double>.CalculateNDCG(actual, predicted, 5);

        // Both should be 1.0 for perfect ranking within the window
        Assert.Equal(1.0, resultK3, 0.01);
        Assert.Equal(1.0, resultK5, 0.01);
    }

    [Fact]
    public void NDCG_GradedRelevance_HigherRelevanceWeightsMore()
    {
        // Higher relevance scores should have more impact
        var actualHigh = Vector<double>.FromArray([5.0, 1.0, 0.0, 0.0, 0.0]);
        var actualLow = Vector<double>.FromArray([1.0, 5.0, 0.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var resultHigh = StatisticsHelper<double>.CalculateNDCG(actualHigh, predicted, 5);
        var resultLow = StatisticsHelper<double>.CalculateNDCG(actualLow, predicted, 5);

        // Putting high relevance item first should give higher NDCG
        Assert.True(resultHigh >= resultLow, "High relevance at top should give higher NDCG");
    }

    [Fact]
    public void NDCG_AllZeroRelevance_ReturnsNaNOrZero()
    {
        var actual = Vector<double>.FromArray([0.0, 0.0, 0.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var result = StatisticsHelper<double>.CalculateNDCG(actual, predicted, 5);

        // IDCG = 0, so NDCG should be NaN or 0
        Assert.True(double.IsNaN(result) || result == 0.0);
    }

    [Fact]
    public void NDCG_BinaryRelevance_MatchesExpected()
    {
        // Binary relevance (0/1)
        var actual = Vector<double>.FromArray([1.0, 0.0, 1.0, 0.0, 0.0]);
        var predicted = Vector<double>.FromArray([0.9, 0.8, 0.7, 0.6, 0.5]);

        var result = StatisticsHelper<double>.CalculateNDCG(actual, predicted, 5);

        // DCG = 1/log(2) + 1/log(4) ≈ 1.443 + 0.721 = 2.164
        // IDCG = 1/log(2) + 1/log(3) ≈ 1.443 + 0.910 = 2.353
        // NDCG = 2.164 / 2.353 ≈ 0.92
        Assert.True(result > 0.9 && result < 1.0);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void MRR_FloatType_ReturnsCorrectValue()
    {
        var actual = Vector<float>.FromArray([1.0f, 0.0f, 0.0f, 0.0f, 0.0f]);
        var predicted = Vector<float>.FromArray([0.9f, 0.5f, 0.3f, 0.2f, 0.1f]);

        var result = StatisticsHelper<float>.CalculateMeanReciprocalRank(actual, predicted);

        Assert.Equal(1.0f, result, 1e-4f);
    }

    [Fact]
    public void MAP_FloatType_ReturnsCorrectValue()
    {
        var actual = Vector<float>.FromArray([1.0f, 1.0f, 0.0f, 0.0f, 0.0f]);
        var predicted = Vector<float>.FromArray([0.9f, 0.8f, 0.3f, 0.2f, 0.1f]);

        var result = StatisticsHelper<float>.CalculateMeanAveragePrecision(actual, predicted, 5);

        Assert.Equal(1.0f, result, 0.01f);
    }

    [Fact]
    public void NDCG_FloatType_ReturnsCorrectValue()
    {
        var actual = Vector<float>.FromArray([3.0f, 2.0f, 1.0f, 0.0f, 0.0f]);
        var predicted = Vector<float>.FromArray([0.9f, 0.8f, 0.7f, 0.6f, 0.5f]);

        var result = StatisticsHelper<float>.CalculateNDCG(actual, predicted, 5);

        Assert.True(result > 0.99f && result <= 1.0f, $"Expected NDCG ≈ 1.0, got {result}");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void MRR_SingleItem_ReturnsExpected()
    {
        var actualRelevant = Vector<double>.FromArray([1.0]);
        var actualIrrelevant = Vector<double>.FromArray([0.0]);
        var predicted = Vector<double>.FromArray([0.5]);

        Assert.Equal(1.0, StatisticsHelper<double>.CalculateMeanReciprocalRank(actualRelevant, predicted), Tolerance);
        Assert.Equal(0.0, StatisticsHelper<double>.CalculateMeanReciprocalRank(actualIrrelevant, predicted), Tolerance);
    }

    [Fact]
    public void MAP_SingleRelevantItem_ReturnsExpected()
    {
        var actual = Vector<double>.FromArray([1.0]);
        var predicted = Vector<double>.FromArray([0.5]);

        var result = StatisticsHelper<double>.CalculateMeanAveragePrecision(actual, predicted, 1);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void NDCG_SingleItem_ReturnsOne()
    {
        var actual = Vector<double>.FromArray([3.0]);
        var predicted = Vector<double>.FromArray([0.5]);

        var result = StatisticsHelper<double>.CalculateNDCG(actual, predicted, 1);

        // Single item, perfect ordering trivially
        Assert.Equal(1.0, result, 0.01);
    }

    #endregion
}
