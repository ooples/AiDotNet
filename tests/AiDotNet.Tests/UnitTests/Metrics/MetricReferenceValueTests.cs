using System;
using AiDotNet.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.Metrics;

/// <summary>
/// Reference values (scikit-learn / SciPy) for metric formulas corrected in the element-wise sweep.
/// </summary>
public class MetricReferenceValueTests
{
    [Fact]
    public void NdcgAtK_TiedScores_MatchesScikitLearnTieAveraging()
    {
        // sklearn.metrics.ndcg_score([[1, 0, 2]], [[1, 1, 0]]) = 0.6900468833579672
        var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0 });
        var relevance = new Vector<double>(new[] { 1.0, 0.0, 2.0 });

        double ndcg = RankingMetrics<double>.NdcgAtK(predicted, relevance, 0, useExponentialGain: false);

        Assert.Equal(0.6900468833579672, ndcg, 12);
    }

    [Fact]
    public void NdcgAtK_TiedScoresWithCutoff_MatchesScikitLearn()
    {
        // sklearn.metrics.ndcg_score([[3, 2, 3, 0, 1, 2]], [[0.9, 0.9, 0.5, 0.5, 0.1, 0.3]], k=4) = 0.8103677588719043
        var predicted = new Vector<double>(new[] { 0.9, 0.9, 0.5, 0.5, 0.1, 0.3 });
        var relevance = new Vector<double>(new[] { 3.0, 2.0, 3.0, 0.0, 1.0, 2.0 });

        double ndcg = RankingMetrics<double>.NdcgAtK(predicted, relevance, 4, useExponentialGain: false);

        Assert.Equal(0.8103677588719043, ndcg, 12);
    }

    [Fact]
    public void FrechetDistance_NonCommutingCovariances_MatchesScipySqrtm()
    {
        // ||m1 - m2||^2 + Tr(S1) + Tr(S2) - 2 Tr(scipy.linalg.sqrtm(S1 @ S2)) = 5.77122044765434
        var mean1 = new Vector<double>(new[] { 0.0, 1.0 });
        var mean2 = new Vector<double>(new[] { 1.0, 3.0 });
        var cov1 = new Matrix<double>(new double[,] { { 1.0, 0.0 }, { 0.0, 4.0 } });
        var cov2 = new Matrix<double>(new double[,] { { 2.0, 1.0 }, { 1.0, 2.0 } });

        double distance = FrechetDistanceHelper<double>.FrechetDistance(mean1, cov1, mean2, cov2);

        Assert.Equal(5.77122044765434, distance, 9);
    }

    [Fact]
    public void FrechetDistance_TinyIdenticalCovariances_IsZero()
    {
        // Identical distributions are at distance 0 whatever the covariance scale; entries far below the
        // eigensolver's absolute 1e-12 threshold must not leave the off-diagonal structure unrotated.
        var mean = new Vector<double>(new[] { 0.0, 0.0 });
        var cov = new Matrix<double>(new double[,] { { 1e-8, 5e-9 }, { 5e-9, 1e-8 } });

        double distance = FrechetDistanceHelper<double>.FrechetDistance(mean, cov, mean, cov);

        Assert.True(Math.Abs(distance) < 1e-20, $"Expected 0, got {distance}.");
    }

    [Fact]
    public void FrechetDistance_Decimal_IdenticalIdentityCovariances_IsZero()
    {
        var mean = new Vector<decimal>(new[] { 0m });
        var cov = new Matrix<decimal>(new decimal[,] { { 1m } });

        decimal distance = FrechetDistanceHelper<decimal>.FrechetDistance(mean, cov, mean, cov);

        Assert.True(Math.Abs(distance) < 1e-12m, $"Expected 0, got {distance}.");
    }

    [Fact]
    public void Perplexity_LargeEqualFloatLogits_IsTwo()
    {
        // Two equal logits give cross-entropy ln 2 and perplexity 2 however large the logits are.
        var logits = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 16777216f, 16777216f }, new[] { 1, 2 });

        float perplexity = LanguageModelMetrics<float>.Perplexity(logits, new[] { 0 });

        Assert.Equal(2f, perplexity, 4);
    }

    [Fact]
    public void TextDetection_FloatPolygonsAtImageScale_KeepAreaAndIntersection()
    {
        // A unit square at (4096, 4096) has area 1 in float, and two crossing diagonals of it meet at
        // its centre, so the square clipped by its own diamond (area 1/2) has IoU 1/2.
        var square = new System.Collections.Generic.List<(float X, float Y)>
        {
            (4096f, 4096f), (4097f, 4096f), (4097f, 4097f), (4096f, 4097f),
        };
        var diamond = new System.Collections.Generic.List<(float X, float Y)>
        {
            (4096.5f, 4096f), (4097f, 4096.5f), (4096.5f, 4097f), (4096f, 4096.5f),
        };

        Assert.Equal(1f, TextDetectionMetrics<float>.PolygonArea(square), 6);
        Assert.Equal(0.5f, TextDetectionMetrics<float>.PolygonIoU(square, diamond), 5);
    }
}
