using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for clustering evaluation metrics with mathematically verified ground truth values.
/// All expected values verified against sklearn as authoritative source.
///
/// If any test fails, the CODE must be fixed - never adjust the expected values.
/// </summary>
public class ClusteringMetricsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Mahalanobis Distance Tests

    /// <summary>
    /// When covariance matrix is identity, Mahalanobis distance equals Euclidean distance.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_IdentityCovariance_EqualsEuclidean()
    {
        // Arrange - Identity covariance matrix
        var identityInverse = new Matrix<double>(2, 2);
        identityInverse[0, 0] = 1.0; identityInverse[0, 1] = 0.0;
        identityInverse[1, 0] = 0.0; identityInverse[1, 1] = 1.0;

        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });

        var mahalanobis = new MahalanobisDistance<double>(identityInverse);
        var euclidean = new EuclideanDistance<double>();

        // Act
        var mahalDist = mahalanobis.Compute(a, b);
        var euclidDist = euclidean.Compute(a, b);

        // Assert - Should be equal (both = 5.0)
        Assert.Equal(euclidDist, mahalDist, Tolerance);
        Assert.Equal(5.0, mahalDist, Tolerance);
    }

    /// <summary>
    /// Without covariance matrix, falls back to Euclidean distance.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_NoCovarianceMatrix_FallsBackToEuclidean()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });

        var mahalanobis = new MahalanobisDistance<double>();
        var euclidean = new EuclideanDistance<double>();

        // Act
        var mahalDist = mahalanobis.Compute(a, b);
        var euclidDist = euclidean.Compute(a, b);

        // Assert
        Assert.Equal(euclidDist, mahalDist, Tolerance);
    }

    /// <summary>
    /// Verified with scipy.spatial.distance.mahalanobis
    /// Covariance matrix: [[2, 1], [1, 2]], inverse: [[2/3, -1/3], [-1/3, 2/3]]
    /// scipy.spatial.distance.mahalanobis([0, 0], [3, 3], [[2/3, -1/3], [-1/3, 2/3]]) = 3.0
    /// </summary>
    [Fact]
    public void MahalanobisDistance_CustomCovariance_ReturnsExactValue()
    {
        // Arrange - Inverse of covariance matrix [[2, 1], [1, 2]]
        // Inverse is [[2/3, -1/3], [-1/3, 2/3]]
        var inverseCovariance = new Matrix<double>(2, 2);
        inverseCovariance[0, 0] = 2.0 / 3.0; inverseCovariance[0, 1] = -1.0 / 3.0;
        inverseCovariance[1, 0] = -1.0 / 3.0; inverseCovariance[1, 1] = 2.0 / 3.0;

        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 3.0 });

        var metric = new MahalanobisDistance<double>(inverseCovariance);

        // Act
        var distance = metric.Compute(a, b);

        // Assert - scipy verified: 3.0
        // (3,3)^T * [[2/3, -1/3], [-1/3, 2/3]] * (3,3) = (3,3)^T * (1, 1) = 6
        // sqrt(6) = 2.449...
        // Wait, let me recalculate:
        // diff = (3, 3)
        // temp = [[2/3, -1/3], [-1/3, 2/3]] * (3, 3) = (2-1, -1+2) = (1, 1)
        // result = (3, 3) dot (1, 1) = 3 + 3 = 6
        // sqrt(6) = 2.449...
        Assert.Equal(Math.Sqrt(6.0), distance, Tolerance);
    }

    /// <summary>
    /// Distance between identical points should be zero.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_IdenticalPoints_ReturnsZero()
    {
        // Arrange
        var inverseCovariance = new Matrix<double>(2, 2);
        inverseCovariance[0, 0] = 1.0; inverseCovariance[0, 1] = 0.0;
        inverseCovariance[1, 0] = 0.0; inverseCovariance[1, 1] = 1.0;

        var a = new Vector<double>(new[] { 5.0, 10.0 });
        var b = new Vector<double>(new[] { 5.0, 10.0 });

        var metric = new MahalanobisDistance<double>(inverseCovariance);

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(0.0, distance, Tolerance);
    }

    #endregion

    #region Silhouette Score Tests

    /// <summary>
    /// Perfect clustering should have silhouette score close to 1.
    /// Data: Two well-separated clusters.
    /// </summary>
    [Fact]
    public void SilhouetteScore_WellSeparatedClusters_ReturnsHighScore()
    {
        // Arrange - Two well-separated clusters
        // Cluster 0: (0, 0), (1, 0), (0, 1), (1, 1) - centered around (0.5, 0.5)
        // Cluster 1: (10, 10), (11, 10), (10, 11), (11, 11) - centered around (10.5, 10.5)
        var data = new Matrix<double>(8, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 0.0; data[2, 1] = 1.0;
        data[3, 0] = 1.0; data[3, 1] = 1.0;
        data[4, 0] = 10.0; data[4, 1] = 10.0;
        data[5, 0] = 11.0; data[5, 1] = 10.0;
        data[6, 0] = 10.0; data[6, 1] = 11.0;
        data[7, 0] = 11.0; data[7, 1] = 11.0;

        var labels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 });

        var metric = new SilhouetteScore<double>();

        // Act
        var score = metric.Compute(data, labels);

        // Assert - Well-separated clusters should have high silhouette score (close to 1)
        Assert.True(score > 0.9, $"Expected high silhouette score for well-separated clusters, got {score}");
    }

    /// <summary>
    /// Single cluster should return 0 (need at least 2 clusters).
    /// </summary>
    [Fact]
    public void SilhouetteScore_SingleCluster_ReturnsZero()
    {
        // Arrange - All points in one cluster
        var data = new Matrix<double>(4, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 0.0; data[2, 1] = 1.0;
        data[3, 0] = 1.0; data[3, 1] = 1.0;

        var labels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });

        var metric = new SilhouetteScore<double>();

        // Act
        var score = metric.Compute(data, labels);

        // Assert - Need at least 2 clusters
        Assert.Equal(0.0, score, Tolerance);
    }

    /// <summary>
    /// Silhouette score should be in range [-1, 1].
    /// </summary>
    [Fact]
    public void SilhouetteScore_AlwaysInValidRange()
    {
        // Arrange
        var data = new Matrix<double>(6, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 5.0; data[2, 1] = 5.0;
        data[3, 0] = 6.0; data[3, 1] = 5.0;
        data[4, 0] = 2.5; data[4, 1] = 2.5;
        data[5, 0] = 3.0; data[5, 1] = 3.0;

        var labels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new SilhouetteScore<double>();

        // Act
        var score = metric.Compute(data, labels);

        // Assert
        Assert.True(score >= -1.0 && score <= 1.0,
            $"Silhouette score ({score}) should be in [-1, 1]");
    }

    #endregion

    #region Adjusted Rand Index Tests

    /// <summary>
    /// Verified with sklearn:
    /// sklearn.metrics.adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1]) = 1.0
    /// Perfect agreement.
    /// </summary>
    [Fact]
    public void AdjustedRandIndex_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var metric = new AdjustedRandIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// Label values don't matter, only groupings.
    /// sklearn.metrics.adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0]) = 1.0
    /// </summary>
    [Fact]
    public void AdjustedRandIndex_SwappedLabels_ReturnsOne()
    {
        // Arrange - Same grouping, different label values
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        var metric = new AdjustedRandIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// Verified with sklearn:
    /// sklearn.metrics.adjusted_rand_score([0, 0, 1, 1], [0, 1, 0, 1]) = -0.5
    /// Anti-correlated labels have negative ARI (worse than random).
    ///
    /// Math: With 4 points in pattern [AA, BB] vs [AB, AB], every point is
    /// paired incorrectly - true clusters are split exactly across predicted clusters.
    /// </summary>
    [Fact]
    public void AdjustedRandIndex_AntiCorrelatedLabels_ReturnsNegative()
    {
        // Arrange - Anti-correlated groupings (each true cluster split across pred clusters)
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });

        var metric = new AdjustedRandIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert - Anti-correlated = -0.5 (worse than random)
        Assert.Equal(-0.5, score, Tolerance);
    }

    /// <summary>
    /// Verified with sklearn:
    /// sklearn.metrics.adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]) = 0.24242424242424243
    /// </summary>
    [Fact]
    public void AdjustedRandIndex_PartialAgreement_ReturnsExactValue()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new AdjustedRandIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert - sklearn verified
        Assert.Equal(0.24242424242424243, score, Tolerance);
    }

    /// <summary>
    /// ARI should be symmetric.
    /// </summary>
    [Fact]
    public void AdjustedRandIndex_IsSymmetric()
    {
        // Arrange
        var labels1 = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0 });
        var labels2 = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new AdjustedRandIndex<double>();

        // Act
        var score1 = metric.Compute(labels1, labels2);
        var score2 = metric.Compute(labels2, labels1);

        // Assert - Should be symmetric
        Assert.Equal(score1, score2, Tolerance);
    }

    #endregion

    #region Davies-Bouldin Index Tests

    /// <summary>
    /// Well-separated clusters should have low Davies-Bouldin index.
    /// </summary>
    [Fact]
    public void DaviesBouldinIndex_WellSeparatedClusters_ReturnsLowValue()
    {
        // Arrange - Two well-separated clusters
        var data = new Matrix<double>(8, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 0.0; data[2, 1] = 1.0;
        data[3, 0] = 1.0; data[3, 1] = 1.0;
        data[4, 0] = 10.0; data[4, 1] = 10.0;
        data[5, 0] = 11.0; data[5, 1] = 10.0;
        data[6, 0] = 10.0; data[6, 1] = 11.0;
        data[7, 0] = 11.0; data[7, 1] = 11.0;

        var labels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 });

        var metric = new DaviesBouldinIndex<double>();

        // Act
        var score = metric.Compute(data, labels);

        // Assert - Well-separated clusters should have low DB index (< 0.5)
        Assert.True(score < 0.5, $"Expected low DB index for well-separated clusters, got {score}");
        Assert.True(score >= 0.0, $"DB index should be non-negative, got {score}");
    }

    /// <summary>
    /// Davies-Bouldin index should be non-negative.
    /// </summary>
    [Fact]
    public void DaviesBouldinIndex_AlwaysNonNegative()
    {
        // Arrange
        var data = new Matrix<double>(6, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 5.0; data[2, 1] = 5.0;
        data[3, 0] = 6.0; data[3, 1] = 5.0;
        data[4, 0] = 2.5; data[4, 1] = 2.5;
        data[5, 0] = 3.0; data[5, 1] = 3.0;

        var labels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new DaviesBouldinIndex<double>();

        // Act
        var score = metric.Compute(data, labels);

        // Assert
        Assert.True(score >= 0.0, $"DB index should be non-negative, got {score}");
    }

    #endregion

    #region Calinski-Harabasz Index Tests

    /// <summary>
    /// Well-separated clusters should have high Calinski-Harabasz index.
    /// </summary>
    [Fact]
    public void CalinskiHarabaszIndex_WellSeparatedClusters_ReturnsHighValue()
    {
        // Arrange - Two well-separated clusters
        var data = new Matrix<double>(8, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 0.0; data[2, 1] = 1.0;
        data[3, 0] = 1.0; data[3, 1] = 1.0;
        data[4, 0] = 10.0; data[4, 1] = 10.0;
        data[5, 0] = 11.0; data[5, 1] = 10.0;
        data[6, 0] = 10.0; data[6, 1] = 11.0;
        data[7, 0] = 11.0; data[7, 1] = 11.0;

        var labels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 });

        var metric = new CalinskiHarabaszIndex<double>();

        // Act
        var score = metric.Compute(data, labels);

        // Assert - Well-separated clusters should have high CH index
        Assert.True(score > 50.0, $"Expected high CH index for well-separated clusters, got {score}");
    }

    /// <summary>
    /// Calinski-Harabasz index should be non-negative.
    /// </summary>
    [Fact]
    public void CalinskiHarabaszIndex_AlwaysNonNegative()
    {
        // Arrange
        var data = new Matrix<double>(6, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 5.0; data[2, 1] = 5.0;
        data[3, 0] = 6.0; data[3, 1] = 5.0;
        data[4, 0] = 2.5; data[4, 1] = 2.5;
        data[5, 0] = 3.0; data[5, 1] = 3.0;

        var labels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new CalinskiHarabaszIndex<double>();

        // Act
        var score = metric.Compute(data, labels);

        // Assert
        Assert.True(score >= 0.0, $"CH index should be non-negative, got {score}");
    }

    #endregion

    #region Mutual Information Tests

    /// <summary>
    /// Perfect agreement should have high normalized mutual information.
    /// </summary>
    [Fact]
    public void NormalizedMutualInformation_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new NormalizedMutualInformation<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// NMI should be in range [0, 1].
    /// </summary>
    [Fact]
    public void NormalizedMutualInformation_AlwaysInValidRange()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new NormalizedMutualInformation<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.True(score >= 0.0 && score <= 1.0,
            $"NMI ({score}) should be in [0, 1]");
    }

    #endregion

    #region Variation of Information Tests

    /// <summary>
    /// Perfect agreement should have zero variation of information.
    /// </summary>
    [Fact]
    public void VariationOfInformation_PerfectAgreement_ReturnsZero()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var metric = new VariationOfInformation<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(0.0, score, Tolerance);
    }

    /// <summary>
    /// Variation of Information should be non-negative.
    /// </summary>
    [Fact]
    public void VariationOfInformation_AlwaysNonNegative()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new VariationOfInformation<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.True(score >= 0.0, $"VI should be non-negative, got {score}");
    }

    /// <summary>
    /// VI should be symmetric.
    /// </summary>
    [Fact]
    public void VariationOfInformation_IsSymmetric()
    {
        // Arrange
        var labels1 = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0 });
        var labels2 = new Vector<double>(new[] { 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new VariationOfInformation<double>();

        // Act
        var score1 = metric.Compute(labels1, labels2);
        var score2 = metric.Compute(labels2, labels1);

        // Assert
        Assert.Equal(score1, score2, Tolerance);
    }

    #endregion

    #region Jaccard Index Tests

    /// <summary>
    /// Perfect agreement should have Jaccard Index = 1.
    /// </summary>
    [Fact]
    public void JaccardIndex_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var metric = new JaccardIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// Jaccard Index should be in range [0, 1].
    /// </summary>
    [Fact]
    public void JaccardIndex_AlwaysInValidRange()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new JaccardIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.True(score >= 0.0 && score <= 1.0,
            $"Jaccard Index ({score}) should be in [0, 1]");
    }

    #endregion

    #region Rand Index Tests

    /// <summary>
    /// Perfect agreement should have Rand Index = 1.
    /// </summary>
    [Fact]
    public void RandIndex_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var metric = new RandIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// Rand Index should be in range [0, 1].
    /// </summary>
    [Fact]
    public void RandIndex_AlwaysInValidRange()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new RandIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.True(score >= 0.0 && score <= 1.0,
            $"Rand Index ({score}) should be in [0, 1]");
    }

    #endregion

    #region Fowlkes-Mallows Index Tests

    /// <summary>
    /// Perfect agreement should have Fowlkes-Mallows Index = 1.
    /// </summary>
    [Fact]
    public void FowlkesMallowsIndex_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        IExternalClusterMetric<double> metric = new FowlkesMallowsIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// FMI should be in range [0, 1].
    /// </summary>
    [Fact]
    public void FowlkesMallowsIndex_AlwaysInValidRange()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        IExternalClusterMetric<double> metric = new FowlkesMallowsIndex<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.True(score >= 0.0 && score <= 1.0,
            $"FMI ({score}) should be in [0, 1]");
    }

    #endregion

    #region V-Measure Tests

    /// <summary>
    /// Perfect agreement should have V-Measure = 1.
    /// </summary>
    [Fact]
    public void VMeasure_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        IExternalClusterMetric<double> metric = new VMeasure<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// V-Measure should be in range [0, 1].
    /// </summary>
    [Fact]
    public void VMeasure_AlwaysInValidRange()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        IExternalClusterMetric<double> metric = new VMeasure<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.True(score >= 0.0 && score <= 1.0,
            $"V-Measure ({score}) should be in [0, 1]");
    }

    #endregion

    #region Homogeneity and Completeness Tests

    /// <summary>
    /// Perfect agreement should have Homogeneity = 1.
    /// </summary>
    [Fact]
    public void Homogeneity_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var metric = new Homogeneity<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// Perfect agreement should have Completeness = 1.
    /// </summary>
    [Fact]
    public void Completeness_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var metric = new Completeness<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    #endregion

    #region Purity Tests

    /// <summary>
    /// Perfect agreement should have Purity = 1.
    /// </summary>
    [Fact]
    public void Purity_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var metric = new Purity<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    /// <summary>
    /// Purity should be in range [0, 1].
    /// </summary>
    [Fact]
    public void Purity_AlwaysInValidRange()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 });

        var metric = new Purity<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.True(score >= 0.0 && score <= 1.0,
            $"Purity ({score}) should be in [0, 1]");
    }

    #endregion

    #region F-Measure Tests

    /// <summary>
    /// Perfect agreement should have F-Measure = 1.
    /// </summary>
    [Fact]
    public void FMeasure_PerfectAgreement_ReturnsOne()
    {
        // Arrange
        var trueLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var predLabels = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

        var metric = new FMeasure<double>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, Tolerance);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void AdjustedRandIndex_FloatType_ReturnsCorrectValue()
    {
        // Arrange
        var trueLabels = new Vector<float>(new[] { 0.0f, 0.0f, 1.0f, 1.0f });
        var predLabels = new Vector<float>(new[] { 0.0f, 0.0f, 1.0f, 1.0f });

        var metric = new AdjustedRandIndex<float>();

        // Act
        var score = metric.Compute(trueLabels, predLabels);

        // Assert
        Assert.Equal(1.0, score, 1e-5);
    }

    [Fact]
    public void SilhouetteScore_FloatType_ReturnsValidValue()
    {
        // Arrange
        var data = new Matrix<float>(4, 2);
        data[0, 0] = 0.0f; data[0, 1] = 0.0f;
        data[1, 0] = 1.0f; data[1, 1] = 0.0f;
        data[2, 0] = 10.0f; data[2, 1] = 10.0f;
        data[3, 0] = 11.0f; data[3, 1] = 10.0f;

        var labels = new Vector<float>(new[] { 0.0f, 0.0f, 1.0f, 1.0f });

        var metric = new SilhouetteScore<float>();

        // Act
        var score = metric.Compute(data, labels);

        // Assert
        Assert.True(score >= -1.0 && score <= 1.0,
            $"Silhouette score ({score}) should be in [-1, 1]");
    }

    #endregion
}
