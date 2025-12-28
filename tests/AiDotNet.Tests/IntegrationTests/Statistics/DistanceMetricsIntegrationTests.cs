using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for distance metrics with mathematically verified ground truth values.
/// All expected values verified against NumPy/SciPy as authoritative sources.
///
/// If any test fails, the CODE must be fixed - never adjust the expected values.
/// </summary>
public class DistanceMetricsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Euclidean Distance Tests

    /// <summary>
    /// Verified with NumPy:
    /// np.linalg.norm(np.array([0, 0]) - np.array([3, 4])) = 5.0
    /// Classic 3-4-5 right triangle.
    /// </summary>
    [Fact]
    public void EuclideanDistance_ClassicTriangle_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - sqrt(9 + 16) = sqrt(25) = 5
        Assert.Equal(5.0, distance, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy:
    /// np.linalg.norm(np.array([1, 2, 3]) - np.array([4, 5, 6])) = 5.196152422706632
    /// sqrt((4-1)² + (5-2)² + (6-3)²) = sqrt(9+9+9) = sqrt(27) = 3*sqrt(3)
    /// </summary>
    [Fact]
    public void EuclideanDistance_3DVector_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - sqrt(27) = 5.196152422706632
        Assert.Equal(5.196152422706632, distance, Tolerance);
    }

    /// <summary>
    /// Distance between identical points should be zero.
    /// </summary>
    [Fact]
    public void EuclideanDistance_IdenticalPoints_ReturnsZero()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(0.0, distance, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy:
    /// np.linalg.norm(np.array([1]) - np.array([5])) = 4.0
    /// </summary>
    [Fact]
    public void EuclideanDistance_SingleDimension_ReturnsAbsoluteDifference()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0 });
        var b = new Vector<double>(new[] { 5.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(4.0, distance, Tolerance);
    }

    /// <summary>
    /// Squared Euclidean distance for efficiency testing.
    /// np.sum((np.array([0, 0]) - np.array([3, 4]))**2) = 25
    /// </summary>
    [Fact]
    public void EuclideanDistance_Squared_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distanceSquared = metric.ComputeSquared(a, b);

        // Assert - 9 + 16 = 25
        Assert.Equal(25.0, distanceSquared, Tolerance);
    }

    #endregion

    #region Manhattan Distance Tests

    /// <summary>
    /// Verified with NumPy:
    /// np.sum(np.abs(np.array([0, 0]) - np.array([3, 4]))) = 7.0
    /// </summary>
    [Fact]
    public void ManhattanDistance_BasicExample_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });
        var metric = new ManhattanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - |3-0| + |4-0| = 3 + 4 = 7
        Assert.Equal(7.0, distance, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy:
    /// np.sum(np.abs(np.array([1, 2, 3]) - np.array([4, 5, 6]))) = 9.0
    /// </summary>
    [Fact]
    public void ManhattanDistance_3DVector_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        var metric = new ManhattanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
        Assert.Equal(9.0, distance, Tolerance);
    }

    /// <summary>
    /// Manhattan distance with negative values.
    /// np.sum(np.abs(np.array([-1, -2]) - np.array([1, 2]))) = 6.0
    /// </summary>
    [Fact]
    public void ManhattanDistance_NegativeValues_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { -1.0, -2.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0 });
        var metric = new ManhattanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - |1-(-1)| + |2-(-2)| = 2 + 4 = 6
        Assert.Equal(6.0, distance, Tolerance);
    }

    /// <summary>
    /// Distance between identical points should be zero.
    /// </summary>
    [Fact]
    public void ManhattanDistance_IdenticalPoints_ReturnsZero()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var metric = new ManhattanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(0.0, distance, Tolerance);
    }

    #endregion

    #region Chebyshev Distance Tests

    /// <summary>
    /// Verified: max(|3-0|, |4-0|) = max(3, 4) = 4
    /// </summary>
    [Fact]
    public void ChebyshevDistance_BasicExample_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });
        var metric = new ChebyshevDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - max(3, 4) = 4
        Assert.Equal(4.0, distance, Tolerance);
    }

    /// <summary>
    /// Verified: max(|4-1|, |5-2|, |6-3|) = max(3, 3, 3) = 3
    /// </summary>
    [Fact]
    public void ChebyshevDistance_3DVector_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        var metric = new ChebyshevDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - max(3, 3, 3) = 3
        Assert.Equal(3.0, distance, Tolerance);
    }

    /// <summary>
    /// Chebyshev distance with one large difference.
    /// max(|0-0|, |0-10|, |0-0|) = 10
    /// </summary>
    [Fact]
    public void ChebyshevDistance_OneLargeDifference_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var b = new Vector<double>(new[] { 0.0, 10.0, 0.0 });
        var metric = new ChebyshevDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - max(0, 10, 0) = 10
        Assert.Equal(10.0, distance, Tolerance);
    }

    /// <summary>
    /// Distance between identical points should be zero.
    /// </summary>
    [Fact]
    public void ChebyshevDistance_IdenticalPoints_ReturnsZero()
    {
        // Arrange
        var a = new Vector<double>(new[] { 5.0, 5.0, 5.0 });
        var b = new Vector<double>(new[] { 5.0, 5.0, 5.0 });
        var metric = new ChebyshevDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(0.0, distance, Tolerance);
    }

    #endregion

    #region Cosine Distance Tests

    /// <summary>
    /// Identical vectors should have cosine distance = 0.
    /// (same direction)
    /// </summary>
    [Fact]
    public void CosineDistance_IdenticalVectors_ReturnsZero()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var metric = new CosineDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - same direction = 0 distance
        Assert.Equal(0.0, distance, Tolerance);
    }

    /// <summary>
    /// Parallel vectors (same direction, different magnitude) should have distance = 0.
    /// [1, 2] and [2, 4] point in the same direction.
    /// </summary>
    [Fact]
    public void CosineDistance_ParallelVectors_ReturnsZero()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0 });
        var b = new Vector<double>(new[] { 2.0, 4.0 });
        var metric = new CosineDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - same direction = 0 distance
        Assert.Equal(0.0, distance, Tolerance);
    }

    /// <summary>
    /// Perpendicular vectors (90°) should have cosine similarity = 0, distance = 1.
    /// [1, 0] and [0, 1] are perpendicular.
    /// </summary>
    [Fact]
    public void CosineDistance_PerpendicularVectors_ReturnsOne()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 0.0 });
        var b = new Vector<double>(new[] { 0.0, 1.0 });
        var metric = new CosineDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - perpendicular = distance 1
        Assert.Equal(1.0, distance, Tolerance);
    }

    /// <summary>
    /// Opposite vectors (180°) should have cosine similarity = -1, distance = 2.
    /// [1, 0] and [-1, 0] point in opposite directions.
    /// </summary>
    [Fact]
    public void CosineDistance_OppositeVectors_ReturnsTwo()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 0.0 });
        var b = new Vector<double>(new[] { -1.0, 0.0 });
        var metric = new CosineDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - opposite direction = distance 2 (1 - (-1) = 2)
        Assert.Equal(2.0, distance, Tolerance);
    }

    /// <summary>
    /// Verified with SciPy:
    /// from scipy.spatial.distance import cosine
    /// cosine([1, 2, 3], [4, 5, 6]) = 0.025368153802923787
    /// </summary>
    [Fact]
    public void CosineDistance_3DVector_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        var metric = new CosineDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - SciPy verified
        Assert.Equal(0.025368153802923787, distance, Tolerance);
    }

    /// <summary>
    /// Cosine similarity should be 1 - distance.
    /// </summary>
    [Fact]
    public void CosineSimilarity_IsOneMinusDistance()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        var metric = new CosineDistance<double>();

        // Act
        var distance = metric.Compute(a, b);
        var similarity = metric.ComputeSimilarity(a, b);

        // Assert
        Assert.Equal(1.0 - distance, similarity, Tolerance);
    }

    #endregion

    #region Minkowski Distance Tests

    /// <summary>
    /// Minkowski with p=1 should equal Manhattan distance.
    /// </summary>
    [Fact]
    public void MinkowskiDistance_P1_EqualsManhattan()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });
        var minkowski = new MinkowskiDistance<double>(1.0);
        var manhattan = new ManhattanDistance<double>();

        // Act
        var minkowskiDist = minkowski.Compute(a, b);
        var manhattanDist = manhattan.Compute(a, b);

        // Assert
        Assert.Equal(manhattanDist, minkowskiDist, Tolerance);
        Assert.Equal(7.0, minkowskiDist, Tolerance);
    }

    /// <summary>
    /// Minkowski with p=2 should equal Euclidean distance.
    /// </summary>
    [Fact]
    public void MinkowskiDistance_P2_EqualsEuclidean()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });
        var minkowski = new MinkowskiDistance<double>(2.0);
        var euclidean = new EuclideanDistance<double>();

        // Act
        var minkowskiDist = minkowski.Compute(a, b);
        var euclideanDist = euclidean.Compute(a, b);

        // Assert
        Assert.Equal(euclideanDist, minkowskiDist, Tolerance);
        Assert.Equal(5.0, minkowskiDist, Tolerance);
    }

    /// <summary>
    /// Verified with SciPy:
    /// from scipy.spatial.distance import minkowski
    /// minkowski([0, 0], [3, 4], p=3) = 4.497941445275415
    /// </summary>
    [Fact]
    public void MinkowskiDistance_P3_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });
        var metric = new MinkowskiDistance<double>(3.0);

        // Act
        var distance = metric.Compute(a, b);

        // Assert - (3^3 + 4^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3) = 4.497941445275415
        Assert.Equal(4.497941445275415, distance, Tolerance);
    }

    /// <summary>
    /// Distance between identical points should be zero for any p.
    /// </summary>
    [Fact]
    public void MinkowskiDistance_IdenticalPoints_ReturnsZero()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var metric = new MinkowskiDistance<double>(3.0);

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(0.0, distance, Tolerance);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void EuclideanDistance_FloatType_ReturnsCorrectValue()
    {
        // Arrange
        var a = new Vector<float>(new[] { 0.0f, 0.0f });
        var b = new Vector<float>(new[] { 3.0f, 4.0f });
        var metric = new EuclideanDistance<float>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(5.0f, distance, 1e-5f);
    }

    [Fact]
    public void ManhattanDistance_FloatType_ReturnsCorrectValue()
    {
        // Arrange
        var a = new Vector<float>(new[] { 0.0f, 0.0f });
        var b = new Vector<float>(new[] { 3.0f, 4.0f });
        var metric = new ManhattanDistance<float>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(7.0f, distance, 1e-5f);
    }

    [Fact]
    public void CosineDistance_FloatType_ReturnsCorrectValue()
    {
        // Arrange
        var a = new Vector<float>(new[] { 1.0f, 0.0f });
        var b = new Vector<float>(new[] { 0.0f, 1.0f });
        var metric = new CosineDistance<float>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - perpendicular vectors
        Assert.Equal(1.0f, distance, 1e-5f);
    }

    #endregion

    #region Triangle Inequality Tests

    /// <summary>
    /// Distance metrics should satisfy the triangle inequality:
    /// d(a, c) <= d(a, b) + d(b, c)
    /// </summary>
    [Fact]
    public void EuclideanDistance_SatisfiesTriangleInequality()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 1.0, 1.0 });
        var c = new Vector<double>(new[] { 2.0, 0.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var dAC = metric.Compute(a, c);
        var dAB = metric.Compute(a, b);
        var dBC = metric.Compute(b, c);

        // Assert - Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        Assert.True(dAC <= dAB + dBC + Tolerance,
            $"Triangle inequality violated: {dAC} > {dAB} + {dBC}");
    }

    [Fact]
    public void ManhattanDistance_SatisfiesTriangleInequality()
    {
        // Arrange
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 1.0, 1.0 });
        var c = new Vector<double>(new[] { 2.0, 0.0 });
        var metric = new ManhattanDistance<double>();

        // Act
        var dAC = metric.Compute(a, c);
        var dAB = metric.Compute(a, b);
        var dBC = metric.Compute(b, c);

        // Assert
        Assert.True(dAC <= dAB + dBC + Tolerance,
            $"Triangle inequality violated: {dAC} > {dAB} + {dBC}");
    }

    #endregion

    #region Symmetry Tests

    /// <summary>
    /// Distance should be symmetric: d(a, b) = d(b, a)
    /// </summary>
    [Fact]
    public void AllDistances_AreSymmetric()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        var euclidean = new EuclideanDistance<double>();
        var manhattan = new ManhattanDistance<double>();
        var chebyshev = new ChebyshevDistance<double>();
        var cosine = new CosineDistance<double>();
        var minkowski = new MinkowskiDistance<double>(3.0);

        // Act & Assert
        Assert.Equal(euclidean.Compute(a, b), euclidean.Compute(b, a), Tolerance);
        Assert.Equal(manhattan.Compute(a, b), manhattan.Compute(b, a), Tolerance);
        Assert.Equal(chebyshev.Compute(a, b), chebyshev.Compute(b, a), Tolerance);
        Assert.Equal(cosine.Compute(a, b), cosine.Compute(b, a), Tolerance);
        Assert.Equal(minkowski.Compute(a, b), minkowski.Compute(b, a), Tolerance);
    }

    #endregion

    #region High-Dimensional Tests

    /// <summary>
    /// Verified with NumPy:
    /// a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    /// b = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    /// np.linalg.norm(a - b) = 18.16590212458495
    /// </summary>
    [Fact]
    public void EuclideanDistance_10Dimensions_ReturnsExactValue()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });
        var b = new Vector<double>(new[] { 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 });
        var metric = new EuclideanDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert - NumPy verified
        Assert.Equal(18.16590212458495, distance, Tolerance);
    }

    #endregion

    #region Edge Cases

    /// <summary>
    /// Different length vectors should throw ArgumentException.
    /// </summary>
    [Fact]
    public void EuclideanDistance_DifferentLengthVectors_ThrowsException()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0 });
        var metric = new EuclideanDistance<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => metric.Compute(a, b));
    }

    /// <summary>
    /// Empty vectors should return zero distance.
    /// </summary>
    [Fact]
    public void ChebyshevDistance_EmptyVectors_ReturnsZero()
    {
        // Arrange
        var a = new Vector<double>(Array.Empty<double>());
        var b = new Vector<double>(Array.Empty<double>());
        var metric = new ChebyshevDistance<double>();

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(0.0, distance, Tolerance);
    }

    /// <summary>
    /// Minkowski with p < 1 should throw ArgumentException.
    /// </summary>
    [Fact]
    public void MinkowskiDistance_InvalidP_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new MinkowskiDistance<double>(0.5));
    }

    #endregion
}
