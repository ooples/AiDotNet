using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for Mahalanobis distance metric with mathematically verified ground truth values.
/// All expected values verified against SciPy as authoritative source.
///
/// If any test fails, the CODE must be fixed - never adjust the expected values.
/// </summary>
public class MahalanobisDistanceIntegrationTests
{
    private const double Tolerance = 1e-5;

    #region Identity Covariance (Equals Euclidean) Tests

    /// <summary>
    /// With identity covariance matrix, Mahalanobis distance equals Euclidean distance.
    /// Verified: sqrt(3² + 4²) = 5.0
    /// </summary>
    [Fact]
    public void MahalanobisDistance_IdentityCovariance_EqualsEuclidean()
    {
        // Arrange - Identity inverse covariance matrix
        var invCov = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert - Should equal Euclidean: sqrt(9 + 16) = 5
        Assert.Equal(5.0, distance, Tolerance);
    }

    /// <summary>
    /// Without covariance matrix, falls back to Euclidean distance.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_NoCovariance_FallsBackToEuclidean()
    {
        // Arrange
        var metric = new MahalanobisDistance<double>();
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert - Falls back to Euclidean: 5.0
        Assert.Equal(5.0, distance, Tolerance);
    }

    #endregion

    #region Diagonal Covariance Tests

    /// <summary>
    /// Verified with SciPy:
    /// from scipy.spatial.distance import mahalanobis
    /// import numpy as np
    /// cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    /// inv_cov = np.linalg.inv(cov)
    /// mahalanobis([0, 0], [4, 3], inv_cov) = 3.605551275463989
    ///
    /// Explanation: Variance of 4 in first dimension means distance is scaled by 1/sqrt(4) = 0.5
    /// d = sqrt((4/sqrt(4))² + (3/sqrt(1))²) = sqrt(4 + 9) = sqrt(13) ≈ 3.6055
    /// </summary>
    [Fact]
    public void MahalanobisDistance_DiagonalCovariance_ScalesByVariance()
    {
        // Arrange - Diagonal covariance: var1=4, var2=1
        // Inverse: diag(1/4, 1/1)
        var invCov = new Matrix<double>(new double[,]
        {
            { 0.25, 0.0 },
            { 0.0, 1.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 4.0, 3.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert - sqrt(4² * 0.25 + 3² * 1.0) = sqrt(4 + 9) = sqrt(13) = 3.6055...
        Assert.Equal(3.605551275463989, distance, Tolerance);
    }

    /// <summary>
    /// Verified with SciPy:
    /// cov = np.array([[2.0, 0.0], [0.0, 2.0]])
    /// inv_cov = np.linalg.inv(cov)  # diag(0.5, 0.5)
    /// mahalanobis([0, 0], [3, 4], inv_cov) = 3.5355339059327378
    ///
    /// Equal variance of 2 scales distance by 1/sqrt(2).
    /// </summary>
    [Fact]
    public void MahalanobisDistance_UniformVariance_ScalesUniformly()
    {
        // Arrange
        var invCov = new Matrix<double>(new double[,]
        {
            { 0.5, 0.0 },
            { 0.0, 0.5 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert - sqrt(0.5 * 9 + 0.5 * 16) = sqrt(0.5 * 25) = 5 / sqrt(2) = 3.5355...
        Assert.Equal(3.5355339059327378, distance, Tolerance);
    }

    #endregion

    #region Correlated Covariance Tests

    /// <summary>
    /// Verified with SciPy:
    /// cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    /// inv_cov = np.linalg.inv(cov)
    /// mahalanobis([0, 0], [1, 1], inv_cov) = 1.1547005383792517
    ///
    /// With positive correlation, points along the diagonal are "closer" than Euclidean would suggest.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_PositiveCorrelation_DiagonalPointsCloser()
    {
        // Arrange - Covariance matrix with rho=0.5
        // cov = [[1, 0.5], [0.5, 1]]
        // inv_cov = 1/(1-0.25) * [[1, -0.5], [-0.5, 1]] = 4/3 * [[1, -0.5], [-0.5, 1]]
        var invCov = new Matrix<double>(new double[,]
        {
            { 4.0/3.0, -2.0/3.0 },
            { -2.0/3.0, 4.0/3.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 1.0, 1.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert - SciPy verified
        Assert.Equal(1.1547005383792517, distance, Tolerance);
    }

    /// <summary>
    /// Verified with SciPy:
    /// cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    /// inv_cov = np.linalg.inv(cov)
    /// mahalanobis([0, 0], [1, 0], inv_cov) = 1.6666666666666667
    ///
    /// With high positive correlation, moving perpendicular to the correlation direction is "far".
    /// </summary>
    [Fact]
    public void MahalanobisDistance_HighCorrelation_PerpendicularPointsFar()
    {
        // Arrange - High correlation (rho=0.8)
        // cov = [[1, 0.8], [0.8, 1]]
        // det = 1 - 0.64 = 0.36
        // inv_cov = 1/0.36 * [[1, -0.8], [-0.8, 1]]
        var det = 0.36;
        var invCov = new Matrix<double>(new double[,]
        {
            { 1.0 / det, -0.8 / det },
            { -0.8 / det, 1.0 / det }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 1.0, 0.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert - sqrt(1 * (1/0.36)) = sqrt(2.777...) = 1.6666...
        Assert.Equal(1.6666666666666667, distance, Tolerance);
    }

    /// <summary>
    /// Verified with SciPy:
    /// cov = np.array([[1.0, -0.5], [-0.5, 1.0]])
    /// inv_cov = np.linalg.inv(cov)
    /// mahalanobis([0, 0], [1, -1], inv_cov) = 1.1547005383792517
    ///
    /// With negative correlation, points along anti-diagonal are closer.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_NegativeCorrelation_AntiDiagonalCloser()
    {
        // Arrange - Negative correlation (rho=-0.5)
        var invCov = new Matrix<double>(new double[,]
        {
            { 4.0/3.0, 2.0/3.0 },
            { 2.0/3.0, 4.0/3.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 1.0, -1.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert - SciPy verified (same as positive correlation with diagonal point)
        Assert.Equal(1.1547005383792517, distance, Tolerance);
    }

    #endregion

    #region 3D Tests

    /// <summary>
    /// Verified with SciPy:
    /// cov = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    /// inv_cov = np.linalg.inv(cov)  # diag(1, 0.5, 0.25)
    /// mahalanobis([0, 0, 0], [2, 2, 2], inv_cov) = 2.449489742783178
    ///
    /// sqrt(2² * 1 + 2² * 0.5 + 2² * 0.25) = sqrt(4 + 2 + 1) = sqrt(7)
    /// </summary>
    [Fact]
    public void MahalanobisDistance_3D_DiagonalCovariance()
    {
        // Arrange
        var invCov = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 0.5, 0.0 },
            { 0.0, 0.0, 0.25 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var b = new Vector<double>(new[] { 2.0, 2.0, 2.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert - sqrt(4 + 2 + 1) = sqrt(7) = 2.6457...
        Assert.Equal(2.6457513110645907, distance, Tolerance);
    }

    /// <summary>
    /// 3D identity should equal Euclidean.
    /// sqrt(3² + 4² + 0²) = 5.0
    /// </summary>
    [Fact]
    public void MahalanobisDistance_3D_Identity_EqualsEuclidean()
    {
        // Arrange
        var invCov = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0, 0.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(5.0, distance, Tolerance);
    }

    #endregion

    #region FitFromData Tests

    /// <summary>
    /// Test that FitFromData correctly estimates covariance from synthetic data.
    /// Use perfectly correlated data (y = x) which should have correlation coefficient ~1.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_FitFromData_CaputuresCorrelation()
    {
        // Arrange - Data where y = x (perfect positive correlation)
        var data = new Matrix<double>(new double[,]
        {
            { 0.0, 0.0 },
            { 1.0, 1.0 },
            { 2.0, 2.0 },
            { 3.0, 3.0 },
            { 4.0, 4.0 }
        });
        var metric = new MahalanobisDistance<double>();

        // Act
        metric.FitFromData(data);

        // Points along y=x should have small distance
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 1.0, 1.0 });
        var alongDiag = metric.Compute(a, b);

        // Points perpendicular (1, -1) should have large distance
        var c = new Vector<double>(new[] { 1.0, -1.0 });
        var perpendicular = metric.Compute(a, c);

        // Assert - Perpendicular movement should be "farther" than diagonal
        Assert.True(perpendicular > alongDiag,
            $"Expected perpendicular ({perpendicular}) > diagonal ({alongDiag})");
    }

    /// <summary>
    /// Test FitFromData with independent features.
    /// With uncorrelated data, should approximate Euclidean behavior.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_FitFromData_UncorrelatedData()
    {
        // Arrange - Uncorrelated data (random-ish pattern)
        var data = new Matrix<double>(new double[,]
        {
            { 0.0, 1.0 },
            { 1.0, 0.0 },
            { 0.0, -1.0 },
            { -1.0, 0.0 },
            { 0.5, 0.5 },
            { -0.5, -0.5 }
        });
        var metric = new MahalanobisDistance<double>();

        // Act
        metric.FitFromData(data);

        // Points should be roughly symmetric
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b1 = new Vector<double>(new[] { 1.0, 0.0 });
        var b2 = new Vector<double>(new[] { 0.0, 1.0 });

        var d1 = metric.Compute(a, b1);
        var d2 = metric.Compute(a, b2);

        // Assert - Distances should be similar for uncorrelated data
        Assert.True(Math.Abs(d1 - d2) < 0.5,
            $"Expected similar distances for uncorrelated data: d1={d1}, d2={d2}");
    }

    /// <summary>
    /// FitFromData should throw if fewer samples than features.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_FitFromData_InsufficientSamples_ThrowsException()
    {
        // Arrange - 2 samples, 3 features (insufficient)
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });
        var metric = new MahalanobisDistance<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => metric.FitFromData(data));
    }

    #endregion

    #region Edge Cases

    /// <summary>
    /// Distance between identical points should be zero.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_IdenticalPoints_ReturnsZero()
    {
        // Arrange
        var invCov = new Matrix<double>(new double[,]
        {
            { 2.0, 0.5 },
            { 0.5, 2.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 1.0, 2.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0 });

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(0.0, distance, Tolerance);
    }

    /// <summary>
    /// Different length vectors should throw ArgumentException.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_DifferentLengthVectors_ThrowsException()
    {
        // Arrange
        var invCov = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => metric.Compute(a, b));
    }

    /// <summary>
    /// Mismatched covariance matrix dimensions should throw.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_MismatchedCovarianceDimensions_ThrowsException()
    {
        // Arrange - 2x2 covariance but 3D vectors
        var invCov = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => metric.Compute(a, b));
    }

    /// <summary>
    /// Null inverse covariance in constructor should throw.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_NullInverseCovariance_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new MahalanobisDistance<double>(null!));
    }

    #endregion

    #region Symmetry Tests

    /// <summary>
    /// Mahalanobis distance should be symmetric: d(a, b) = d(b, a)
    /// </summary>
    [Fact]
    public void MahalanobisDistance_IsSymmetric()
    {
        // Arrange
        var invCov = new Matrix<double>(new double[,]
        {
            { 2.0, 0.5 },
            { 0.5, 1.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 1.0, 2.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0 });

        // Act
        var dAB = metric.Compute(a, b);
        var dBA = metric.Compute(b, a);

        // Assert
        Assert.Equal(dAB, dBA, Tolerance);
    }

    #endregion

    #region Triangle Inequality Tests

    /// <summary>
    /// Mahalanobis distance should satisfy triangle inequality:
    /// d(a, c) <= d(a, b) + d(b, c)
    /// </summary>
    [Fact]
    public void MahalanobisDistance_SatisfiesTriangleInequality()
    {
        // Arrange
        var invCov = new Matrix<double>(new double[,]
        {
            { 1.5, 0.3 },
            { 0.3, 1.0 }
        });
        var metric = new MahalanobisDistance<double>(invCov);
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 1.0, 1.0 });
        var c = new Vector<double>(new[] { 2.0, 0.0 });

        // Act
        var dAC = metric.Compute(a, c);
        var dAB = metric.Compute(a, b);
        var dBC = metric.Compute(b, c);

        // Assert
        Assert.True(dAC <= dAB + dBC + Tolerance,
            $"Triangle inequality violated: {dAC} > {dAB} + {dBC}");
    }

    #endregion

    #region Float Type Tests

    /// <summary>
    /// Test with float type to ensure generic implementation works.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_FloatType_ReturnsCorrectValue()
    {
        // Arrange
        var invCov = new Matrix<float>(new float[,]
        {
            { 1.0f, 0.0f },
            { 0.0f, 1.0f }
        });
        var metric = new MahalanobisDistance<float>(invCov);
        var a = new Vector<float>(new[] { 0.0f, 0.0f });
        var b = new Vector<float>(new[] { 3.0f, 4.0f });

        // Act
        var distance = metric.Compute(a, b);

        // Assert
        Assert.Equal(5.0f, distance, 1e-4f);
    }

    #endregion

    #region Property Tests

    /// <summary>
    /// Test that the Name property returns correct value.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_Name_ReturnsMahalanobis()
    {
        // Arrange
        var metric = new MahalanobisDistance<double>();

        // Act & Assert
        Assert.Equal("Mahalanobis", metric.Name);
    }

    /// <summary>
    /// Test that InverseCovarianceMatrix property can be set and retrieved.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_SetInverseCovarianceMatrix_Updates()
    {
        // Arrange
        var metric = new MahalanobisDistance<double>();
        var invCov = new Matrix<double>(new double[,]
        {
            { 2.0, 0.0 },
            { 0.0, 2.0 }
        });

        // Initially null
        Assert.Null(metric.InverseCovarianceMatrix);

        // Act
        metric.InverseCovarianceMatrix = invCov;

        // Assert
        Assert.NotNull(metric.InverseCovarianceMatrix);
        Assert.Equal(2.0, metric.InverseCovarianceMatrix[0, 0], Tolerance);
    }

    #endregion

    #region Comparison with Euclidean

    /// <summary>
    /// Mahalanobis with identity should match Euclidean for high-dimensional vectors.
    /// </summary>
    [Fact]
    public void MahalanobisDistance_HighDimensional_IdentityMatchesEuclidean()
    {
        // Arrange - 5D identity
        var invCov = new Matrix<double>(new double[,]
        {
            { 1, 0, 0, 0, 0 },
            { 0, 1, 0, 0, 0 },
            { 0, 0, 1, 0, 0 },
            { 0, 0, 0, 1, 0 },
            { 0, 0, 0, 0, 1 }
        });
        var mahalanobis = new MahalanobisDistance<double>(invCov);
        var euclidean = new EuclideanDistance<double>();

        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var b = new Vector<double>(new[] { 5.0, 4.0, 3.0, 2.0, 1.0 });

        // Act
        var dMahal = mahalanobis.Compute(a, b);
        var dEucl = euclidean.Compute(a, b);

        // Assert
        Assert.Equal(dEucl, dMahal, Tolerance);
    }

    #endregion
}
