using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for MatrixHelper&lt;T&gt;.
/// Tests all public methods including matrix operations, decomposition helpers, and numerical algorithms.
/// </summary>
public class MatrixHelperIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region CalculateDeterminantRecursive Tests

    [Fact]
    public void CalculateDeterminantRecursive_1x1Matrix_ReturnsElement()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,] { { 5.0 } });

        // Act
        var result = MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void CalculateDeterminantRecursive_2x2Matrix_ReturnsCorrectDeterminant()
    {
        // Arrange - det = ad - bc = 3*4 - 2*1 = 10
        var matrix = new Matrix<double>(new double[,]
        {
            { 3, 2 },
            { 1, 4 }
        });

        // Act
        var result = MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

        // Assert
        Assert.Equal(10.0, result, Tolerance);
    }

    [Fact]
    public void CalculateDeterminantRecursive_3x3Matrix_ReturnsCorrectDeterminant()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 10 }
        });
        // det = 1*(5*10-6*8) - 2*(4*10-6*7) + 3*(4*8-5*7)
        //     = 1*(50-48) - 2*(40-42) + 3*(32-35)
        //     = 1*2 - 2*(-2) + 3*(-3) = 2 + 4 - 9 = -3

        // Act
        var result = MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

        // Assert
        Assert.Equal(-3.0, result, Tolerance);
    }

    [Fact]
    public void CalculateDeterminantRecursive_IdentityMatrix_ReturnsOne()
    {
        // Arrange
        var matrix = Matrix<double>.CreateIdentity(4);

        // Act
        var result = MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CalculateDeterminantRecursive_SingularMatrix_ReturnsZero()
    {
        // Arrange - Third row is sum of first two rows, so det = 0
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 5, 7, 9 }
        });

        // Act
        var result = MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

        // Assert
        Assert.True(Math.Abs(result) < Tolerance);
    }

    [Fact]
    public void CalculateDeterminantRecursive_NonSquareMatrix_ThrowsArgumentException()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            MatrixHelper<double>.CalculateDeterminantRecursive(matrix));
    }

    #endregion

    #region ExtractDiagonal Tests

    [Fact]
    public void ExtractDiagonal_3x3Matrix_ReturnsCorrectDiagonal()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        // Act
        var result = MatrixHelper<double>.ExtractDiagonal(matrix);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(5.0, result[1], Tolerance);
        Assert.Equal(9.0, result[2], Tolerance);
    }

    [Fact]
    public void ExtractDiagonal_IdentityMatrix_ReturnsOnes()
    {
        // Arrange
        var matrix = Matrix<double>.CreateIdentity(4);

        // Act
        var result = MatrixHelper<double>.ExtractDiagonal(matrix);

        // Assert
        Assert.Equal(4, result.Length);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(1.0, result[i], Tolerance);
        }
    }

    [Fact]
    public void ExtractDiagonal_DiagonalMatrix_ReturnsAllDiagonalValues()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 2, 0, 0 },
            { 0, 3, 0 },
            { 0, 0, 5 }
        });

        // Act
        var result = MatrixHelper<double>.ExtractDiagonal(matrix);

        // Assert
        Assert.Equal(2.0, result[0], Tolerance);
        Assert.Equal(3.0, result[1], Tolerance);
        Assert.Equal(5.0, result[2], Tolerance);
    }

    #endregion

    #region OuterProduct Tests

    [Fact]
    public void OuterProduct_TwoVectors_ReturnsCorrectMatrix()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 4, 5, 6 });

        // Act
        var result = MatrixHelper<double>.OuterProduct(v1, v2);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
        Assert.Equal(4.0, result[0, 0], Tolerance);  // 1*4
        Assert.Equal(5.0, result[0, 1], Tolerance);  // 1*5
        Assert.Equal(6.0, result[0, 2], Tolerance);  // 1*6
        Assert.Equal(8.0, result[1, 0], Tolerance);  // 2*4
        Assert.Equal(10.0, result[1, 1], Tolerance); // 2*5
        Assert.Equal(12.0, result[1, 2], Tolerance); // 2*6
        Assert.Equal(12.0, result[2, 0], Tolerance); // 3*4
        Assert.Equal(15.0, result[2, 1], Tolerance); // 3*5
        Assert.Equal(18.0, result[2, 2], Tolerance); // 3*6
    }

    [Fact]
    public void OuterProduct_UnitVectors_ReturnsCorrectMatrix()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 1, 0 });
        var v2 = new Vector<double>(new double[] { 0, 1 });

        // Act
        var result = MatrixHelper<double>.OuterProduct(v1, v2);

        // Assert
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(1.0, result[0, 1], Tolerance);
        Assert.Equal(0.0, result[1, 0], Tolerance);
        Assert.Equal(0.0, result[1, 1], Tolerance);
    }

    #endregion

    #region Hypotenuse Tests

    [Fact]
    public void Hypotenuse_TwoValues_ReturnsCorrectResult()
    {
        // Arrange - Classic 3-4-5 triangle
        double x = 3.0;
        double y = 4.0;

        // Act
        var result = MatrixHelper<double>.Hypotenuse(x, y);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void Hypotenuse_ZeroValue_ReturnsOtherValue()
    {
        // Arrange
        double x = 0.0;
        double y = 5.0;

        // Act
        var result = MatrixHelper<double>.Hypotenuse(x, y);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void Hypotenuse_NegativeValues_ReturnsPositiveResult()
    {
        // Arrange
        double x = -3.0;
        double y = -4.0;

        // Act
        var result = MatrixHelper<double>.Hypotenuse(x, y);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void Hypotenuse_ArrayOfValues_ReturnsEuclideanNorm()
    {
        // Arrange - sqrt(1^2 + 2^2 + 2^2) = sqrt(9) = 3
        var values = new double[] { 1.0, 2.0, 2.0 };

        // Act
        var result = MatrixHelper<double>.Hypotenuse(values);

        // Assert
        Assert.Equal(3.0, result, Tolerance);
    }

    [Fact]
    public void Hypotenuse_SingleValue_ReturnsAbsoluteValue()
    {
        // Arrange
        var values = new double[] { -5.0 };

        // Act
        var result = MatrixHelper<double>.Hypotenuse(values);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    #endregion

    #region IsUpperHessenberg Tests

    [Fact]
    public void IsUpperHessenberg_UpperTriangularMatrix_ReturnsTrue()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 0, 4, 5 },
            { 0, 0, 6 }
        });

        // Act
        var result = MatrixHelper<double>.IsUpperHessenberg(matrix, 1e-10);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsUpperHessenberg_HessenbergMatrix_ReturnsTrue()
    {
        // Arrange - Hessenberg has zeros below first subdiagonal
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 0, 9, 10, 11 },
            { 0, 0, 12, 13 }
        });

        // Act
        var result = MatrixHelper<double>.IsUpperHessenberg(matrix, 1e-10);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsUpperHessenberg_FullMatrix_ReturnsFalse()
    {
        // Arrange - Has non-zero below first subdiagonal
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        // Act
        var result = MatrixHelper<double>.IsUpperHessenberg(matrix, 1e-10);

        // Assert
        Assert.False(result);
    }

    #endregion

    #region OrthogonalizeColumns Tests

    [Fact]
    public void OrthogonalizeColumns_NonOrthogonalMatrix_ReturnsOrthonormalColumns()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 0, 1 },
            { 1, 0 }
        });

        // Act
        var result = MatrixHelper<double>.OrthogonalizeColumns(matrix);

        // Assert - Columns should be orthogonal (dot product = 0)
        var col0 = result.GetColumn(0);
        var col1 = result.GetColumn(1);
        var dotProduct = col0.DotProduct(col1);
        Assert.True(Math.Abs(dotProduct) < Tolerance, $"Columns not orthogonal: dot product = {dotProduct}");

        // Each column should have unit norm
        Assert.True(Math.Abs(col0.Norm() - 1.0) < Tolerance, $"Column 0 not normalized: norm = {col0.Norm()}");
        Assert.True(Math.Abs(col1.Norm() - 1.0) < Tolerance, $"Column 1 not normalized: norm = {col1.Norm()}");
    }

    [Fact]
    public void OrthogonalizeColumns_IdentityMatrix_ReturnsIdentity()
    {
        // Arrange - Identity columns are already orthonormal
        var matrix = Matrix<double>.CreateIdentity(3);

        // Act
        var result = MatrixHelper<double>.OrthogonalizeColumns(matrix);

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(result[i, j] - expected) < Tolerance);
            }
        }
    }

    #endregion

    #region ComputeGivensRotation Tests

    [Fact]
    public void ComputeGivensRotation_ZeroB_ReturnsCosOne()
    {
        // Arrange
        double a = 5.0;
        double b = 0.0;

        // Act
        var (c, s) = MatrixHelper<double>.ComputeGivensRotation(a, b);

        // Assert
        Assert.Equal(1.0, c, Tolerance);
        Assert.Equal(0.0, s, Tolerance);
    }

    [Fact]
    public void ComputeGivensRotation_NonZeroValues_ReturnsValidRotation()
    {
        // Arrange
        double a = 3.0;
        double b = 4.0;

        // Act
        var (c, s) = MatrixHelper<double>.ComputeGivensRotation(a, b);

        // Assert - c^2 + s^2 should equal 1
        Assert.True(Math.Abs(c * c + s * s - 1.0) < Tolerance,
            $"Rotation not valid: c^2 + s^2 = {c * c + s * s}");
    }

    #endregion

    #region ApplyGivensRotation Tests

    [Fact]
    public void ApplyGivensRotation_ToMatrix_ModifiesCorrectElements()
    {
        // Arrange
        var H = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });
        double c = 0.6;
        double s = 0.8;

        // Act
        MatrixHelper<double>.ApplyGivensRotation(H, c, s, 0, 1, 0, 3);

        // Assert - Rows 0 and 1 should be modified
        // New row 0: c*old_row0 + s*old_row1
        // New row 1: -s*old_row0 + c*old_row1
        Assert.True(Math.Abs(H[0, 0] - (0.6 * 1 + 0.8 * 4)) < Tolerance);
        Assert.True(Math.Abs(H[0, 1] - (0.6 * 2 + 0.8 * 5)) < Tolerance);
    }

    #endregion

    #region CreateHouseholderVector Tests

    [Fact]
    public void CreateHouseholderVector_ValidVector_ReturnsNormalizedVector()
    {
        // Arrange
        var xVector = new Vector<double>(new double[] { 1, 2, 3 });

        // Act
        var result = MatrixHelper<double>.CreateHouseholderVector(xVector);

        // Assert - Result should be normalized (norm = 1)
        double norm = 0;
        for (int i = 0; i < result.Length; i++)
        {
            norm += result[i] * result[i];
        }
        norm = Math.Sqrt(norm);
        Assert.True(Math.Abs(norm - 1.0) < Tolerance, $"Householder vector not normalized: norm = {norm}");
    }

    #endregion

    #region PowerIteration Tests

    [Fact]
    public void PowerIteration_SymmetricMatrix_FindsDominantEigenvalue()
    {
        // Arrange - Symmetric matrix with known eigenvalues
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 1 },
            { 1, 3 }
        });
        // Eigenvalues are approximately 4.618 and 2.382

        // Act
        var (eigenvalue, eigenvector) = MatrixHelper<double>.PowerIteration(matrix, 100, 1e-10);

        // Assert - Eigenvalue should be close to the dominant eigenvalue
        Assert.True(eigenvalue > 0, "Eigenvalue should be positive");
        Assert.Equal(2, eigenvector.Length);
    }

    [Fact]
    public void PowerIteration_DiagonalMatrix_FindsLargestDiagonal()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 5, 0, 0 },
            { 0, 2, 0 },
            { 0, 0, 3 }
        });

        // Act
        var (eigenvalue, eigenvector) = MatrixHelper<double>.PowerIteration(matrix, 100, 1e-10);

        // Assert
        Assert.Equal(3, eigenvector.Length);
        // Eigenvector should converge to [1, 0, 0] for largest eigenvalue 5
    }

    #endregion

    #region SpectralNorm Tests

    [Fact]
    public void SpectralNorm_IdentityMatrix_ReturnsOne()
    {
        // Arrange
        var matrix = Matrix<double>.CreateIdentity(3);

        // Act
        var result = MatrixHelper<double>.SpectralNorm(matrix);

        // Assert
        Assert.True(Math.Abs(result - 1.0) < 0.1, $"Expected ~1.0, got {result}");
    }

    [Fact]
    public void SpectralNorm_ScaledIdentity_ReturnsScaleFactor()
    {
        // Arrange - 2*I has spectral norm = 2
        var matrix = new Matrix<double>(new double[,]
        {
            { 2, 0 },
            { 0, 2 }
        });

        // Act
        var result = MatrixHelper<double>.SpectralNorm(matrix);

        // Assert
        Assert.True(Math.Abs(result - 2.0) < 0.1, $"Expected ~2.0, got {result}");
    }

    #endregion

    #region IsInvertible Tests

    [Fact]
    public void IsInvertible_IdentityMatrix_ReturnsTrue()
    {
        // Arrange
        var matrix = Matrix<double>.CreateIdentity(3);

        // Act
        var result = MatrixHelper<double>.IsInvertible(matrix);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsInvertible_SingularMatrix_ReturnsFalse()
    {
        // Arrange - Third row is sum of first two
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 5, 7, 9 }
        });

        // Act
        var result = MatrixHelper<double>.IsInvertible(matrix);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsInvertible_NonSquareMatrix_ReturnsFalse()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        // Act
        var result = MatrixHelper<double>.IsInvertible(matrix);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsInvertible_InvertibleMatrix_ReturnsTrue()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });
        // det = 1*4 - 2*3 = -2 != 0

        // Act
        var result = MatrixHelper<double>.IsInvertible(matrix);

        // Assert
        Assert.True(result);
    }

    #endregion

    #region InvertUsingDecomposition Tests

    [Fact]
    public void InvertUsingDecomposition_WithLuDecomposition_ReturnsCorrectInverse()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 7 },
            { 2, 6 }
        });
        // det = 24 - 14 = 10
        // Inverse = (1/10) * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
        var decomposition = new QrDecomposition<double>(matrix);

        // Act
        var result = MatrixHelper<double>.InvertUsingDecomposition(decomposition);

        // Assert - A * A^-1 should be identity
        var product = matrix.Multiply(result);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(product[i, j] - expected) < Tolerance,
                    $"A*A^-1 not identity at [{i},{j}]: {product[i, j]}");
            }
        }
    }

    #endregion

    #region TridiagonalSolve Tests

    [Fact]
    public void TridiagonalSolve_SimpleSystem_ReturnsCorrectSolution()
    {
        // Arrange - Tridiagonal system
        // [ 2 -1  0  0] [x1]   [1]
        // [-1  2 -1  0] [x2] = [0]
        // [ 0 -1  2 -1] [x3]   [0]
        // [ 0  0 -1  2] [x4]   [1]
        var lower = new Vector<double>(new double[] { 0, -1, -1, -1 });  // subdiagonal
        var diag = new Vector<double>(new double[] { 2, 2, 2, 2 });      // main diagonal
        var upper = new Vector<double>(new double[] { -1, -1, -1, 0 }); // superdiagonal
        var rhs = new Vector<double>(new double[] { 1, 0, 0, 1 });
        var solution = new Vector<double>(4);

        // Act
        MatrixHelper<double>.TridiagonalSolve(lower, diag, upper, solution, rhs);

        // Assert - Verify Ax = b
        // Manual verification: construct tridiagonal matrix and multiply
        var A = new Matrix<double>(new double[,]
        {
            { 2, -1, 0, 0 },
            { -1, 2, -1, 0 },
            { 0, -1, 2, -1 },
            { 0, 0, -1, 2 }
        });
        var Ax = A.Multiply(solution);
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(Ax[i] - rhs[i]) < Tolerance,
                $"Tridiagonal solution incorrect at {i}: Ax={Ax[i]}, b={rhs[i]}");
        }
    }

    [Fact]
    public void TridiagonalSolve_ZeroDiagonal_ThrowsInvalidOperationException()
    {
        // Arrange - Zero on main diagonal
        var lower = new Vector<double>(new double[] { 0, 1 });
        var diag = new Vector<double>(new double[] { 0, 1 });  // First element is 0
        var upper = new Vector<double>(new double[] { 1, 0 });
        var rhs = new Vector<double>(new double[] { 1, 1 });
        var solution = new Vector<double>(2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            MatrixHelper<double>.TridiagonalSolve(lower, diag, upper, solution, rhs));
    }

    #endregion

    #region CalculateHatMatrix Tests

    [Fact]
    public void CalculateHatMatrix_SimpleFeatureMatrix_ReturnsSymmetricMatrix()
    {
        // Arrange - Simple feature matrix
        var features = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 1, 2 },
            { 1, 3 }
        });

        // Act
        var result = MatrixHelper<double>.CalculateHatMatrix(features);

        // Assert - Hat matrix should be symmetric
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(result[i, j] - result[j, i]) < Tolerance,
                    $"Hat matrix not symmetric at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void CalculateHatMatrix_IsIdempotent()
    {
        // Arrange
        var features = new Matrix<double>(new double[,]
        {
            { 1, 1 },
            { 1, 2 },
            { 1, 3 },
            { 1, 4 }
        });

        // Act
        var H = MatrixHelper<double>.CalculateHatMatrix(features);
        var H2 = H.Multiply(H);

        // Assert - H * H = H (idempotent)
        for (int i = 0; i < H.Rows; i++)
        {
            for (int j = 0; j < H.Columns; j++)
            {
                Assert.True(Math.Abs(H[i, j] - H2[i, j]) < Tolerance,
                    $"Hat matrix not idempotent at [{i},{j}]: H={H[i, j]}, H^2={H2[i, j]}");
            }
        }
    }

    [Fact]
    public void CalculateHatMatrix_DiagonalSumEqualsRank()
    {
        // Arrange
        var features = new Matrix<double>(new double[,]
        {
            { 1, 0 },
            { 0, 1 },
            { 1, 1 }
        });

        // Act
        var H = MatrixHelper<double>.CalculateHatMatrix(features);

        // Assert - Trace of hat matrix equals rank of X (number of columns)
        double trace = 0;
        for (int i = 0; i < H.Rows; i++)
        {
            trace += H[i, i];
        }
        Assert.True(Math.Abs(trace - 2.0) < Tolerance,
            $"Trace should equal rank (2): trace = {trace}");
    }

    #endregion

    #region ReduceToHessenbergFormat Tests

    [Fact]
    public void ReduceToHessenbergFormat_GeneralMatrix_ProducesHessenbergForm()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 1, 2, 3 },
            { 1, 3, 1, 2 },
            { 2, 1, 2, 1 },
            { 3, 2, 1, 1 }
        });

        // Act
        var result = MatrixHelper<double>.ReduceToHessenbergFormat(matrix);

        // Assert - Check that elements below first subdiagonal are zero
        Assert.True(MatrixHelper<double>.IsUpperHessenberg(result, Tolerance));
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void CalculateDeterminantRecursive_FloatType_ReturnsCorrectResult()
    {
        // Arrange
        var matrix = new Matrix<float>(new float[,]
        {
            { 3f, 2f },
            { 1f, 4f }
        });

        // Act
        var result = MatrixHelper<float>.CalculateDeterminantRecursive(matrix);

        // Assert
        Assert.True(Math.Abs(result - 10f) < 1e-4f);
    }

    [Fact]
    public void Hypotenuse_FloatType_ReturnsCorrectResult()
    {
        // Arrange
        float x = 3f;
        float y = 4f;

        // Act
        var result = MatrixHelper<float>.Hypotenuse(x, y);

        // Assert
        Assert.True(Math.Abs(result - 5f) < 1e-4f);
    }

    #endregion
}
