using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for MatrixSolutionHelper.
/// Tests all public methods and all supported decomposition types.
/// </summary>
public class MatrixSolutionHelperIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region SolveLinearSystem with DecompositionType Tests

    [Fact]
    public void SolveLinearSystem_WithLuDecomposition_ReturnsCorrectSolution()
    {
        // Arrange - Simple 3x3 system: Ax = b
        // 2x + y - z = 8
        // -3x - y + 2z = -11
        // -2x + y + 2z = -3
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 }
        });
        var b = new Vector<double>(new double[] { 8, -11, -3 });
        // Expected solution: x = 2, y = 3, z = -1

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lu);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.True(Math.Abs(result[0] - 2.0) < Tolerance, $"Expected x=2, got {result[0]}");
        Assert.True(Math.Abs(result[1] - 3.0) < Tolerance, $"Expected y=3, got {result[1]}");
        Assert.True(Math.Abs(result[2] - (-1.0)) < Tolerance, $"Expected z=-1, got {result[2]}");
    }

    [Fact]
    public void SolveLinearSystem_WithCholeskyDecomposition_ReturnsCorrectSolution()
    {
        // Arrange - Symmetric positive-definite matrix (required for Cholesky)
        // A = [4 12 -16; 12 37 -43; -16 -43 98]
        var A = new Matrix<double>(new double[,]
        {
            { 4, 12, -16 },
            { 12, 37, -43 },
            { -16, -43, 98 }
        });
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Cholesky);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithQrDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 10 }
        });
        var b = new Vector<double>(new double[] { 6, 15, 25 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Qr);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithSvdDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 }
        });
        var b = new Vector<double>(new double[] { 8, -11, -3 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Svd);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithCramerDecomposition_ReturnsCorrectSolution()
    {
        // Arrange - Small system (Cramer's rule is O(n!) so only practical for small systems)
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1 },
            { 5, 7 }
        });
        var b = new Vector<double>(new double[] { 11, 13 });
        // Solution: x = 74/9, y = -23/9 (approximately 8.22, -2.56)
        // Wait, let me recalculate: 2x + y = 11, 5x + 7y = 13
        // From first: y = 11 - 2x
        // Substituting: 5x + 7(11 - 2x) = 13 => 5x + 77 - 14x = 13 => -9x = -64 => x = 64/9
        // y = 11 - 2(64/9) = 99/9 - 128/9 = -29/9

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Cramer);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithGramSchmidtDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 10 }
        });
        var b = new Vector<double>(new double[] { 6, 15, 25 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.GramSchmidt);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithNormalDecomposition_ReturnsCorrectSolution()
    {
        // Arrange - Symmetric positive-definite result after A^T*A
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Normal);

        // Assert - For overdetermined systems, verify least squares solution (A^T * A * x = A^T * b)
        var ATA = A.Transpose().Multiply(A);
        var ATb = A.Transpose().Multiply(b);
        var ATAx = ATA.Multiply(result);
        for (int i = 0; i < ATb.Length; i++)
        {
            Assert.True(Math.Abs(ATAx[i] - ATb[i]) < Tolerance,
                $"Normal equations not satisfied at index {i}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithLqDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 }
        });
        var b = new Vector<double>(new double[] { 8, -11, -3 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lq);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithHessenbergDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 }
        });
        var b = new Vector<double>(new double[] { 8, -11, -3 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Hessenberg);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithSchurDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 }
        });
        var b = new Vector<double>(new double[] { 8, -11, -3 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Schur);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithEigenDecomposition_ReturnsCorrectSolution()
    {
        // Arrange - Symmetric matrix for reliable eigenvalue decomposition
        var A = new Matrix<double>(new double[,]
        {
            { 4, 2, 2 },
            { 2, 5, 1 },
            { 2, 1, 6 }
        });
        var b = new Vector<double>(new double[] { 8, 8, 9 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Eigen);

        // Assert - Verify Ax = b
        // Eigenvalue decomposition is iterative and may have slightly lower precision
        const double EigenTolerance = 1e-4;
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < EigenTolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithTakagiDecomposition_ThrowsNotSupportedException()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });
        var b = new Vector<double>(new double[] { 1, 2 });

        // Act & Assert
        Assert.Throws<NotSupportedException>(() =>
            MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Takagi));
    }

    [Fact]
    public void SolveLinearSystem_WithUnsupportedDecomposition_ThrowsArgumentException()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });
        var b = new Vector<double>(new double[] { 1, 2 });
        var invalidType = (MatrixDecompositionType)999;

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            MatrixSolutionHelper.SolveLinearSystem(A, b, invalidType));
    }

    #endregion

    #region SolveLinearSystem with Pre-computed Decomposition Tests

    [Fact]
    public void SolveLinearSystem_WithPreComputedLuDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 }
        });
        var b = new Vector<double>(new double[] { 8, -11, -3 });
        var decomposition = new LuDecomposition<double>(A);

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(b, decomposition);

        // Assert
        Assert.True(Math.Abs(result[0] - 2.0) < Tolerance, $"Expected x=2, got {result[0]}");
        Assert.True(Math.Abs(result[1] - 3.0) < Tolerance, $"Expected y=3, got {result[1]}");
        Assert.True(Math.Abs(result[2] - (-1.0)) < Tolerance, $"Expected z=-1, got {result[2]}");
    }

    [Fact]
    public void SolveLinearSystem_WithPreComputedQrDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 10 }
        });
        var b = new Vector<double>(new double[] { 6, 15, 25 });
        var decomposition = new QrDecomposition<double>(A);

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(b, decomposition);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithPreComputedSvdDecomposition_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 }
        });
        var b = new Vector<double>(new double[] { 8, -11, -3 });
        var decomposition = new SvdDecomposition<double>(A);

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(b, decomposition);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_WithPreComputedCholeskyDecomposition_ReturnsCorrectSolution()
    {
        // Arrange - Symmetric positive-definite matrix
        var A = new Matrix<double>(new double[,]
        {
            { 4, 12, -16 },
            { 12, 37, -43 },
            { -16, -43, 98 }
        });
        var b = new Vector<double>(new double[] { 1, 2, 3 });
        var decomposition = new CholeskyDecomposition<double>(A);

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(b, decomposition);

        // Assert - Verify Ax = b
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_PreComputedDecomposition_SolvesMultipleSystems()
    {
        // Arrange - Test efficiency of pre-computed decomposition
        var A = new Matrix<double>(new double[,]
        {
            { 2, 1, -1 },
            { -3, -1, 2 },
            { -2, 1, 2 }
        });
        var decomposition = new LuDecomposition<double>(A);

        var b1 = new Vector<double>(new double[] { 8, -11, -3 });
        var b2 = new Vector<double>(new double[] { 1, 2, 3 });
        var b3 = new Vector<double>(new double[] { -5, 10, 7 });

        // Act
        var result1 = MatrixSolutionHelper.SolveLinearSystem(b1, decomposition);
        var result2 = MatrixSolutionHelper.SolveLinearSystem(b2, decomposition);
        var result3 = MatrixSolutionHelper.SolveLinearSystem(b3, decomposition);

        // Assert - Verify all solutions
        var Ax1 = A.Multiply(result1);
        var Ax2 = A.Multiply(result2);
        var Ax3 = A.Multiply(result3);

        for (int i = 0; i < b1.Length; i++)
        {
            Assert.True(Math.Abs(Ax1[i] - b1[i]) < Tolerance, $"Solution 1 incorrect at index {i}");
            Assert.True(Math.Abs(Ax2[i] - b2[i]) < Tolerance, $"Solution 2 incorrect at index {i}");
            Assert.True(Math.Abs(Ax3[i] - b3[i]) < Tolerance, $"Solution 3 incorrect at index {i}");
        }
    }

    #endregion

    #region Edge Cases and Special Matrix Tests

    [Fact]
    public void SolveLinearSystem_2x2Matrix_ReturnsCorrectSolution()
    {
        // Arrange - Simple 2x2 system
        var A = new Matrix<double>(new double[,]
        {
            { 3, 2 },
            { 1, 4 }
        });
        var b = new Vector<double>(new double[] { 5, 6 });
        // Solution: x = 0.8, y = 1.3

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lu);

        // Assert
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_4x4Matrix_ReturnsCorrectSolution()
    {
        // Arrange - 4x4 system
        var A = new Matrix<double>(new double[,]
        {
            { 4, 1, 2, 1 },
            { 1, 3, 1, 1 },
            { 2, 1, 5, 2 },
            { 1, 1, 2, 4 }
        });
        var b = new Vector<double>(new double[] { 8, 6, 10, 8 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lu);

        // Assert
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_DiagonalMatrix_ReturnsCorrectSolution()
    {
        // Arrange - Diagonal matrix (simplest case)
        var A = new Matrix<double>(new double[,]
        {
            { 2, 0, 0 },
            { 0, 3, 0 },
            { 0, 0, 4 }
        });
        var b = new Vector<double>(new double[] { 4, 9, 16 });
        // Expected: x = 2, y = 3, z = 4

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lu);

        // Assert
        Assert.True(Math.Abs(result[0] - 2.0) < Tolerance);
        Assert.True(Math.Abs(result[1] - 3.0) < Tolerance);
        Assert.True(Math.Abs(result[2] - 4.0) < Tolerance);
    }

    [Fact]
    public void SolveLinearSystem_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange - Identity matrix: Ix = b => x = b
        var A = Matrix<double>.CreateIdentity(3);
        var b = new Vector<double>(new double[] { 1, 2, 3 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lu);

        // Assert
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(result[i] - b[i]) < Tolerance,
                $"For identity matrix, result should equal b at index {i}");
        }
    }

    [Fact]
    public void SolveLinearSystem_TridiagonalMatrix_ReturnsCorrectSolution()
    {
        // Arrange - Tridiagonal matrix (common in numerical methods)
        var A = new Matrix<double>(new double[,]
        {
            { 4, 1, 0, 0 },
            { 1, 4, 1, 0 },
            { 0, 1, 4, 1 },
            { 0, 0, 1, 4 }
        });
        var b = new Vector<double>(new double[] { 1, 2, 3, 4 });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lu);

        // Assert
        var Ax = A.Multiply(result);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < Tolerance,
                $"Ax[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void SolveLinearSystem_AllDecompositions_ProduceSimilarResults()
    {
        // Arrange - Test that all decompositions give the same solution for a well-conditioned matrix
        var A = new Matrix<double>(new double[,]
        {
            { 4, 2, 2 },
            { 2, 5, 1 },
            { 2, 1, 6 }
        });
        var b = new Vector<double>(new double[] { 8, 8, 9 });

        // Act - Get solutions from different methods
        var luSolution = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lu);
        var qrSolution = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Qr);
        var svdSolution = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Svd);

        // Assert - All solutions should be very close
        for (int i = 0; i < b.Length; i++)
        {
            Assert.True(Math.Abs(luSolution[i] - qrSolution[i]) < Tolerance,
                $"LU and QR solutions differ at index {i}");
            Assert.True(Math.Abs(luSolution[i] - svdSolution[i]) < Tolerance,
                $"LU and SVD solutions differ at index {i}");
        }
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void SolveLinearSystem_WithFloatType_ReturnsCorrectSolution()
    {
        // Arrange
        var A = new Matrix<float>(new float[,]
        {
            { 2f, 1f, -1f },
            { -3f, -1f, 2f },
            { -2f, 1f, 2f }
        });
        var b = new Vector<float>(new float[] { 8f, -11f, -3f });

        // Act
        var result = MatrixSolutionHelper.SolveLinearSystem(A, b, MatrixDecompositionType.Lu);

        // Assert - With float, use larger tolerance
        const float floatTolerance = 1e-4f;
        Assert.True(Math.Abs(result[0] - 2f) < floatTolerance, $"Expected x=2, got {result[0]}");
        Assert.True(Math.Abs(result[1] - 3f) < floatTolerance, $"Expected y=3, got {result[1]}");
        Assert.True(Math.Abs(result[2] - (-1f)) < floatTolerance, $"Expected z=-1, got {result[2]}");
    }

    #endregion
}
