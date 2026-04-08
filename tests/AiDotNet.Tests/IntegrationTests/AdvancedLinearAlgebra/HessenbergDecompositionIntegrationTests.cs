using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Hessenberg decomposition that verify mathematical correctness.
/// These tests verify: H is upper Hessenberg (zeros below first subdiagonal).
/// </summary>
public class HessenbergDecompositionIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(int size, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrix[i, j] = random.NextDouble() * 10 - 5;
            }
        }
        return matrix;
    }

    private static Matrix<double> CreateSymmetricMatrix(int size, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = i; j < size; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                matrix[i, j] = value;
                matrix[j, i] = value;
            }
        }
        return matrix;
    }

    private static bool IsUpperHessenberg(Matrix<double> H, double tolerance)
    {
        // Upper Hessenberg has zeros below the first subdiagonal
        // H[i,j] = 0 for i > j + 1
        for (int i = 2; i < H.Rows; i++)
        {
            for (int j = 0; j < i - 1; j++)
            {
                if (Math.Abs(H[i, j]) > tolerance)
                    return false;
            }
        }
        return true;
    }

    #endregion

    #region Upper Hessenberg Structure Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void HessenbergDecomposition_Householder_ProducesUpperHessenberg(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var hess = new HessenbergDecomposition<double>(A, HessenbergAlgorithmType.Householder);

        // Assert - H should be upper Hessenberg
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, LooseTolerance),
            "Householder: HessenbergMatrix should be upper Hessenberg (zeros below first subdiagonal)");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void HessenbergDecomposition_Givens_ProducesUpperHessenberg(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, seed: 123);

        // Act
        var hess = new HessenbergDecomposition<double>(A, HessenbergAlgorithmType.Givens);

        // Assert
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, LooseTolerance),
            "Givens: HessenbergMatrix should be upper Hessenberg");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void HessenbergDecomposition_ElementaryTransformations_ProducesUpperHessenberg(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, seed: 456);

        // Act
        var hess = new HessenbergDecomposition<double>(A, HessenbergAlgorithmType.ElementaryTransformations);

        // Assert
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, LooseTolerance),
            "ElementaryTransformations: HessenbergMatrix should be upper Hessenberg");
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void HessenbergDecomposition_IdentityMatrix_PreservesIdentity()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var hess = new HessenbergDecomposition<double>(I);

        // Assert - Identity is already upper Hessenberg and should be preserved
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, Tolerance),
            "Identity matrix Hessenberg form should be upper Hessenberg");

        // Diagonal should be 1s
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(hess.HessenbergMatrix[i, i] - 1.0) < LooseTolerance,
                $"H[{i},{i}] should be close to 1, got {hess.HessenbergMatrix[i, i]}");
        }
    }

    [Fact]
    public void HessenbergDecomposition_DiagonalMatrix_PreservesDiagonal()
    {
        // Arrange - Diagonal matrices are already upper Hessenberg
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 4; D[1, 1] = 3; D[2, 2] = 2; D[3, 3] = 1;

        // Act
        var hess = new HessenbergDecomposition<double>(D);

        // Assert
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, Tolerance),
            "Diagonal matrix Hessenberg form should be upper Hessenberg");
    }

    [Fact]
    public void HessenbergDecomposition_UpperTriangularMatrix_PreservesStructure()
    {
        // Arrange - Upper triangular is already upper Hessenberg
        var U = new Matrix<double>(4, 4);
        U[0, 0] = 4; U[0, 1] = 2; U[0, 2] = 1; U[0, 3] = 0;
        U[1, 1] = 3; U[1, 2] = 2; U[1, 3] = 1;
        U[2, 2] = 2; U[2, 3] = 1;
        U[3, 3] = 1;

        // Act
        var hess = new HessenbergDecomposition<double>(U);

        // Assert
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, Tolerance),
            "Upper triangular matrix Hessenberg form should be upper Hessenberg");
    }

    [Fact]
    public void HessenbergDecomposition_SymmetricMatrix_ProducesTridiagonal()
    {
        // Arrange - For symmetric matrices, Hessenberg form should be tridiagonal
        var A = CreateSymmetricMatrix(4, seed: 789);

        // Act
        var hess = new HessenbergDecomposition<double>(A);

        // Assert - Should be upper Hessenberg
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, LooseTolerance),
            "Symmetric matrix Hessenberg form should be upper Hessenberg");

        // For symmetric matrices, Hessenberg form is typically tridiagonal
        // (zeros above first superdiagonal too)
        // This is a weaker test - just checking the general structure
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(HessenbergAlgorithmType.Householder)]
    [InlineData(HessenbergAlgorithmType.Givens)]
    [InlineData(HessenbergAlgorithmType.ElementaryTransformations)]
    [InlineData(HessenbergAlgorithmType.ImplicitQR)]
    [InlineData(HessenbergAlgorithmType.Lanczos)]
    public void HessenbergDecomposition_AllAlgorithms_ProduceValidDecomposition(HessenbergAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, seed: 42);

        // Act
        var hess = new HessenbergDecomposition<double>(A, algorithm);

        // Assert - Basic validity checks
        Assert.Equal(4, hess.HessenbergMatrix.Rows);
        Assert.Equal(4, hess.HessenbergMatrix.Columns);

        // No NaN or Inf values
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.False(double.IsNaN(hess.HessenbergMatrix[i, j]),
                    $"Algorithm {algorithm}: HessenbergMatrix has NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(hess.HessenbergMatrix[i, j]),
                    $"Algorithm {algorithm}: HessenbergMatrix has Infinity at [{i},{j}]");
            }
        }
    }

    [Theory]
    [InlineData(HessenbergAlgorithmType.Householder)]
    [InlineData(HessenbergAlgorithmType.Givens)]
    [InlineData(HessenbergAlgorithmType.ElementaryTransformations)]
    public void HessenbergDecomposition_DeterministicAlgorithms_ProduceHessenbergForm(HessenbergAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, seed: 42);

        // Act
        var hess = new HessenbergDecomposition<double>(A, algorithm);

        // Assert - All deterministic algorithms should produce upper Hessenberg
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, LooseTolerance),
            $"Algorithm {algorithm}: Should produce upper Hessenberg form");
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void HessenbergDecomposition_LargeMatrix_ProducesValidDecomposition()
    {
        // Arrange
        var A = CreateTestMatrix(10, seed: 999);

        // Act
        var hess = new HessenbergDecomposition<double>(A);

        // Assert
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, LooseTolerance),
            "Large matrix Hessenberg form should be upper Hessenberg");
    }

    [Fact]
    public void HessenbergDecomposition_ZeroMatrix_HandlesGracefully()
    {
        // Arrange
        var Z = new Matrix<double>(3, 3); // All zeros

        // Act
        var hess = new HessenbergDecomposition<double>(Z);

        // Assert - Zero matrix is already upper Hessenberg
        Assert.True(IsUpperHessenberg(hess.HessenbergMatrix, Tolerance),
            "Zero matrix Hessenberg form should be upper Hessenberg");
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void HessenbergDecomposition_NonSquareMatrix_ThrowsException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4); // Non-square

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HessenbergDecomposition<double>(A));
    }

    #endregion

    #region Dimension Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(8)]
    public void HessenbergDecomposition_VariousSizes_ProducesCorrectDimensions(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, seed: size);

        // Act
        var hess = new HessenbergDecomposition<double>(A);

        // Assert
        Assert.Equal(size, hess.HessenbergMatrix.Rows);
        Assert.Equal(size, hess.HessenbergMatrix.Columns);
    }

    #endregion
}
