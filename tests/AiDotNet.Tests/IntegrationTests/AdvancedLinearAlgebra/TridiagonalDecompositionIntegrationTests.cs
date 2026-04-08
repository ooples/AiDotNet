using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Tridiagonal decomposition that verify mathematical correctness.
/// These tests verify: A = Q*T*Q^T, Q is orthogonal, T is tridiagonal.
/// </summary>
public class TridiagonalDecompositionIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    #region Helper Methods

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

    private static double MaxAbsDiff(Matrix<double> a, Matrix<double> b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return double.MaxValue;

        double maxDiff = 0;
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(a[i, j] - b[i, j]));
            }
        }
        return maxDiff;
    }

    private static bool IsOrthogonal(Matrix<double> Q, double tolerance)
    {
        var QtQ = Q.Transpose().Multiply(Q);
        int n = QtQ.Rows;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                if (Math.Abs(QtQ[i, j] - expected) > tolerance)
                    return false;
            }
        }
        return true;
    }

    private static bool IsTridiagonal(Matrix<double> T, double tolerance)
    {
        // Tridiagonal matrix has non-zero elements only on main diagonal,
        // subdiagonal, and superdiagonal
        for (int i = 0; i < T.Rows; i++)
        {
            for (int j = 0; j < T.Columns; j++)
            {
                bool isOnDiagonal = (i == j);
                bool isOnSubdiagonal = (i == j + 1);
                bool isOnSuperdiagonal = (j == i + 1);

                if (!isOnDiagonal && !isOnSubdiagonal && !isOnSuperdiagonal)
                {
                    if (Math.Abs(T[i, j]) > tolerance)
                        return false;
                }
            }
        }
        return true;
    }

    #endregion

    #region Reconstruction Tests (A = Q*T*Q^T)

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void TridiagonalDecomposition_Householder_Reconstruction(int size)
    {
        // Arrange - Tridiagonal decomposition works on symmetric matrices
        var A = CreateSymmetricMatrix(size);

        // Act
        var tridiag = new TridiagonalDecomposition<double>(A, TridiagonalAlgorithmType.Householder);

        // Assert - Verify A = Q*T*Q^T
        var reconstructed = tridiag.QMatrix
            .Multiply(tridiag.TMatrix)
            .Multiply(tridiag.QMatrix.Transpose());
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal Q*T*Q^T. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void TridiagonalDecomposition_Givens_Reconstruction(int size)
    {
        // Arrange
        var A = CreateSymmetricMatrix(size, seed: 123);

        // Act
        var tridiag = new TridiagonalDecomposition<double>(A, TridiagonalAlgorithmType.Givens);

        // Assert
        var reconstructed = tridiag.QMatrix
            .Multiply(tridiag.TMatrix)
            .Multiply(tridiag.QMatrix.Transpose());
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Givens: A should equal Q*T*Q^T. Max difference: {maxDiff}");
    }

    #endregion

    #region Orthogonality Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void TridiagonalDecomposition_QMatrix_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateSymmetricMatrix(size);

        // Act
        var tridiag = new TridiagonalDecomposition<double>(A);

        // Assert - Q^T * Q should be identity
        Assert.True(IsOrthogonal(tridiag.QMatrix, LooseTolerance),
            "Q should be orthogonal (Q^T * Q = I)");
    }

    #endregion

    #region Tridiagonal Structure Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void TridiagonalDecomposition_TMatrix_IsTridiagonal(int size)
    {
        // Arrange
        var A = CreateSymmetricMatrix(size);

        // Act
        var tridiag = new TridiagonalDecomposition<double>(A);

        // Assert - T should be tridiagonal
        Assert.True(IsTridiagonal(tridiag.TMatrix, LooseTolerance),
            "T should be tridiagonal (non-zero only on main, sub, and super diagonals)");
    }

    [Fact]
    public void TridiagonalDecomposition_SymmetricMatrix_TMatrixIsSymmetric()
    {
        // Arrange
        var A = CreateSymmetricMatrix(4, seed: 789);

        // Act
        var tridiag = new TridiagonalDecomposition<double>(A);

        // Assert - T should also be symmetric for symmetric input
        double maxDiff = MaxAbsDiff(tridiag.TMatrix, tridiag.TMatrix.Transpose());
        Assert.True(maxDiff < LooseTolerance,
            $"T should be symmetric for symmetric input. Max asymmetry: {maxDiff}");
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void TridiagonalDecomposition_IdentityMatrix_ValidDecomposition()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var tridiag = new TridiagonalDecomposition<double>(I);

        // Assert - T should be tridiagonal (identity is already tridiagonal)
        Assert.True(IsTridiagonal(tridiag.TMatrix, Tolerance),
            "Identity matrix T should be tridiagonal");
    }

    [Fact]
    public void TridiagonalDecomposition_DiagonalMatrix_ValidDecomposition()
    {
        // Arrange - Diagonal matrices are already tridiagonal
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 4; D[1, 1] = 3; D[2, 2] = 2; D[3, 3] = 1;

        // Act
        var tridiag = new TridiagonalDecomposition<double>(D);

        // Assert
        Assert.True(IsTridiagonal(tridiag.TMatrix, Tolerance),
            "Diagonal matrix T should be tridiagonal");
    }

    [Fact]
    public void TridiagonalDecomposition_ZeroMatrix_ValidDecomposition()
    {
        // Arrange
        var Z = new Matrix<double>(3, 3); // All zeros

        // Act
        var tridiag = new TridiagonalDecomposition<double>(Z);

        // Assert - Zero matrix is already tridiagonal
        Assert.True(IsTridiagonal(tridiag.TMatrix, Tolerance),
            "Zero matrix T should be tridiagonal");
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(TridiagonalAlgorithmType.Householder)]
    [InlineData(TridiagonalAlgorithmType.Givens)]
    [InlineData(TridiagonalAlgorithmType.Lanczos)]
    public void TridiagonalDecomposition_AllAlgorithms_ProduceValidDecomposition(TridiagonalAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateSymmetricMatrix(4, seed: 42);

        // Act
        var tridiag = new TridiagonalDecomposition<double>(A, algorithm);

        // Assert - Basic validity checks
        Assert.Equal(4, tridiag.QMatrix.Rows);
        Assert.Equal(4, tridiag.QMatrix.Columns);
        Assert.Equal(4, tridiag.TMatrix.Rows);
        Assert.Equal(4, tridiag.TMatrix.Columns);

        // No NaN or Inf values
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.False(double.IsNaN(tridiag.QMatrix[i, j]),
                    $"Algorithm {algorithm}: QMatrix has NaN at [{i},{j}]");
                Assert.False(double.IsNaN(tridiag.TMatrix[i, j]),
                    $"Algorithm {algorithm}: TMatrix has NaN at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void TridiagonalDecomposition_LargeMatrix_ValidDecomposition()
    {
        // Arrange
        var A = CreateSymmetricMatrix(10, seed: 999);

        // Act
        var tridiag = new TridiagonalDecomposition<double>(A);

        // Assert
        Assert.True(IsTridiagonal(tridiag.TMatrix, LooseTolerance),
            "Large matrix T should be tridiagonal");

        // Verify reconstruction
        var reconstructed = tridiag.QMatrix
            .Multiply(tridiag.TMatrix)
            .Multiply(tridiag.QMatrix.Transpose());
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance * 10, // Looser tolerance for larger matrices
            $"Large matrix reconstruction failed. Max difference: {maxDiff}");
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void TridiagonalDecomposition_NonSquareMatrix_ThrowsException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4); // Non-square

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new TridiagonalDecomposition<double>(A));
    }

    #endregion
}
