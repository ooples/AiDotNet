using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Bidiagonal decomposition that verify mathematical correctness.
/// These tests verify: A = U*B*V^T, U and V are orthogonal, B is bidiagonal.
/// </summary>
public class BidiagonalDecompositionIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(int rows, int cols, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = random.NextDouble() * 10 - 5;
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

    private static bool IsBidiagonal(Matrix<double> B, double tolerance)
    {
        // Bidiagonal matrix has non-zero elements only on main diagonal and superdiagonal
        for (int i = 0; i < B.Rows; i++)
        {
            for (int j = 0; j < B.Columns; j++)
            {
                bool isOnDiagonal = (i == j);
                bool isOnSuperdiagonal = (j == i + 1);

                if (!isOnDiagonal && !isOnSuperdiagonal)
                {
                    if (Math.Abs(B[i, j]) > tolerance)
                        return false;
                }
            }
        }
        return true;
    }

    #endregion

    #region Reconstruction Tests (A = U*B*V^T)

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 3)]
    [InlineData(3, 5)]
    public void BidiagonalDecomposition_Householder_Reconstruction(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A, BidiagonalAlgorithmType.Householder);

        // Assert - Verify A = U*B*V^T
        var reconstructed = bidiag.U.Multiply(bidiag.B).Multiply(bidiag.V.Transpose());
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal U*B*V^T. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void BidiagonalDecomposition_Givens_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size, seed: 123);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A, BidiagonalAlgorithmType.Givens);

        // Assert
        var reconstructed = bidiag.U.Multiply(bidiag.B).Multiply(bidiag.V.Transpose());
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Givens: A should equal U*B*V^T. Max difference: {maxDiff}");
    }

    #endregion

    #region Orthogonality Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void BidiagonalDecomposition_U_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A);

        // Assert - U^T * U should be identity
        Assert.True(IsOrthogonal(bidiag.U, LooseTolerance),
            "U should be orthogonal (U^T * U = I)");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void BidiagonalDecomposition_V_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A);

        // Assert - V^T * V should be identity
        Assert.True(IsOrthogonal(bidiag.V, LooseTolerance),
            "V should be orthogonal (V^T * V = I)");
    }

    #endregion

    #region Bidiagonal Structure Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void BidiagonalDecomposition_B_IsBidiagonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A);

        // Assert - B should be bidiagonal (main diagonal and superdiagonal only)
        Assert.True(IsBidiagonal(bidiag.B, LooseTolerance),
            "B should be bidiagonal (non-zero only on main diagonal and superdiagonal)");
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void BidiagonalDecomposition_IdentityMatrix_ValidDecomposition()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(I);

        // Assert - B should be bidiagonal
        Assert.True(IsBidiagonal(bidiag.B, Tolerance),
            "Identity matrix B should be bidiagonal");
    }

    [Fact]
    public void BidiagonalDecomposition_DiagonalMatrix_ValidDecomposition()
    {
        // Arrange
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 4; D[1, 1] = 3; D[2, 2] = 2; D[3, 3] = 1;

        // Act
        var bidiag = new BidiagonalDecomposition<double>(D);

        // Assert
        Assert.True(IsBidiagonal(bidiag.B, Tolerance),
            "Diagonal matrix B should be bidiagonal");

        // Verify reconstruction
        var reconstructed = bidiag.U.Multiply(bidiag.B).Multiply(bidiag.V.Transpose());
        double maxDiff = MaxAbsDiff(D, reconstructed);
        Assert.True(maxDiff < LooseTolerance,
            $"Diagonal matrix reconstruction failed. Max diff: {maxDiff}");
    }

    [Fact]
    public void BidiagonalDecomposition_ZeroMatrix_ValidDecomposition()
    {
        // Arrange
        var Z = new Matrix<double>(3, 3); // All zeros

        // Act
        var bidiag = new BidiagonalDecomposition<double>(Z);

        // Assert - B should be bidiagonal (all zeros is valid)
        Assert.True(IsBidiagonal(bidiag.B, Tolerance),
            "Zero matrix B should be bidiagonal");
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(BidiagonalAlgorithmType.Householder)]
    [InlineData(BidiagonalAlgorithmType.Givens)]
    [InlineData(BidiagonalAlgorithmType.Lanczos)]
    public void BidiagonalDecomposition_AllAlgorithms_ProduceValidDecomposition(BidiagonalAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, 4, seed: 42);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A, algorithm);

        // Assert - Basic validity checks
        Assert.Equal(4, bidiag.U.Rows);
        Assert.Equal(4, bidiag.B.Rows);
        Assert.Equal(4, bidiag.V.Rows);

        // No NaN or Inf values
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.False(double.IsNaN(bidiag.U[i, j]),
                    $"Algorithm {algorithm}: U has NaN at [{i},{j}]");
                Assert.False(double.IsNaN(bidiag.B[i, j]),
                    $"Algorithm {algorithm}: B has NaN at [{i},{j}]");
                Assert.False(double.IsNaN(bidiag.V[i, j]),
                    $"Algorithm {algorithm}: V has NaN at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Rectangular Matrix Tests

    [Fact]
    public void BidiagonalDecomposition_TallMatrix_ValidDecomposition()
    {
        // Arrange - More rows than columns
        var A = CreateTestMatrix(6, 3);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A);

        // Assert
        Assert.Equal(6, bidiag.U.Rows);
        Assert.Equal(3, bidiag.B.Columns);
        Assert.Equal(3, bidiag.V.Rows);
    }

    [Fact]
    public void BidiagonalDecomposition_WideMatrix_ValidDecomposition()
    {
        // Arrange - More columns than rows
        var A = CreateTestMatrix(3, 6);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A);

        // Assert
        Assert.Equal(3, bidiag.U.Rows);
        Assert.Equal(6, bidiag.B.Columns);
        Assert.Equal(6, bidiag.V.Rows);
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void BidiagonalDecomposition_LargeMatrix_ValidDecomposition()
    {
        // Arrange
        var A = CreateTestMatrix(10, 10, seed: 999);

        // Act
        var bidiag = new BidiagonalDecomposition<double>(A);

        // Assert
        Assert.True(IsBidiagonal(bidiag.B, LooseTolerance),
            "Large matrix B should be bidiagonal");
    }

    #endregion
}
