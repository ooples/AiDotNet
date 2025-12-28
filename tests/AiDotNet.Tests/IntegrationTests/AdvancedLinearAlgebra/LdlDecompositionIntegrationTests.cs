using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for LDL decomposition that verify mathematical correctness.
/// These tests verify: A = L*D*L^T for symmetric matrices.
/// </summary>
public class LdlDecompositionIntegrationTests
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

    private static Matrix<double> CreateSpdMatrix(int size, int seed = 42)
    {
        // Create SPD matrix via A^T*A + epsilon*I
        var random = new Random(seed);
        var B = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                B[i, j] = random.NextDouble() * 2 - 1;
            }
        }
        var result = B.Transpose().Multiply(B);
        // Add small value to diagonal to ensure positive definiteness
        for (int i = 0; i < size; i++)
        {
            result[i, i] += 1.0;
        }
        return result;
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

    private static Matrix<double> ReconstructFromLdl(Matrix<double> L, Vector<double> D)
    {
        int n = L.Rows;
        // Compute L*D*L^T
        // First compute L*D
        var LD = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                LD[i, j] = L[i, j] * D[j];
            }
        }
        // Then compute (L*D)*L^T
        return LD.Multiply(L.Transpose());
    }

    #endregion

    #region Reconstruction Tests (A = L*D*L^T)

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void LdlDecomposition_Cholesky_Reconstruction(int size)
    {
        // Arrange - Use SPD matrix for Cholesky-based LDL
        var A = CreateSpdMatrix(size);

        // Act
        var ldl = new LdlDecomposition<double>(A, LdlAlgorithmType.Cholesky);

        // Assert - Verify A = L*D*L^T
        var reconstructed = ReconstructFromLdl(ldl.L, ldl.D);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal L*D*L^T. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void LdlDecomposition_Crout_Reconstruction(int size)
    {
        // Arrange - Crout handles symmetric matrices that may not be positive-definite
        var A = CreateSymmetricMatrix(size, seed: 123);

        // Act
        var ldl = new LdlDecomposition<double>(A, LdlAlgorithmType.Crout);

        // Assert
        var reconstructed = ReconstructFromLdl(ldl.L, ldl.D);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Crout: A should equal L*D*L^T. Max difference: {maxDiff}");
    }

    #endregion

    #region L Matrix Structure Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void LdlDecomposition_L_IsLowerTriangular(int size)
    {
        // Arrange
        var A = CreateSpdMatrix(size);

        // Act
        var ldl = new LdlDecomposition<double>(A);

        // Assert - L should be lower triangular
        for (int i = 0; i < size; i++)
        {
            for (int j = i + 1; j < size; j++)
            {
                Assert.True(Math.Abs(ldl.L[i, j]) < Tolerance,
                    $"L[{i},{j}] should be 0, got {ldl.L[i, j]}");
            }
        }
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void LdlDecomposition_L_HasOnesOnDiagonal(int size)
    {
        // Arrange
        var A = CreateSpdMatrix(size, seed: 456);

        // Act
        var ldl = new LdlDecomposition<double>(A);

        // Assert - L should have 1s on diagonal
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(ldl.L[i, i] - 1.0) < Tolerance,
                $"L[{i},{i}] should be 1, got {ldl.L[i, i]}");
        }
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void LdlDecomposition_IdentityMatrix_HasAllOnesInD()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var ldl = new LdlDecomposition<double>(I);

        // Assert - D should be all 1s
        for (int i = 0; i < ldl.D.Length; i++)
        {
            Assert.True(Math.Abs(ldl.D[i] - 1.0) < Tolerance,
                $"D[{i}] = {ldl.D[i]}, expected 1.0");
        }
    }

    [Fact]
    public void LdlDecomposition_DiagonalMatrix_PreservesDiagonalInD()
    {
        // Arrange - Diagonal matrix with positive values
        var D_input = new Matrix<double>(4, 4);
        D_input[0, 0] = 4; D_input[1, 1] = 3; D_input[2, 2] = 2; D_input[3, 3] = 1;

        // Act
        var ldl = new LdlDecomposition<double>(D_input);

        // Assert - D output should match diagonal input
        Assert.True(Math.Abs(ldl.D[0] - 4.0) < Tolerance, $"D[0] = {ldl.D[0]}, expected 4");
        Assert.True(Math.Abs(ldl.D[1] - 3.0) < Tolerance, $"D[1] = {ldl.D[1]}, expected 3");
        Assert.True(Math.Abs(ldl.D[2] - 2.0) < Tolerance, $"D[2] = {ldl.D[2]}, expected 2");
        Assert.True(Math.Abs(ldl.D[3] - 1.0) < Tolerance, $"D[3] = {ldl.D[3]}, expected 1");
    }

    [Fact]
    public void LdlDecomposition_PositiveDefiniteMatrix_HasPositiveD()
    {
        // Arrange - Positive definite matrix
        var A = CreateSpdMatrix(4, seed: 789);

        // Act
        var ldl = new LdlDecomposition<double>(A);

        // Assert - D should have all positive values for positive definite
        for (int i = 0; i < ldl.D.Length; i++)
        {
            Assert.True(ldl.D[i] > -Tolerance,
                $"D[{i}] = {ldl.D[i]}, should be positive for SPD matrix");
        }
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(LdlAlgorithmType.Cholesky)]
    [InlineData(LdlAlgorithmType.Crout)]
    public void LdlDecomposition_AllAlgorithms_ProduceValidDecomposition(LdlAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateSpdMatrix(4, seed: 42);

        // Act
        var ldl = new LdlDecomposition<double>(A, algorithm);

        // Assert - Basic validity checks
        Assert.Equal(4, ldl.L.Rows);
        Assert.Equal(4, ldl.L.Columns);
        Assert.Equal(4, ldl.D.Length);

        // No NaN values
        for (int i = 0; i < 4; i++)
        {
            Assert.False(double.IsNaN(ldl.D[i]),
                $"Algorithm {algorithm}: D has NaN at [{i}]");
            for (int j = 0; j < 4; j++)
            {
                Assert.False(double.IsNaN(ldl.L[i, j]),
                    $"Algorithm {algorithm}: L has NaN at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void LdlDecomposition_NonSquareMatrix_ThrowsException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new LdlDecomposition<double>(A));
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void LdlDecomposition_LargeMatrix_CorrectDecomposition()
    {
        // Arrange
        var A = CreateSpdMatrix(10, seed: 999);

        // Act
        var ldl = new LdlDecomposition<double>(A);

        // Assert
        var reconstructed = ReconstructFromLdl(ldl.L, ldl.D);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Large matrix reconstruction failed. Max difference: {maxDiff}");
    }

    #endregion
}
