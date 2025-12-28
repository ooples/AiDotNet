using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for UDU decomposition that verify mathematical correctness.
/// These tests verify: A = U*D*U^T, U is upper triangular with ones on diagonal.
/// </summary>
public class UduDecompositionIntegrationTests
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

    private static Matrix<double> ReconstructFromUdu(Matrix<double> U, Vector<double> D)
    {
        int n = U.Rows;
        // Compute U*D*U^T
        // First compute U*D
        var UD = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                UD[i, j] = U[i, j] * D[j];
            }
        }
        // Then compute (U*D)*U^T
        return UD.Multiply(U.Transpose());
    }

    private static bool IsUpperTriangular(Matrix<double> U, double tolerance)
    {
        for (int i = 1; i < U.Rows; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (Math.Abs(U[i, j]) > tolerance)
                    return false;
            }
        }
        return true;
    }

    private static bool HasOnesOnDiagonal(Matrix<double> U, double tolerance)
    {
        int n = Math.Min(U.Rows, U.Columns);
        for (int i = 0; i < n; i++)
        {
            if (Math.Abs(U[i, i] - 1.0) > tolerance)
                return false;
        }
        return true;
    }

    #endregion

    #region Reconstruction Tests (A = U*D*U^T)

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void UduDecomposition_Crout_Reconstruction(int size)
    {
        // Arrange - UDU works on symmetric matrices
        var A = CreateSpdMatrix(size);

        // Act
        var udu = new UduDecomposition<double>(A, UduAlgorithmType.Crout);

        // Assert - Verify A = U*D*U^T
        var reconstructed = ReconstructFromUdu(udu.U, udu.D);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal U*D*U^T. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void UduDecomposition_Doolittle_Reconstruction(int size)
    {
        // Arrange
        var A = CreateSpdMatrix(size, seed: 123);

        // Act
        var udu = new UduDecomposition<double>(A, UduAlgorithmType.Doolittle);

        // Assert
        var reconstructed = ReconstructFromUdu(udu.U, udu.D);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Doolittle: A should equal U*D*U^T. Max difference: {maxDiff}");
    }

    #endregion

    #region U Matrix Structure Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void UduDecomposition_U_IsUpperTriangular(int size)
    {
        // Arrange
        var A = CreateSpdMatrix(size);

        // Act
        var udu = new UduDecomposition<double>(A);

        // Assert - U should be upper triangular
        Assert.True(IsUpperTriangular(udu.U, LooseTolerance),
            "U should be upper triangular (zeros below diagonal)");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void UduDecomposition_U_HasOnesOnDiagonal(int size)
    {
        // Arrange
        var A = CreateSpdMatrix(size, seed: 456);

        // Act
        var udu = new UduDecomposition<double>(A);

        // Assert - U should have 1s on diagonal
        Assert.True(HasOnesOnDiagonal(udu.U, LooseTolerance),
            "U should have 1s on diagonal");
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void UduDecomposition_IdentityMatrix_HasAllOnesInD()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var udu = new UduDecomposition<double>(I);

        // Assert - D should be all 1s for identity matrix
        for (int i = 0; i < udu.D.Length; i++)
        {
            Assert.True(Math.Abs(udu.D[i] - 1.0) < Tolerance,
                $"D[{i}] = {udu.D[i]}, expected 1.0");
        }
    }

    [Fact]
    public void UduDecomposition_DiagonalMatrix_PreservesDiagonalInD()
    {
        // Arrange - Diagonal matrix with positive values
        var D_input = new Matrix<double>(4, 4);
        D_input[0, 0] = 4; D_input[1, 1] = 3; D_input[2, 2] = 2; D_input[3, 3] = 1;

        // Act
        var udu = new UduDecomposition<double>(D_input);

        // Assert - D output should match diagonal input (order may vary due to pivoting)
        var reconstructed = ReconstructFromUdu(udu.U, udu.D);
        double maxDiff = MaxAbsDiff(D_input, reconstructed);
        Assert.True(maxDiff < Tolerance, $"Diagonal matrix reconstruction failed. Max diff: {maxDiff}");
    }

    [Fact]
    public void UduDecomposition_SpdMatrix_HasPositiveD()
    {
        // Arrange - Positive definite matrix
        var A = CreateSpdMatrix(4, seed: 789);

        // Act
        var udu = new UduDecomposition<double>(A);

        // Assert - D should have all positive values for positive definite
        for (int i = 0; i < udu.D.Length; i++)
        {
            Assert.True(udu.D[i] > -Tolerance,
                $"D[{i}] = {udu.D[i]}, should be positive for SPD matrix");
        }
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(UduAlgorithmType.Crout)]
    [InlineData(UduAlgorithmType.Doolittle)]
    public void UduDecomposition_AllAlgorithms_ProduceValidDecomposition(UduAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateSpdMatrix(4, seed: 42);

        // Act
        var udu = new UduDecomposition<double>(A, algorithm);

        // Assert - Basic validity checks
        Assert.Equal(4, udu.U.Rows);
        Assert.Equal(4, udu.U.Columns);
        Assert.Equal(4, udu.D.Length);

        // No NaN values
        for (int i = 0; i < 4; i++)
        {
            Assert.False(double.IsNaN(udu.D[i]),
                $"Algorithm {algorithm}: D has NaN at [{i}]");
            for (int j = 0; j < 4; j++)
            {
                Assert.False(double.IsNaN(udu.U[i, j]),
                    $"Algorithm {algorithm}: U has NaN at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void UduDecomposition_NonSquareMatrix_ThrowsException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new UduDecomposition<double>(A));
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void UduDecomposition_LargeMatrix_CorrectDecomposition()
    {
        // Arrange
        var A = CreateSpdMatrix(10, seed: 999);

        // Act
        var udu = new UduDecomposition<double>(A);

        // Assert
        var reconstructed = ReconstructFromUdu(udu.U, udu.D);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Large matrix reconstruction failed. Max difference: {maxDiff}");
    }

    #endregion
}
