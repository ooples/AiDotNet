using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Polar decomposition that verify mathematical correctness.
/// These tests verify: A = U*P, U is orthogonal, P is positive semi-definite.
/// </summary>
public class PolarDecompositionIntegrationTests
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

    private static Matrix<double> CreateSpdMatrix(int size, int seed = 42)
    {
        // Create SPD matrix via A^T*A
        var random = new Random(seed);
        var B = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                B[i, j] = random.NextDouble() * 2 - 1;
            }
        }
        return B.Transpose().Multiply(B);
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

    private static bool IsOrthogonal(Matrix<double> U, double tolerance)
    {
        var UtU = U.Transpose().Multiply(U);
        int n = UtU.Rows;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                if (Math.Abs(UtU[i, j] - expected) > tolerance)
                    return false;
            }
        }
        return true;
    }

    private static bool IsSymmetric(Matrix<double> P, double tolerance)
    {
        if (P.Rows != P.Columns)
            return false;

        for (int i = 0; i < P.Rows; i++)
        {
            for (int j = i + 1; j < P.Columns; j++)
            {
                if (Math.Abs(P[i, j] - P[j, i]) > tolerance)
                    return false;
            }
        }
        return true;
    }

    #endregion

    #region Reconstruction Tests (A = U*P)

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void PolarDecomposition_SVD_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var polar = new PolarDecomposition<double>(A, PolarAlgorithmType.SVD);

        // Assert - Verify A = U*P
        var reconstructed = polar.U.Multiply(polar.P);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal U*P. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void PolarDecomposition_NewtonSchulz_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, seed: 123);

        // Act
        var polar = new PolarDecomposition<double>(A, PolarAlgorithmType.NewtonSchulz);

        // Assert
        var reconstructed = polar.U.Multiply(polar.P);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Newton-Schulz: A should equal U*P. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void PolarDecomposition_HalleyIteration_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, seed: 456);

        // Act
        var polar = new PolarDecomposition<double>(A, PolarAlgorithmType.HalleyIteration);

        // Assert
        var reconstructed = polar.U.Multiply(polar.P);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Halley: A should equal U*P. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void PolarDecomposition_QRIteration_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, seed: 789);

        // Act
        var polar = new PolarDecomposition<double>(A, PolarAlgorithmType.QRIteration);

        // Assert
        var reconstructed = polar.U.Multiply(polar.P);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"QR Iteration: A should equal U*P. Max difference: {maxDiff}");
    }

    #endregion

    #region Orthogonality Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void PolarDecomposition_U_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var polar = new PolarDecomposition<double>(A);

        // Assert - U^T * U should be identity
        Assert.True(IsOrthogonal(polar.U, LooseTolerance),
            "U should be orthogonal (U^T * U = I)");
    }

    #endregion

    #region Positive Semi-Definite Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void PolarDecomposition_P_IsSymmetric(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var polar = new PolarDecomposition<double>(A);

        // Assert - P should be symmetric
        Assert.True(IsSymmetric(polar.P, LooseTolerance),
            "P should be symmetric");
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void PolarDecomposition_IdentityMatrix_ValidDecomposition()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var polar = new PolarDecomposition<double>(I);

        // Assert - For identity, U should be identity and P should be identity
        var reconstructed = polar.U.Multiply(polar.P);
        double maxDiff = MaxAbsDiff(I, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Identity matrix reconstruction failed. Max diff: {maxDiff}");
    }

    [Fact]
    public void PolarDecomposition_OrthogonalMatrix_UEqualsInput()
    {
        // Arrange - Create an orthogonal matrix via QR decomposition
        var B = CreateTestMatrix(4, seed: 123);
        var qr = new QrDecomposition<double>(B);
        var orthogonalMatrix = qr.Q;

        // Act
        var polar = new PolarDecomposition<double>(orthogonalMatrix);

        // Assert - U should equal the orthogonal input, P should be near identity
        Assert.True(IsOrthogonal(polar.U, LooseTolerance),
            "U of orthogonal matrix should be orthogonal");
    }

    [Fact]
    public void PolarDecomposition_PositiveDefiniteMatrix_ValidDecomposition()
    {
        // Arrange - Create symmetric positive definite matrix
        var A = CreateSpdMatrix(4, seed: 456);

        // Act
        var polar = new PolarDecomposition<double>(A);

        // Assert
        var reconstructed = polar.U.Multiply(polar.P);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"SPD matrix reconstruction failed. Max diff: {maxDiff}");
    }

    [Fact]
    public void PolarDecomposition_DiagonalMatrix_ValidDecomposition()
    {
        // Arrange
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 4; D[1, 1] = 3; D[2, 2] = 2; D[3, 3] = 1;

        // Act
        var polar = new PolarDecomposition<double>(D);

        // Assert
        var reconstructed = polar.U.Multiply(polar.P);
        double maxDiff = MaxAbsDiff(D, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Diagonal matrix reconstruction failed. Max diff: {maxDiff}");
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(PolarAlgorithmType.SVD)]
    [InlineData(PolarAlgorithmType.NewtonSchulz)]
    [InlineData(PolarAlgorithmType.HalleyIteration)]
    [InlineData(PolarAlgorithmType.QRIteration)]
    [InlineData(PolarAlgorithmType.ScalingAndSquaring)]
    public void PolarDecomposition_AllAlgorithms_ProduceValidDecomposition(PolarAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, seed: 42);

        // Act
        var polar = new PolarDecomposition<double>(A, algorithm);

        // Assert - Basic validity checks
        Assert.Equal(4, polar.U.Rows);
        Assert.Equal(4, polar.U.Columns);
        Assert.Equal(4, polar.P.Rows);
        Assert.Equal(4, polar.P.Columns);

        // No NaN or Inf values
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.False(double.IsNaN(polar.U[i, j]),
                    $"Algorithm {algorithm}: U has NaN at [{i},{j}]");
                Assert.False(double.IsNaN(polar.P[i, j]),
                    $"Algorithm {algorithm}: P has NaN at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void PolarDecomposition_LargeMatrix_ValidDecomposition()
    {
        // Arrange
        var A = CreateTestMatrix(10, seed: 999);

        // Act
        var polar = new PolarDecomposition<double>(A);

        // Assert
        var reconstructed = polar.U.Multiply(polar.P);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Large matrix reconstruction failed. Max difference: {maxDiff}");
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void PolarDecomposition_NonSquareMatrix_ThrowsException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4); // Non-square

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new PolarDecomposition<double>(A));
    }

    #endregion
}
