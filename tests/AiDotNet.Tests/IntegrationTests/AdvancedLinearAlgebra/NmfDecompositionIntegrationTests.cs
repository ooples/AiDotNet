using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Non-negative Matrix Factorization (NMF) decomposition.
/// These tests verify: V ≈ W*H, W and H are non-negative, reconstruction accuracy.
/// </summary>
public class NmfDecompositionIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double ReconstructionTolerance = 0.3; // NMF is approximate

    #region Helper Methods

    private static Matrix<double> CreateNonNegativeMatrix(int rows, int cols, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = random.NextDouble() * 10; // Non-negative values
            }
        }
        return matrix;
    }

    private static bool IsNonNegative(Matrix<double> matrix)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                if (matrix[i, j] < -Tolerance)
                    return false;
            }
        }
        return true;
    }

    private static double FrobeniusNorm(Matrix<double> m)
    {
        double sum = 0;
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                sum += m[i, j] * m[i, j];
            }
        }
        return Math.Sqrt(sum);
    }

    private static double RelativeReconstructionError(Matrix<double> original, Matrix<double> reconstructed)
    {
        double diffNorm = 0;
        double origNorm = 0;
        for (int i = 0; i < original.Rows; i++)
        {
            for (int j = 0; j < original.Columns; j++)
            {
                double diff = original[i, j] - reconstructed[i, j];
                diffNorm += diff * diff;
                origNorm += original[i, j] * original[i, j];
            }
        }
        return Math.Sqrt(diffNorm) / Math.Sqrt(origNorm);
    }

    #endregion

    #region Basic Decomposition Tests

    [Theory]
    [InlineData(5, 5, 2)]
    [InlineData(6, 4, 2)]
    [InlineData(8, 6, 3)]
    public void NmfDecomposition_BasicDecomposition_ProducesNonNegativeFactors(int rows, int cols, int components)
    {
        // Arrange
        var V = CreateNonNegativeMatrix(rows, cols);

        // Act
        var nmf = new NmfDecomposition<double>(V, components);

        // Assert - W and H should be non-negative
        Assert.True(IsNonNegative(nmf.W), "W matrix should be non-negative");
        Assert.True(IsNonNegative(nmf.H), "H matrix should be non-negative");
    }

    [Theory]
    [InlineData(5, 5, 2)]
    [InlineData(6, 4, 2)]
    [InlineData(8, 6, 3)]
    public void NmfDecomposition_FactorDimensions_AreCorrect(int rows, int cols, int components)
    {
        // Arrange
        var V = CreateNonNegativeMatrix(rows, cols);

        // Act
        var nmf = new NmfDecomposition<double>(V, components);

        // Assert - Check dimensions
        Assert.Equal(rows, nmf.W.Rows);
        Assert.Equal(components, nmf.W.Columns);
        Assert.Equal(components, nmf.H.Rows);
        Assert.Equal(cols, nmf.H.Columns);
        Assert.Equal(components, nmf.Components);
    }

    [Theory]
    [InlineData(6, 6, 2)]
    [InlineData(8, 8, 3)]
    [InlineData(10, 10, 4)]
    public void NmfDecomposition_Reconstruct_ApproximatesOriginal(int size, int _, int components)
    {
        // Arrange
        var V = CreateNonNegativeMatrix(size, size);

        // Act
        var nmf = new NmfDecomposition<double>(V, components, maxIterations: 300);
        var reconstructed = nmf.Reconstruct();

        // Assert - Reconstruction should approximate original
        double relError = RelativeReconstructionError(V, reconstructed);
        Assert.True(relError < ReconstructionTolerance,
            $"Relative reconstruction error {relError} should be less than {ReconstructionTolerance}");
    }

    [Fact]
    public void NmfDecomposition_WH_Product_EqualsReconstruct()
    {
        // Arrange
        var V = CreateNonNegativeMatrix(6, 5);

        // Act
        var nmf = new NmfDecomposition<double>(V, 2);
        var reconstructed = nmf.Reconstruct();
        var whProduct = nmf.W.Multiply(nmf.H);

        // Assert - Reconstruct() should equal W*H
        for (int i = 0; i < reconstructed.Rows; i++)
        {
            for (int j = 0; j < reconstructed.Columns; j++)
            {
                Assert.True(Math.Abs(reconstructed[i, j] - whProduct[i, j]) < Tolerance,
                    $"Reconstruct()[{i},{j}] should equal W*H");
            }
        }
    }

    #endregion

    #region Convergence Tests

    [Fact]
    public void NmfDecomposition_MoreIterations_ImproveReconstruction()
    {
        // Arrange
        var V = CreateNonNegativeMatrix(8, 8);

        // Act
        var nmfFew = new NmfDecomposition<double>(V, 3, maxIterations: 10);
        var nmfMany = new NmfDecomposition<double>(V, 3, maxIterations: 300);

        var reconFew = nmfFew.Reconstruct();
        var reconMany = nmfMany.Reconstruct();

        // Assert - More iterations should give better (or equal) reconstruction
        double errorFew = RelativeReconstructionError(V, reconFew);
        double errorMany = RelativeReconstructionError(V, reconMany);

        Assert.True(errorMany <= errorFew + 0.1,
            $"More iterations should not significantly worsen error. Few: {errorFew}, Many: {errorMany}");
    }

    [Fact]
    public void NmfDecomposition_MoreComponents_ImproveReconstruction()
    {
        // Arrange
        var V = CreateNonNegativeMatrix(10, 10);

        // Act
        var nmfFew = new NmfDecomposition<double>(V, 2, maxIterations: 200);
        var nmfMore = new NmfDecomposition<double>(V, 5, maxIterations: 200);

        var reconFew = nmfFew.Reconstruct();
        var reconMore = nmfMore.Reconstruct();

        // Assert - More components should give better (or equal) reconstruction
        double errorFew = RelativeReconstructionError(V, reconFew);
        double errorMore = RelativeReconstructionError(V, reconMore);

        Assert.True(errorMore <= errorFew + 0.05,
            $"More components should give better reconstruction. Few: {errorFew}, More: {errorMore}");
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void NmfDecomposition_NegativeMatrix_ThrowsArgumentException()
    {
        // Arrange
        var V = new Matrix<double>(3, 3);
        V[0, 0] = -1; // Negative value

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NmfDecomposition<double>(V, 2));
    }

    [Fact]
    public void NmfDecomposition_ZeroComponents_ThrowsArgumentException()
    {
        // Arrange
        var V = CreateNonNegativeMatrix(3, 3);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NmfDecomposition<double>(V, 0));
    }

    [Fact]
    public void NmfDecomposition_TooManyComponents_ThrowsArgumentException()
    {
        // Arrange
        var V = CreateNonNegativeMatrix(3, 4);

        // Act & Assert - Components can't exceed min(rows, cols)
        Assert.Throws<ArgumentException>(() => new NmfDecomposition<double>(V, 10));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void NmfDecomposition_ZeroMatrix_ProducesZeroFactors()
    {
        // Arrange - Matrix with all zeros
        var V = new Matrix<double>(4, 4);

        // Act
        var nmf = new NmfDecomposition<double>(V, 2);
        var reconstructed = nmf.Reconstruct();

        // Assert - Reconstruction should be near zero
        double error = FrobeniusNorm(reconstructed);
        Assert.True(error < Tolerance,
            $"Zero matrix reconstruction should be zero. Error: {error}");
    }

    [Fact]
    public void NmfDecomposition_SparseMatrix_HandledCorrectly()
    {
        // Arrange - Sparse matrix with many zeros
        var V = new Matrix<double>(5, 5);
        V[0, 0] = 1; V[1, 1] = 2; V[2, 2] = 3;
        V[3, 3] = 4; V[4, 4] = 5;

        // Act
        var nmf = new NmfDecomposition<double>(V, 2, maxIterations: 200);

        // Assert
        Assert.True(IsNonNegative(nmf.W), "W should be non-negative for sparse input");
        Assert.True(IsNonNegative(nmf.H), "H should be non-negative for sparse input");
    }

    [Fact]
    public void NmfDecomposition_DefaultComponents_UsesHalfMinDimension()
    {
        // Arrange
        var V = CreateNonNegativeMatrix(6, 4); // min = 4, so default = 2

        // Act
        var nmf = new NmfDecomposition<double>(V); // No components specified

        // Assert
        Assert.Equal(2, nmf.Components);
    }

    #endregion

    #region Numerical Properties

    [Fact]
    public void NmfDecomposition_ReconstructedValues_AreNonNegative()
    {
        // Arrange
        var V = CreateNonNegativeMatrix(6, 6);

        // Act
        var nmf = new NmfDecomposition<double>(V, 3);
        var reconstructed = nmf.Reconstruct();

        // Assert
        Assert.True(IsNonNegative(reconstructed),
            "Reconstructed matrix should be non-negative");
    }

    [Fact]
    public void NmfDecomposition_NoNaNOrInfinity_InFactors()
    {
        // Arrange
        var V = CreateNonNegativeMatrix(5, 5);

        // Act
        var nmf = new NmfDecomposition<double>(V, 2);

        // Assert
        for (int i = 0; i < nmf.W.Rows; i++)
        {
            for (int j = 0; j < nmf.W.Columns; j++)
            {
                Assert.False(double.IsNaN(nmf.W[i, j]), $"W[{i},{j}] should not be NaN");
                Assert.False(double.IsInfinity(nmf.W[i, j]), $"W[{i},{j}] should not be infinity");
            }
        }

        for (int i = 0; i < nmf.H.Rows; i++)
        {
            for (int j = 0; j < nmf.H.Columns; j++)
            {
                Assert.False(double.IsNaN(nmf.H[i, j]), $"H[{i},{j}] should not be NaN");
                Assert.False(double.IsInfinity(nmf.H[i, j]), $"H[{i},{j}] should not be infinity");
            }
        }
    }

    #endregion
}
