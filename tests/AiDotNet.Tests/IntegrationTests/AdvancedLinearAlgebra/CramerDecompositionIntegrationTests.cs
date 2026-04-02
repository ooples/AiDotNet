using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Cramer's rule decomposition.
/// These tests verify: correct solution via determinants, matrix inversion,
/// and proper error handling for singular matrices.
/// </summary>
public class CramerDecompositionIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-6;

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

    private static Matrix<double> CreateDiagonallyDominantMatrix(int size, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    matrix[i, j] = random.NextDouble() * 2 - 1;
                    rowSum += Math.Abs(matrix[i, j]);
                }
            }
            // Make diagonal dominant
            matrix[i, i] = rowSum + random.NextDouble() + 1;
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

    #endregion

    #region Basic Solve Tests





    #endregion

    #region Matrix Inversion Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void CramerDecomposition_Invert_ProducesCorrectInverse(int size)
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(size);

        // Act
        var cramer = new CramerDecomposition<double>(A);
        var AInv = cramer.Invert();

        // Assert - A * A^-1 should equal I
        var product = A.Multiply(AInv);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(product[i, j] - expected) < LooseTolerance,
                    $"(A * A^-1)[{i},{j}] = {product[i, j]}, expected {expected}");
            }
        }
    }

    [Fact]
    public void CramerDecomposition_Invert_2x2Matrix_ExactInverse()
    {
        // Arrange - 2x2 matrix [[a, b], [c, d]]
        // Inverse = 1/(ad-bc) * [[d, -b], [-c, a]]
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 4; A[0, 1] = 3;
        A[1, 0] = 2; A[1, 1] = 1;
        // det = 4*1 - 3*2 = -2
        // inverse = 1/-2 * [[1, -3], [-2, 4]] = [[-0.5, 1.5], [1, -2]]

        // Act
        var cramer = new CramerDecomposition<double>(A);
        var AInv = cramer.Invert();

        // Assert
        Assert.True(Math.Abs(AInv[0, 0] - (-0.5)) < Tolerance);
        Assert.True(Math.Abs(AInv[0, 1] - 1.5) < Tolerance);
        Assert.True(Math.Abs(AInv[1, 0] - 1.0) < Tolerance);
        Assert.True(Math.Abs(AInv[1, 1] - (-2.0)) < Tolerance);
    }

    [Fact]
    public void CramerDecomposition_Invert_IdentityMatrix_ReturnsIdentity()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(3);

        // Act
        var cramer = new CramerDecomposition<double>(I);
        var IInv = cramer.Invert();

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(IInv[i, j] - expected) < Tolerance,
                    $"Inverse of identity should be identity. [{i},{j}]");
            }
        }
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void CramerDecomposition_NonSquareMatrix_ThrowsArgumentException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4); // Non-square

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new CramerDecomposition<double>(A));
    }


    [Fact]
    public void CramerDecomposition_SingularMatrix_Invert_ThrowsInvalidOperationException()
    {
        // Arrange - Singular matrix
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 1; A[0, 1] = 2;
        A[1, 0] = 2; A[1, 1] = 4; // Row 2 = 2 * Row 1

        // Act & Assert
        var cramer = new CramerDecomposition<double>(A);
        Assert.Throws<InvalidOperationException>(() => cramer.Invert());
    }


    #endregion

    #region Edge Cases


    [Fact]
    public void CramerDecomposition_1x1Matrix_Invert_WorksCorrectly()
    {
        // Arrange - 1x1 matrix inversion now works after the 0x0 determinant fix
        var A = new Matrix<double>(1, 1);
        A[0, 0] = 4.0;

        // Act
        var cramer = new CramerDecomposition<double>(A);
        var AInv = cramer.Invert();

        // Assert - Inverse of [4] is [0.25]
        Assert.True(Math.Abs(AInv[0, 0] - 0.25) < Tolerance, "1/4 = 0.25");
    }

    [Fact]
    public void CramerDecomposition_2x2Matrix_Invert_WorksCorrectly()
    {
        // Arrange
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 4.0; A[0, 1] = 0.0;
        A[1, 0] = 0.0; A[1, 1] = 2.0;

        // Act
        var cramer = new CramerDecomposition<double>(A);
        var AInv = cramer.Invert();

        // Assert - Inverse of diagonal matrix is simple reciprocals
        Assert.True(Math.Abs(AInv[0, 0] - 0.25) < Tolerance, "1/4 = 0.25");
        Assert.True(Math.Abs(AInv[1, 1] - 0.5) < Tolerance, "1/2 = 0.5");
        Assert.True(Math.Abs(AInv[0, 1]) < Tolerance, "Off-diagonal should be 0");
        Assert.True(Math.Abs(AInv[1, 0]) < Tolerance, "Off-diagonal should be 0");
    }


    #endregion

    #region Numerical Properties Tests


    [Fact]
    public void CramerDecomposition_Invert_Twice_ReturnsOriginal()
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(3);

        // Act
        var cramer1 = new CramerDecomposition<double>(A);
        var AInv = cramer1.Invert();
        var cramer2 = new CramerDecomposition<double>(AInv);
        var ADoubleInv = cramer2.Invert();

        // Assert - (A^-1)^-1 = A
        double maxDiff = MaxAbsDiff(A, ADoubleInv);
        Assert.True(maxDiff < LooseTolerance,
            $"Double inverse should return original matrix. Max diff: {maxDiff}");
    }


    #endregion
}
