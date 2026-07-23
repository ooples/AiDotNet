using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Complex Matrix Decomposition wrapper.
/// These tests verify: wrapper correctly delegates to base decomposition,
/// complex conversions work properly, and Solve/Invert produce valid results.
/// </summary>
public class ComplexMatrixDecompositionIntegrationTests
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
            matrix[i, i] = rowSum + random.NextDouble() + 1;
        }
        return matrix;
    }

    #endregion

    #region Constructor Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void ComplexMatrixDecomposition_Constructor_WrapsBaseDecomposition(int size)
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(size);
        var baseDecomp = new LuDecomposition<double>(A);

        // Act
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        // Assert - The wrapped decomposition should have same dimensions
        Assert.Equal(size, complexDecomp.A.Rows);
        Assert.Equal(size, complexDecomp.A.Columns);
    }

    [Fact(Timeout = 120000)]
    public async Task ComplexMatrixDecomposition_A_HasZeroImaginaryParts()
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(3);
        var baseDecomp = new LuDecomposition<double>(A);

        // Act
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        // Assert - All imaginary parts should be zero
        for (int i = 0; i < complexDecomp.A.Rows; i++)
        {
            for (int j = 0; j < complexDecomp.A.Columns; j++)
            {
                Assert.True(Math.Abs(complexDecomp.A[i, j].Imaginary) < Tolerance,
                    $"A[{i},{j}] imaginary part should be zero");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ComplexMatrixDecomposition_A_RealPartsMatchOriginal()
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(3);
        var baseDecomp = new LuDecomposition<double>(A);

        // Act
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        // Assert - Real parts should match original matrix
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = 0; j < A.Columns; j++)
            {
                Assert.True(Math.Abs(complexDecomp.A[i, j].Real - A[i, j]) < Tolerance,
                    $"A[{i},{j}] real part should match original");
            }
        }
    }

    #endregion

    #region Solve Tests



    #endregion

    #region Invert Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void ComplexMatrixDecomposition_Invert_ProducesCorrectInverse(int size)
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(size);
        var baseDecomp = new LuDecomposition<double>(A);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        // Act
        var AInv = complexDecomp.Invert();

        // Assert - Convert back to real and verify A * A^-1 = I
        var AInvReal = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                AInvReal[i, j] = AInv[i, j].Real;
            }
        }

        var product = A.Multiply(AInvReal);
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

    [Fact(Timeout = 120000)]
    public async Task ComplexMatrixDecomposition_Invert_HasZeroImaginaryParts()
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(3);
        var baseDecomp = new LuDecomposition<double>(A);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        // Act
        var AInv = complexDecomp.Invert();

        // Assert - For real input, inverse should have zero imaginary parts
        for (int i = 0; i < AInv.Rows; i++)
        {
            for (int j = 0; j < AInv.Columns; j++)
            {
                Assert.True(Math.Abs(AInv[i, j].Imaginary) < Tolerance,
                    $"Inverse[{i},{j}] imaginary part should be zero");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ComplexMatrixDecomposition_Invert_IdentityMatrix_ReturnsIdentity()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(3);
        var baseDecomp = new LuDecomposition<double>(I);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        // Act
        var IInv = complexDecomp.Invert();

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(IInv[i, j].Real - expected) < Tolerance);
                Assert.True(Math.Abs(IInv[i, j].Imaginary) < Tolerance);
            }
        }
    }

    #endregion

    #region Different Base Decomposition Tests



    #endregion

    #region Numerical Properties Tests



    #endregion
}
