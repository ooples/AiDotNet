using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

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

    [Fact]
    public void ComplexMatrixDecomposition_A_HasZeroImaginaryParts()
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

    [Fact]
    public void ComplexMatrixDecomposition_A_RealPartsMatchOriginal()
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

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void ComplexMatrixDecomposition_Solve_ProducesCorrectSolution(int size)
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(size);
        var xExpected = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);
        var bComplex = new Vector<Complex<double>>(size);
        for (int i = 0; i < size; i++)
            bComplex[i] = new Complex<double>(b[i], 0);

        var baseDecomp = new LuDecomposition<double>(A);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        // Act
        var xComputed = complexDecomp.Solve(bComplex);

        // Assert
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(xComputed[i].Real - xExpected[i]) < LooseTolerance,
                $"x[{i}] should be {xExpected[i]}, got {xComputed[i].Real}");
            Assert.True(Math.Abs(xComputed[i].Imaginary) < LooseTolerance,
                $"x[{i}] imaginary part should be zero");
        }
    }

    [Fact]
    public void ComplexMatrixDecomposition_Solve_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(3);
        var bComplex = new Vector<Complex<double>>(new[]
        {
            new Complex<double>(1.0, 0),
            new Complex<double>(2.0, 0),
            new Complex<double>(3.0, 0)
        });

        var baseDecomp = new LuDecomposition<double>(I);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        // Act
        var x = complexDecomp.Solve(bComplex);

        // Assert
        for (int i = 0; i < 3; i++)
        {
            Assert.True(Math.Abs(x[i].Real - bComplex[i].Real) < Tolerance);
            Assert.True(Math.Abs(x[i].Imaginary) < Tolerance);
        }
    }

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

    [Fact]
    public void ComplexMatrixDecomposition_Invert_HasZeroImaginaryParts()
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

    [Fact]
    public void ComplexMatrixDecomposition_Invert_IdentityMatrix_ReturnsIdentity()
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

    [Fact]
    public void ComplexMatrixDecomposition_WithQrDecomposition_WorksCorrectly()
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(3);
        var baseDecomp = new QrDecomposition<double>(A);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        var bComplex = new Vector<Complex<double>>(new[]
        {
            new Complex<double>(1.0, 0),
            new Complex<double>(2.0, 0),
            new Complex<double>(3.0, 0)
        });

        // Act
        var x = complexDecomp.Solve(bComplex);

        // Assert - Verify solution
        var xReal = new Vector<double>(3);
        for (int i = 0; i < 3; i++)
            xReal[i] = x[i].Real;

        var bComputed = A.Multiply(xReal);
        for (int i = 0; i < 3; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - bComplex[i].Real) < LooseTolerance);
        }
    }

    [Fact]
    public void ComplexMatrixDecomposition_WithCholeskyDecomposition_WorksCorrectly()
    {
        // Arrange - Create positive definite matrix
        var temp = CreateDiagonallyDominantMatrix(3);
        var A = temp.Transpose().Multiply(temp); // A^T * A is positive definite

        var baseDecomp = new CholeskyDecomposition<double>(A);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        var bComplex = new Vector<Complex<double>>(new[]
        {
            new Complex<double>(1.0, 0),
            new Complex<double>(2.0, 0),
            new Complex<double>(3.0, 0)
        });

        // Act
        var x = complexDecomp.Solve(bComplex);

        // Assert - Verify solution
        var xReal = new Vector<double>(3);
        for (int i = 0; i < 3; i++)
            xReal[i] = x[i].Real;

        var bComputed = A.Multiply(xReal);
        for (int i = 0; i < 3; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - bComplex[i].Real) < LooseTolerance);
        }
    }

    #endregion

    #region Numerical Properties Tests

    [Fact]
    public void ComplexMatrixDecomposition_Solve_AndInvert_AreConsistent()
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(3);
        var baseDecomp = new LuDecomposition<double>(A);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        var bComplex = new Vector<Complex<double>>(new[]
        {
            new Complex<double>(1.0, 0),
            new Complex<double>(2.0, 0),
            new Complex<double>(3.0, 0)
        });

        // Act
        var xFromSolve = complexDecomp.Solve(bComplex);
        var AInv = complexDecomp.Invert();
        var xFromInverse = AInv.Multiply(bComplex);

        // Assert
        for (int i = 0; i < 3; i++)
        {
            Assert.True(Math.Abs(xFromSolve[i].Real - xFromInverse[i].Real) < LooseTolerance,
                $"Solve and inverse multiplication should match. Index {i}");
        }
    }

    [Fact]
    public void ComplexMatrixDecomposition_NoNaNOrInfinity_InResults()
    {
        // Arrange
        var A = CreateDiagonallyDominantMatrix(4);
        var baseDecomp = new LuDecomposition<double>(A);
        var complexDecomp = new ComplexMatrixDecomposition<double>(baseDecomp);

        var bComplex = new Vector<Complex<double>>(4);
        for (int i = 0; i < 4; i++)
            bComplex[i] = new Complex<double>(i + 1.0, 0);

        // Act
        var x = complexDecomp.Solve(bComplex);
        var AInv = complexDecomp.Invert();

        // Assert - Check solution
        for (int i = 0; i < x.Length; i++)
        {
            Assert.False(double.IsNaN(x[i].Real), $"x[{i}].Real should not be NaN");
            Assert.False(double.IsNaN(x[i].Imaginary), $"x[{i}].Imaginary should not be NaN");
            Assert.False(double.IsInfinity(x[i].Real), $"x[{i}].Real should not be infinity");
            Assert.False(double.IsInfinity(x[i].Imaginary), $"x[{i}].Imaginary should not be infinity");
        }

        // Check inverse
        for (int i = 0; i < AInv.Rows; i++)
        {
            for (int j = 0; j < AInv.Columns; j++)
            {
                Assert.False(double.IsNaN(AInv[i, j].Real), $"AInv[{i},{j}].Real should not be NaN");
                Assert.False(double.IsNaN(AInv[i, j].Imaginary), $"AInv[{i},{j}].Imaginary should not be NaN");
            }
        }
    }

    #endregion
}
