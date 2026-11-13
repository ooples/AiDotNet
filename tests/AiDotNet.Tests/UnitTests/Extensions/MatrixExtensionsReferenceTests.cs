using System;
using AiDotNet.Extensions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Extensions;

/// <summary>
/// Reference values for the <see cref="MatrixExtensions"/> members whose engine-op conversion (#2490) also
/// fixed a wrong result. Each expected value is worked out by hand from the definition in the member's
/// documentation, and each one failed on the element-wise loop the member replaced.
/// </summary>
public class MatrixExtensionsReferenceTests
{
    private const double Tolerance = 1e-9;

    [Fact]
    public void Eigenvalues_SymmetricMatrix_ReturnsAscendingEigenvalues()
    {
        // [[2, 1], [1, 2]] has characteristic polynomial (2 - x)^2 - 1, so its eigenvalues are 1 and 3.
        var matrix = new Matrix<double>(new double[,] { { 2, 1 }, { 1, 2 } });

        var eigenvalues = matrix.Eigenvalues();

        Assert.Equal(2, eigenvalues.Length);
        Assert.Equal(1.0, eigenvalues[0], Tolerance);
        Assert.Equal(3.0, eigenvalues[1], Tolerance);
    }

    [Fact]
    public void Nullspace_RankOneMatrix_ReturnsUnitVectorAlongOneMinusOne()
    {
        // [[1, 1], [1, 1]] x = 0 exactly when x1 = -x2, so the null space is spanned by [1, -1] / sqrt(2).
        var matrix = new Matrix<double>(new double[,] { { 1, 1 }, { 1, 1 } });

        var nullspace = matrix.Nullspace(1e-8);

        Assert.Equal(2, nullspace.Rows);
        Assert.Equal(1, nullspace.Columns);
        double scale = 1.0 / Math.Sqrt(2.0);
        Assert.Equal(scale, Math.Abs(nullspace[0, 0]), Tolerance);
        Assert.Equal(-nullspace[0, 0], nullspace[1, 0], Tolerance);
    }

    [Fact]
    public void GetRange_RankOneMatrix_ReturnsUnitVectorAlongOneOne()
    {
        // Every column of [[1, 1], [1, 1]] is a multiple of [1, 1], so the range is spanned by [1, 1] / sqrt(2).
        var matrix = new Matrix<double>(new double[,] { { 1, 1 }, { 1, 1 } });

        var range = matrix.GetRange(1e-8);

        Assert.Equal(2, range.Rows);
        Assert.Equal(1, range.Columns);
        double scale = 1.0 / Math.Sqrt(2.0);
        Assert.Equal(scale, Math.Abs(range[0, 0]), Tolerance);
        Assert.Equal(range[0, 0], range[1, 0], Tolerance);
    }

    [Fact]
    public void Determinant_OneByOne_IsTheOnlyEntry()
    {
        var matrix = new Matrix<double>(new double[,] { { 5 } });

        Assert.Equal(5.0, matrix.Determinant(), Tolerance);
        Assert.True(matrix.IsInvertible());
    }

    [Fact]
    public void ToRealMatrix_Rectangular_KeepsEveryColumn()
    {
        var matrix = new Matrix<Complex<double>>(2, 3);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                matrix[i, j] = new Complex<double>(10 * i + j, -1);
            }
        }

        var real = matrix.ToRealMatrix();

        Assert.Equal(2, real.Rows);
        Assert.Equal(3, real.Columns);
        Assert.Equal(2.0, real[0, 2], Tolerance);
        Assert.Equal(12.0, real[1, 2], Tolerance);
    }

    [Fact]
    public void IsCirculantMatrix_RowsShiftedRight_IsTrue()
    {
        // Each row is the row above shifted one place to the right, the documented circulant layout.
        var matrix = new Matrix<double>(new double[,] { { 1, 2, 3 }, { 3, 1, 2 }, { 2, 3, 1 } });

        Assert.True(matrix.IsCirculantMatrix());
    }

    [Fact]
    public void IsBandMatrix_AsymmetricBandwidths_UsesEachBandwidth()
    {
        // Lower triangular with a nonzero two places below the diagonal: two subdiagonals, no superdiagonal.
        var matrix = new Matrix<double>(new double[,] { { 1, 0, 0 }, { 1, 1, 0 }, { 1, 1, 1 } });

        Assert.True(matrix.IsBandMatrix(2, 0));
        Assert.False(matrix.IsBandMatrix(1, 0));
    }

    [Fact]
    public void IsSymmetricMatrix_Rectangular_IsFalse()
    {
        var matrix = new Matrix<double>(new double[,] { { 1, 2, 3 }, { 2, 1, 4 } });

        Assert.False(matrix.IsSymmetricMatrix());
    }

    [Fact]
    public void IsLaplacianMatrix_DecimalWeights_IsTrue()
    {
        // L = D - A for edge weights 0.1 (0-1) and 0.2 (0-2): symmetric, non-positive off the diagonal and
        // every row sums to zero up to the rounding of 0.3 - 0.1 - 0.2 in double.
        var matrix = new Matrix<double>(new double[,]
        {
            { 0.3, -0.1, -0.2 },
            { -0.1, 0.1, 0 },
            { -0.2, 0, 0.2 }
        });

        Assert.True(matrix.IsLaplacianMatrix());
    }

    [Fact]
    public void IsSparseAndIsDense_DefaultThreshold_IsTheDocumentedHalf()
    {
        var dense = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        var zeros = new Matrix<double>(2, 2);

        Assert.False(dense.IsSparseMatrix());
        Assert.True(dense.IsDenseMatrix());
        Assert.True(zeros.IsSparseMatrix());
        Assert.False(zeros.IsDenseMatrix());
    }

    [Fact]
    public void IsSparseMatrix_LargeHalfMatrix_CountsBeyondHalfRange()
    {
        // 90000 elements is more than Half can count (its largest finite value is 65504).
        var zeros = new Matrix<Half>(300, 300);

        Assert.True(zeros.IsSparseMatrix((Half)0.5));
    }

    [Fact]
    public void IsLowerTriangularMatrix_DefaultTolerance_IgnoresRoundingNoise()
    {
        // The documented default tolerance is 1e-10, so 1e-12 above the diagonal is treated as zero.
        var matrix = new Matrix<double>(new double[,] { { 1, 1e-12 }, { 2, 1 } });

        Assert.True(matrix.IsLowerTriangularMatrix());
    }

    [Fact]
    public void IsPositiveDefiniteMatrix_DefaultTolerance_RejectsNearSingular()
    {
        // The Cholesky factor's diagonal is [1e-12, 1]; 1e-12 is below the documented 1e-10 default tolerance.
        var matrix = new Matrix<double>(new double[,] { { 1e-24, 0 }, { 0, 1 } });

        Assert.False(matrix.IsPositiveDefiniteMatrix());
    }
}
