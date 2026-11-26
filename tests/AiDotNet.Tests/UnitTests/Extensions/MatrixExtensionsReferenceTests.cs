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

    [Fact]
    public void GetRank_IllConditioned_CountsSingularValuesAboveThreshold()
    {
        // [[1e-3, 1], [0, 1e-3]] has singular values about 1 and 1e-6, so numpy.linalg.matrix_rank(A, tol=1e-5)
        // is 1. Gaussian elimination keeps both pivots (1e-3 each), which are far above the threshold, and
        // reported rank 2.
        var matrix = new Matrix<double>(new double[,] { { 1e-3, 1 }, { 0, 1e-3 } });

        Assert.Equal(1, matrix.GetRank(1e-5));
        Assert.Equal(1, matrix.GetNullity(1e-5));
        Assert.Equal(1, matrix.Nullspace(1e-5).Columns);
    }

    [Fact]
    public void GetNullity_DefaultThreshold_IsNumpyMatrixRankTolerance()
    {
        // [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]] has rank 2 (row 2 is the mean of rows 1 and 3);
        // numpy.linalg.matrix_rank gives 2 with its default tolerance sigma_max * max(m, n) * eps. The omitted
        // threshold arrived as 0, and elimination left a 1.1e-16 rounding pivot, so the nullity was 0.
        var matrix = new Matrix<double>(new double[,] { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 }, { 0.7, 0.8, 0.9 } });

        Assert.Equal(1, matrix.GetNullity());
    }

    [Fact]
    public void Nullspace_NegativeThreshold_UsesMachineEpsilon()
    {
        // Numerical Recipes' default threshold is 0.5 * sqrt(m + n + 1) * w_max * eps with eps the machine epsilon
        // (about 2.5e-16 here), so the singular value 1e-12 of diag(1, 1e-12) is kept and the null space is empty,
        // as scipy.linalg.null_space also finds. A 1e-10 "eps" put the threshold at 1.1e-10.
        var matrix = new Matrix<double>(new double[,] { { 1, 0 }, { 0, 1e-12 } });

        Assert.Equal(0, matrix.Nullspace(-1.0).Columns);
    }

    [Fact]
    public void FrobeniusNorm_Half_AboveSquareRootOfHalfMax_IsFinite()
    {
        // ||diag(300, 300)||_F = 300 * sqrt(2) = 424.26; its sum of squares, 180000, overflows Half (65504).
        var matrix = new Matrix<Half>(new[,] { { (Half)300, (Half)0 }, { (Half)0, (Half)300 } });

        Assert.Equal(424.26, (double)matrix.FrobeniusNorm(), 0.5);
    }

    [Fact]
    public void InverseNewton_Half_LargeNorm_Converges()
    {
        // The initial guess is A^T / ||A||_F^2; ||diag(512, 512)||_F^2 = 524288 overflows Half, which made the
        // guess 0 and the iteration never converge. The inverse is diag(1/512, 1/512), exact in Half.
        var matrix = new Matrix<Half>(new[,] { { (Half)512, (Half)0 }, { (Half)0, (Half)512 } });

        var inverse = matrix.InverseNewton();

        Assert.Equal(1.0 / 512, (double)inverse[0, 0], 1e-6);
        Assert.Equal(1.0 / 512, (double)inverse[1, 1], 1e-6);
        Assert.Equal(0.0, (double)inverse[0, 1], 1e-6);
        Assert.Equal(0.0, (double)inverse[1, 0], 1e-6);
    }

    [Fact]
    public void InverseNewton_Half_NormAboveTheReciprocalRange_Converges()
    {
        // ||diag(8192, 8192)||_F = 11585. The scalar 1 / ||A||_F^2 = 7.5e-9 is below the smallest Half (6e-8), so
        // forming it first made the initial guess 0 and the iteration never converge. A^T / ||A||_F / ||A||_F
        // = 6.1e-5 is a Half, and so is the inverse diag(2^-13, 2^-13).
        var matrix = new Matrix<Half>(new[,] { { (Half)8192, (Half)0 }, { (Half)0, (Half)8192 } });

        var inverse = matrix.InverseNewton();

        Assert.Equal(1.0 / 8192, (double)inverse[0, 0], 1e-6);
        Assert.Equal(1.0 / 8192, (double)inverse[1, 1], 1e-6);
        Assert.Equal(0.0, (double)inverse[0, 1], 1e-6);
        Assert.Equal(0.0, (double)inverse[1, 0], 1e-6);
    }

    [Fact]
    public void IsInvertible_MatrixWithNaN_IsFalse()
    {
        // A NaN pivot is not "below the tolerance", so counting small pivots alone called this 4x4 invertible.
        var matrix = Matrix<double>.CreateIdentity(4);
        matrix[2, 2] = double.NaN;

        Assert.False(matrix.IsInvertible());
        Assert.True(Matrix<double>.CreateIdentity(4).IsInvertible());
    }

    [Fact]
    public void MatrixExponential_Half_TermsStayInRangeWhenRawPowersDoNot()
    {
        // sum_{k=0..8} 10^k / k! = 7330.84. Every term is a Half (the largest is 10^8 / 8! = 2480), but the raw
        // power 10^5 is not, so multiplying the power by 1 / k! afterwards returned infinity.
        var matrix = new Matrix<Half>(new[,] { { (Half)10 } });

        var exponential = matrix.MatrixExponential();

        Assert.Equal(7330.84, (double)exponential[0, 0], 30.0);
    }

    [Fact]
    public void MatrixExponential_Double_MatchesTheTruncatedSeries()
    {
        // A = [[0, 1], [0, 0]] is nilpotent (A^2 = 0), so e^A = I + A exactly.
        var matrix = new Matrix<double>(new[,] { { 0.0, 1.0 }, { 0.0, 0.0 } });

        var exponential = matrix.MatrixExponential();

        Assert.Equal(1.0, exponential[0, 0], 12);
        Assert.Equal(1.0, exponential[0, 1], 12);
        Assert.Equal(0.0, exponential[1, 0], 12);
        Assert.Equal(1.0, exponential[1, 1], 12);
    }

    [Fact]
    public void IsPositiveSemiDefiniteMatrix_EigenvalueWithinRounding_IsTrue()
    {
        // [[1, 1 + 1e-12], [1 + 1e-12, 1]] has eigenvalues 2 + 1e-12 and -1e-12. scipy's PSD check
        // (_eigvalsh_to_eps) treats eigenvalues above -1e6 * eps * max|lambda| (about -4.4e-10) as zero, so this
        // numerically singular matrix is positive semi-definite. The exact sign test rejected it.
        var nearSingular = new Matrix<double>(new double[,] { { 1, 1 + 1e-12 }, { 1 + 1e-12, 1 } });
        var indefinite = new Matrix<double>(new double[,] { { 1, 2 }, { 2, 1 } });

        Assert.True(nearSingular.IsPositiveSemiDefiniteMatrix());
        Assert.False(indefinite.IsPositiveSemiDefiniteMatrix());
    }

    [Fact]
    public void Determinant_And_IsInvertible_SupportDecimal()
    {
        // The tensor kernels only implement float and double; other element types are factorized in double.
        // [[2, 1], [1, 3]] has determinant 2 * 3 - 1 * 1 = 5; [[1, 2], [2, 4]] is singular (determinant 0).
        var invertible = new Matrix<decimal>(new decimal[,] { { 2m, 1m }, { 1m, 3m } });
        var singular = new Matrix<decimal>(new decimal[,] { { 1m, 2m }, { 2m, 4m } });

        Assert.Equal(5.0, (double)invertible.Determinant(), Tolerance);
        Assert.True(invertible.IsInvertible());
        Assert.False(singular.IsInvertible());
    }
}
