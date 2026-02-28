using AiDotNet.Extensions;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Extensions;

/// <summary>
/// Deep math integration tests for MatrixExtensions: matrix type classification,
/// structural operations, backward substitution, and block matrix operations.
/// </summary>
public class MatrixExtensionsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    // ============================
    // Identity Matrix Tests
    // ============================

    [Fact]
    public void IsIdentityMatrix_TrueForIdentity()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 1.0; m[2, 2] = 1.0;
        Assert.True(m.IsIdentityMatrix());
    }

    [Fact]
    public void IsIdentityMatrix_FalseForScaledIdentity()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 2.0; m[1, 1] = 2.0; m[2, 2] = 2.0;
        Assert.False(m.IsIdentityMatrix());
    }

    [Fact]
    public void IsIdentityMatrix_FalseForNonSquare()
    {
        var m = new Matrix<double>(2, 3);
        Assert.False(m.IsIdentityMatrix());
    }

    // ============================
    // Diagonal Matrix Tests
    // ============================

    [Fact]
    public void IsDiagonalMatrix_TrueForDiagonal()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 2.0; m[1, 1] = 5.0; m[2, 2] = -3.0;
        Assert.True(m.IsDiagonalMatrix());
    }

    [Fact]
    public void IsDiagonalMatrix_FalseForNonDiagonal()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[0, 1] = 0.5; m[1, 1] = 1.0; m[2, 2] = 1.0;
        Assert.False(m.IsDiagonalMatrix());
    }

    [Fact]
    public void IsDiagonalMatrix_TrueForZeroMatrix()
    {
        var m = new Matrix<double>(3, 3);
        Assert.True(m.IsDiagonalMatrix());
    }

    [Fact]
    public void IdentityIsDiagonal()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 1.0; m[2, 2] = 1.0;
        Assert.True(m.IsDiagonalMatrix());
    }

    // ============================
    // Scalar Matrix Tests
    // ============================

    [Fact]
    public void IsScalarMatrix_TrueForScaledIdentity()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 5.0; m[1, 1] = 5.0; m[2, 2] = 5.0;
        Assert.True(m.IsScalarMatrix());
    }

    [Fact]
    public void IsScalarMatrix_TrueForIdentity()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 1.0; m[2, 2] = 1.0;
        Assert.True(m.IsScalarMatrix());
    }

    [Fact]
    public void IsScalarMatrix_FalseForDifferentDiagonals()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 2.0; m[2, 2] = 1.0;
        Assert.False(m.IsScalarMatrix());
    }

    // ============================
    // Symmetric Matrix Tests
    // ============================

    [Fact]
    public void IsSymmetricMatrix_TrueForSymmetric()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0;
        m[1, 0] = 2.0; m[1, 1] = 4.0; m[1, 2] = 5.0;
        m[2, 0] = 3.0; m[2, 1] = 5.0; m[2, 2] = 6.0;
        Assert.True(m.IsSymmetricMatrix());
    }

    [Fact]
    public void IsSymmetricMatrix_FalseForNonSymmetric()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0;
        m[1, 0] = 3.0; m[1, 1] = 4.0; // m[0,1] != m[1,0]
        Assert.False(m.IsSymmetricMatrix());
    }

    [Fact]
    public void DiagonalIsAlwaysSymmetric()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 2.0; m[2, 2] = 3.0;
        Assert.True(m.IsSymmetricMatrix());
    }

    // ============================
    // Skew-Symmetric Matrix Tests
    // ============================

    [Fact]
    public void IsSkewSymmetricMatrix_TrueForSkewSymmetric()
    {
        // A = -A^T, diagonal must be zero
        var m = new Matrix<double>(3, 3);
        m[0, 1] = 2.0; m[0, 2] = -3.0;
        m[1, 0] = -2.0; m[1, 2] = 5.0;
        m[2, 0] = 3.0; m[2, 1] = -5.0;
        Assert.True(m.IsSkewSymmetricMatrix());
    }

    [Fact]
    public void IsSkewSymmetricMatrix_FalseForNonZeroDiagonal()
    {
        var m = new Matrix<double>(2, 2);
        m[0, 0] = 1.0; m[0, 1] = 2.0;
        m[1, 0] = -2.0; m[1, 1] = 0.0;
        Assert.False(m.IsSkewSymmetricMatrix());
    }

    [Fact]
    public void ZeroMatrixIsBothSymmetricAndSkewSymmetric()
    {
        var m = new Matrix<double>(3, 3);
        Assert.True(m.IsSymmetricMatrix());
        Assert.True(m.IsSkewSymmetricMatrix());
    }

    // ============================
    // Upper Triangular Matrix Tests
    // ============================

    [Fact]
    public void IsUpperTriangular_TrueForUpperTriangular()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0;
        m[1, 1] = 4.0; m[1, 2] = 5.0;
        m[2, 2] = 6.0;
        Assert.True(m.IsUpperTriangularMatrix());
    }

    [Fact]
    public void IsUpperTriangular_FalseForLowerElements()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0;
        m[1, 0] = 0.5; // non-zero below diagonal
        m[1, 1] = 4.0;
        Assert.False(m.IsUpperTriangularMatrix());
    }

    // ============================
    // Lower Triangular Matrix Tests
    // ============================

    [Fact]
    public void IsLowerTriangular_TrueForLowerTriangular()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0;
        m[1, 0] = 2.0; m[1, 1] = 3.0;
        m[2, 0] = 4.0; m[2, 1] = 5.0; m[2, 2] = 6.0;
        Assert.True(m.IsLowerTriangularMatrix());
    }

    [Fact]
    public void DiagonalIsBothUpperAndLowerTriangular()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 2.0; m[2, 2] = 3.0;
        Assert.True(m.IsUpperTriangularMatrix());
        Assert.True(m.IsLowerTriangularMatrix());
    }

    // ============================
    // Tridiagonal Matrix Tests
    // ============================

    [Fact]
    public void IsTridiagonal_TrueForTridiagonal()
    {
        // Non-zero only on main diagonal and first sub/super diagonals
        var m = new Matrix<double>(4, 4);
        m[0, 0] = 2.0; m[0, 1] = -1.0;
        m[1, 0] = -1.0; m[1, 1] = 2.0; m[1, 2] = -1.0;
        m[2, 1] = -1.0; m[2, 2] = 2.0; m[2, 3] = -1.0;
        m[3, 2] = -1.0; m[3, 3] = 2.0;
        Assert.True(m.IsTridiagonalMatrix());
    }

    [Fact]
    public void IsTridiagonal_FalseForFullMatrix()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0;
        m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0;
        m[2, 0] = 7.0; m[2, 1] = 8.0; m[2, 2] = 9.0;
        Assert.False(m.IsTridiagonalMatrix());
    }

    // ============================
    // Upper/Lower Bidiagonal Tests
    // ============================

    [Fact]
    public void IsUpperBidiagonal_TrueForUpperBidiagonal()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0;
        m[1, 1] = 3.0; m[1, 2] = 4.0;
        m[2, 2] = 5.0;
        Assert.True(m.IsUpperBidiagonalMatrix());
    }

    [Fact]
    public void IsLowerBidiagonal_TrueForLowerBidiagonal()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0;
        m[1, 0] = 2.0; m[1, 1] = 3.0;
        m[2, 1] = 4.0; m[2, 2] = 5.0;
        Assert.True(m.IsLowerBidiagonalMatrix());
    }

    // ============================
    // Band Matrix Tests
    // ============================

    [Fact]
    public void IsBandMatrix_TridiagonalIsBand11()
    {
        var m = new Matrix<double>(4, 4);
        m[0, 0] = 2.0; m[0, 1] = -1.0;
        m[1, 0] = -1.0; m[1, 1] = 2.0; m[1, 2] = -1.0;
        m[2, 1] = -1.0; m[2, 2] = 2.0; m[2, 3] = -1.0;
        m[3, 2] = -1.0; m[3, 3] = 2.0;
        Assert.True(m.IsBandMatrix(1, 1));
    }

    [Fact]
    public void IsBandMatrix_DiagonalIsBand00()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 2.0; m[2, 2] = 3.0;
        Assert.True(m.IsBandMatrix(0, 0));
    }

    // ============================
    // Square / Rectangular Tests
    // ============================

    [Fact]
    public void IsSquareMatrix_TrueForSquare()
    {
        var m = new Matrix<double>(3, 3);
        Assert.True(m.IsSquareMatrix());
    }

    [Fact]
    public void IsSquareMatrix_FalseForRectangular()
    {
        var m = new Matrix<double>(2, 3);
        Assert.False(m.IsSquareMatrix());
    }

    [Fact]
    public void IsRectangularMatrix_TrueForRectangular()
    {
        var m = new Matrix<double>(2, 3);
        Assert.True(m.IsRectangularMatrix());
    }

    [Fact]
    public void IsRectangularMatrix_FalseForSquare()
    {
        var m = new Matrix<double>(3, 3);
        Assert.False(m.IsRectangularMatrix());
    }

    // ============================
    // Sparse / Dense Tests
    // ============================

    [Fact]
    public void IsSparseMatrix_TrueForMostlyZeros()
    {
        var m = new Matrix<double>(4, 4);
        m[0, 0] = 1.0; // Only 1 out of 16 elements non-zero (6.25% density)
        Assert.True(m.IsSparseMatrix());
    }

    [Fact]
    public void IsDenseMatrix_TrueForMostlyNonZero()
    {
        var m = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m[i, j] = i * 3 + j + 1;
        Assert.True(m.IsDenseMatrix());
    }

    // ============================
    // AddConstantColumn Tests
    // ============================

    [Fact]
    public void AddConstantColumn_AddsOnesColumn()
    {
        var m = new Matrix<double>(2, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0;
        m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0;

        var result = m.AddConstantColumn(1.0);
        Assert.Equal(2, result.Rows);
        Assert.Equal(4, result.Columns);

        // First column should be 1.0
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(1.0, result[1, 0], Tolerance);

        // Original columns should be shifted right by 1
        Assert.Equal(1.0, result[0, 1], Tolerance);
        Assert.Equal(2.0, result[0, 2], Tolerance);
        Assert.Equal(3.0, result[0, 3], Tolerance);
    }

    // ============================
    // ToVector Tests
    // ============================

    [Fact]
    public void ToVector_FlattensMatrix()
    {
        var m = new Matrix<double>(2, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0;
        m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0;

        var v = m.ToVector();
        Assert.Equal(6, v.Length);
        Assert.Equal(1.0, v[0], Tolerance);
        Assert.Equal(2.0, v[1], Tolerance);
        Assert.Equal(3.0, v[2], Tolerance);
        Assert.Equal(4.0, v[3], Tolerance);
        Assert.Equal(5.0, v[4], Tolerance);
        Assert.Equal(6.0, v[5], Tolerance);
    }

    // ============================
    // SumColumns Tests
    // ============================

    [Fact]
    public void SumColumns_HandComputed()
    {
        // [[1,2,3],[4,5,6]] -> sum columns = [5, 7, 9]
        var m = new Matrix<double>(2, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0;
        m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0;

        var sums = m.SumColumns();
        Assert.Equal(3, sums.Length);
        Assert.Equal(5.0, sums[0], Tolerance);
        Assert.Equal(7.0, sums[1], Tolerance);
        Assert.Equal(9.0, sums[2], Tolerance);
    }

    // ============================
    // GetColumn Tests
    // ============================

    [Fact]
    public void GetColumn_HandComputed()
    {
        var m = new Matrix<double>(3, 2);
        m[0, 0] = 1.0; m[0, 1] = 10.0;
        m[1, 0] = 2.0; m[1, 1] = 20.0;
        m[2, 0] = 3.0; m[2, 1] = 30.0;

        var col0 = m.GetColumn(0);
        Assert.Equal(3, col0.Length);
        Assert.Equal(1.0, col0[0], Tolerance);
        Assert.Equal(2.0, col0[1], Tolerance);
        Assert.Equal(3.0, col0[2], Tolerance);

        var col1 = m.GetColumn(1);
        Assert.Equal(10.0, col1[0], Tolerance);
        Assert.Equal(20.0, col1[1], Tolerance);
        Assert.Equal(30.0, col1[2], Tolerance);
    }

    // ============================
    // AddVectorToEachRow Tests
    // ============================

    [Fact]
    public void AddVectorToEachRow_HandComputed()
    {
        var m = new Matrix<double>(2, 3);
        m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0;
        m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0;

        var v = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var result = m.AddVectorToEachRow(v);

        Assert.Equal(11.0, result[0, 0], Tolerance);
        Assert.Equal(22.0, result[0, 1], Tolerance);
        Assert.Equal(33.0, result[0, 2], Tolerance);
        Assert.Equal(14.0, result[1, 0], Tolerance);
        Assert.Equal(25.0, result[1, 1], Tolerance);
        Assert.Equal(36.0, result[1, 2], Tolerance);
    }

    // ============================
    // Backward Substitution Tests
    // ============================

    [Fact]
    public void BackwardSubstitution_UpperTriangular_HandComputed()
    {
        // Solve Ux = b where U is upper triangular
        // U = [[2, -1, 1], [0, 3, -1], [0, 0, 4]]
        // b = [3, 5, 8]
        //
        // x[2] = 8/4 = 2
        // x[1] = (5 - (-1)*2)/3 = 7/3
        // x[0] = (3 - (-1)*(7/3) + 1*2)/2 = (3 + 7/3 - 2)/2 = (1 + 7/3)/2 = (10/3)/2 = 5/3
        //
        // Wait let me recompute:
        // x[0] = (b[0] - U[0,1]*x[1] - U[0,2]*x[2]) / U[0,0]
        // x[0] = (3 - (-1)*(7/3) - 1*2) / 2 = (3 + 7/3 - 2) / 2 = (1 + 7/3) / 2 = (10/3) / 2 = 5/3

        var u = new Matrix<double>(3, 3);
        u[0, 0] = 2.0; u[0, 1] = -1.0; u[0, 2] = 1.0;
        u[1, 1] = 3.0; u[1, 2] = -1.0;
        u[2, 2] = 4.0;

        var b = new Vector<double>(new[] { 3.0, 5.0, 8.0 });
        var x = u.BackwardSubstitution(b);

        Assert.Equal(3, x.Length);
        Assert.Equal(2.0, x[2], Tolerance);           // 8/4 = 2
        Assert.Equal(7.0 / 3.0, x[1], Tolerance);     // (5+2)/3 = 7/3
        Assert.Equal(5.0 / 3.0, x[0], Tolerance);     // (3+7/3-2)/2 = 5/3
    }

    [Fact]
    public void BackwardSubstitution_DiagonalMatrix()
    {
        // For diagonal matrix, x[i] = b[i] / d[i]
        var d = new Matrix<double>(3, 3);
        d[0, 0] = 2.0; d[1, 1] = 5.0; d[2, 2] = 10.0;

        var b = new Vector<double>(new[] { 4.0, 15.0, 30.0 });
        var x = d.BackwardSubstitution(b);

        Assert.Equal(2.0, x[0], Tolerance);
        Assert.Equal(3.0, x[1], Tolerance);
        Assert.Equal(3.0, x[2], Tolerance);
    }

    [Fact]
    public void BackwardSubstitution_IdentityMatrix()
    {
        // Ix = b => x = b
        var identity = new Matrix<double>(3, 3);
        identity[0, 0] = 1.0; identity[1, 1] = 1.0; identity[2, 2] = 1.0;

        var b = new Vector<double>(new[] { 3.14, 2.71, 1.41 });
        var x = identity.BackwardSubstitution(b);

        for (int i = 0; i < 3; i++)
            Assert.Equal(b[i], x[i], Tolerance);
    }

    // ============================
    // Block Matrix Tests
    // ============================

    [Fact]
    public void IsBlockMatrix_TrueForConsistentBlocks()
    {
        // 4x4 matrix with 2x2 blocks where each block has all identical elements
        // IsConsistentBlock requires all elements in the block to be the same value
        var m = new Matrix<double>(4, 4);
        // Block (0,0) - all 3.0
        m[0, 0] = 3.0; m[0, 1] = 3.0;
        m[1, 0] = 3.0; m[1, 1] = 3.0;
        // Block (0,1) - all 0.0 (default)
        // Block (1,0) - all 0.0 (default)
        // Block (1,1) - all 5.0
        m[2, 2] = 5.0; m[2, 3] = 5.0;
        m[3, 2] = 5.0; m[3, 3] = 5.0;
        Assert.True(m.IsBlockMatrix(2, 2));
    }

    [Fact]
    public void GetBlock_ExtractsSubmatrix()
    {
        var m = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i, j] = i * 4 + j + 1;

        var block = m.GetBlock(1, 1, 2, 2);
        Assert.Equal(2, block.Rows);
        Assert.Equal(2, block.Columns);
        Assert.Equal(6.0, block[0, 0], Tolerance);  // m[1,1]
        Assert.Equal(7.0, block[0, 1], Tolerance);  // m[1,2]
        Assert.Equal(10.0, block[1, 0], Tolerance); // m[2,1]
        Assert.Equal(11.0, block[1, 1], Tolerance); // m[2,2]
    }

    // ============================
    // Matrix Type Classification Hierarchy
    // ============================

    [Fact]
    public void IdentityMatrix_IsAllSpecialTypes()
    {
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 1.0; m[2, 2] = 1.0;

        Assert.True(m.IsIdentityMatrix());
        Assert.True(m.IsDiagonalMatrix());
        Assert.True(m.IsScalarMatrix());
        Assert.True(m.IsSymmetricMatrix());
        Assert.True(m.IsUpperTriangularMatrix());
        Assert.True(m.IsLowerTriangularMatrix());
        Assert.True(m.IsSquareMatrix());
    }

    [Fact]
    public void DiagonalMatrix_TypeHierarchy()
    {
        // Diagonal but not identity and not scalar
        var m = new Matrix<double>(3, 3);
        m[0, 0] = 1.0; m[1, 1] = 2.0; m[2, 2] = 3.0;

        Assert.True(m.IsDiagonalMatrix());
        Assert.True(m.IsSymmetricMatrix());
        Assert.True(m.IsUpperTriangularMatrix());
        Assert.True(m.IsLowerTriangularMatrix());
        Assert.False(m.IsScalarMatrix());
        Assert.False(m.IsIdentityMatrix());
    }

    [Fact]
    public void SymmetricTridiagonal_TypeHierarchy()
    {
        // Symmetric tridiagonal (common in physics: 1D Laplacian)
        var m = new Matrix<double>(4, 4);
        m[0, 0] = 2.0; m[0, 1] = -1.0;
        m[1, 0] = -1.0; m[1, 1] = 2.0; m[1, 2] = -1.0;
        m[2, 1] = -1.0; m[2, 2] = 2.0; m[2, 3] = -1.0;
        m[3, 2] = -1.0; m[3, 3] = 2.0;

        Assert.True(m.IsSymmetricMatrix());
        Assert.True(m.IsTridiagonalMatrix());
        Assert.True(m.IsBandMatrix(1, 1));
        Assert.False(m.IsDiagonalMatrix());
        Assert.False(m.IsUpperTriangularMatrix());
    }

    // ============================
    // ConsistentBlock Tests
    // ============================

    [Fact]
    public void IsConsistentBlock_AllSameValue()
    {
        var m = new Matrix<double>(2, 2);
        m[0, 0] = 5.0; m[0, 1] = 5.0;
        m[1, 0] = 5.0; m[1, 1] = 5.0;
        Assert.True(m.IsConsistentBlock());
    }

    [Fact]
    public void IsConsistentBlock_DifferentValues_False()
    {
        var m = new Matrix<double>(2, 2);
        m[0, 0] = 1.0; m[0, 1] = 2.0;
        m[1, 0] = 1.0; m[1, 1] = 1.0;
        Assert.False(m.IsConsistentBlock());
    }
}
