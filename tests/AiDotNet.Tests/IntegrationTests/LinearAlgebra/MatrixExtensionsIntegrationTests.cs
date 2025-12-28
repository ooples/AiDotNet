using AiDotNet.Extensions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LinearAlgebra;

/// <summary>
/// Integration tests for MatrixExtensions methods.
/// </summary>
public class MatrixExtensionsIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region AddConstantColumn Tests

    [Fact]
    public void AddConstantColumn_AddsColumnAtFront()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;

        var result = matrix.AddConstantColumn(1.0);

        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
        Assert.Equal(1, result[0, 0]); // constant column
        Assert.Equal(1, result[1, 0]); // constant column
        Assert.Equal(1, result[0, 1]); // original data
        Assert.Equal(2, result[0, 2]);
        Assert.Equal(3, result[1, 1]);
        Assert.Equal(4, result[1, 2]);
    }

    #endregion

    #region ToVector Tests

    [Fact]
    public void ToVector_FlattensMatrixRowMajor()
    {
        var matrix = new Matrix<double>(2, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;

        var result = matrix.ToVector();

        Assert.Equal(6, result.Length);
        Assert.Equal(1, result[0]);
        Assert.Equal(2, result[1]);
        Assert.Equal(3, result[2]);
        Assert.Equal(4, result[3]);
        Assert.Equal(5, result[4]);
        Assert.Equal(6, result[5]);
    }

    #endregion

    #region AddVectorToEachRow Tests

    [Fact]
    public void AddVectorToEachRow_AddsVectorToAllRows()
    {
        var matrix = new Matrix<double>(2, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;

        var vector = new Vector<double>([10, 20, 30]);
        var result = matrix.AddVectorToEachRow(vector);

        Assert.Equal(11, result[0, 0]);
        Assert.Equal(22, result[0, 1]);
        Assert.Equal(33, result[0, 2]);
        Assert.Equal(14, result[1, 0]);
        Assert.Equal(25, result[1, 1]);
        Assert.Equal(36, result[1, 2]);
    }

    #endregion

    #region SumColumns Tests

    [Fact]
    public void SumColumns_ReturnsColumnSums()
    {
        var matrix = new Matrix<double>(3, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;
        matrix[2, 0] = 5; matrix[2, 1] = 6;

        var sums = matrix.SumColumns();

        Assert.Equal(2, sums.Length);
        Assert.Equal(9, sums[0]); // 1 + 3 + 5
        Assert.Equal(12, sums[1]); // 2 + 4 + 6
    }

    #endregion

    #region GetColumn Tests

    [Fact]
    public void GetColumn_ExtractsCorrectColumn()
    {
        var matrix = new Matrix<double>(3, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;
        matrix[2, 0] = 5; matrix[2, 1] = 6;

        var col = matrix.GetColumn(1);

        Assert.Equal(3, col.Length);
        Assert.Equal(2, col[0]);
        Assert.Equal(4, col[1]);
        Assert.Equal(6, col[2]);
    }

    #endregion

    #region BackwardSubstitution Tests

    [Fact]
    public void BackwardSubstitution_SolvesUpperTriangularSystem()
    {
        // Upper triangular matrix A
        var A = new Matrix<double>(3, 3);
        A[0, 0] = 1; A[0, 1] = 2; A[0, 2] = 3;
        A[1, 0] = 0; A[1, 1] = 4; A[1, 2] = 5;
        A[2, 0] = 0; A[2, 1] = 0; A[2, 2] = 6;

        var b = new Vector<double>([14, 17, 6]);
        var x = A.BackwardSubstitution(b);

        // Verify Ax = b
        Assert.Equal(3, x.Length);
        Assert.Equal(1, x[2], Tolerance); // x3 = 6/6 = 1
        Assert.Equal(3, x[1], Tolerance); // x2 = (17 - 5*1)/4 = 3
        Assert.Equal(5, x[0], Tolerance); // x1 = (14 - 2*3 - 3*1)/1 = 5
    }

    #endregion

    #region ForwardSubstitution Tests

    [Fact]
    public void ForwardSubstitution_SolvesLowerTriangularSystem()
    {
        // Lower triangular matrix A
        var A = new Matrix<double>(3, 3);
        A[0, 0] = 2; A[0, 1] = 0; A[0, 2] = 0;
        A[1, 0] = 3; A[1, 1] = 4; A[1, 2] = 0;
        A[2, 0] = 5; A[2, 1] = 6; A[2, 2] = 7;

        var b = new Vector<double>([2, 11, 45]);
        var x = A.ForwardSubstitution(b);

        Assert.Equal(3, x.Length);
        Assert.Equal(1, x[0], Tolerance); // x1 = 2/2 = 1
        Assert.Equal(2, x[1], Tolerance); // x2 = (11 - 3*1)/4 = 2
        Assert.Equal(4, x[2], Tolerance); // x3 = (45 - 5*1 - 6*2)/7 = 28/7 = 4
    }

    #endregion

    #region GetBlock Tests

    [Fact]
    public void GetBlock_ExtractsSubmatrix()
    {
        var matrix = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                matrix[i, j] = i * 4 + j + 1;

        var block = matrix.GetBlock(1, 1, 2, 2);

        Assert.Equal(2, block.Rows);
        Assert.Equal(2, block.Columns);
        Assert.Equal(6, block[0, 0]);  // matrix[1,1]
        Assert.Equal(7, block[0, 1]);  // matrix[1,2]
        Assert.Equal(10, block[1, 0]); // matrix[2,1]
        Assert.Equal(11, block[1, 1]); // matrix[2,2]
    }

    #endregion

    #region Matrix Type Detection Tests

    [Fact]
    public void IsSquareMatrix_SquareMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        Assert.True(matrix.IsSquareMatrix());
    }

    [Fact]
    public void IsSquareMatrix_RectangularMatrix_ReturnsFalse()
    {
        var matrix = new Matrix<double>(2, 3);
        Assert.False(matrix.IsSquareMatrix());
    }

    [Fact]
    public void IsRectangularMatrix_NonSquareMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(2, 3);
        Assert.True(matrix.IsRectangularMatrix());
    }

    [Fact]
    public void IsSymmetricMatrix_SymmetricMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 2; matrix[1, 1] = 4; matrix[1, 2] = 5;
        matrix[2, 0] = 3; matrix[2, 1] = 5; matrix[2, 2] = 6;

        Assert.True(matrix.IsSymmetricMatrix());
    }

    [Fact]
    public void IsSymmetricMatrix_NonSymmetricMatrix_ReturnsFalse()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;

        Assert.False(matrix.IsSymmetricMatrix());
    }

    [Fact]
    public void IsDiagonalMatrix_DiagonalMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[1, 1] = 2; matrix[2, 2] = 3;

        Assert.True(matrix.IsDiagonalMatrix());
    }

    [Fact]
    public void IsIdentityMatrix_IdentityMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[1, 1] = 1; matrix[2, 2] = 1;
        Assert.True(matrix.IsIdentityMatrix());
    }

    [Fact]
    public void IsUpperTriangularMatrix_UpperTriangular_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 0; matrix[1, 1] = 4; matrix[1, 2] = 5;
        matrix[2, 0] = 0; matrix[2, 1] = 0; matrix[2, 2] = 6;

        Assert.True(matrix.IsUpperTriangularMatrix());
    }

    [Fact]
    public void IsLowerTriangularMatrix_LowerTriangular_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 0; matrix[0, 2] = 0;
        matrix[1, 0] = 2; matrix[1, 1] = 3; matrix[1, 2] = 0;
        matrix[2, 0] = 4; matrix[2, 1] = 5; matrix[2, 2] = 6;

        Assert.True(matrix.IsLowerTriangularMatrix());
    }

    [Fact]
    public void IsZeroMatrix_ZeroMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        Assert.True(matrix.IsZeroMatrix());
    }

    [Fact]
    public void IsSparseMatrix_MostlyZeros_ReturnsTrue()
    {
        var matrix = new Matrix<double>(10, 10);
        matrix[0, 0] = 1;
        matrix[5, 5] = 2;

        Assert.True(matrix.IsSparseMatrix());
    }

    [Fact]
    public void IsDenseMatrix_MostlyNonZeros_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                matrix[i, j] = 1;

        Assert.True(matrix.IsDenseMatrix());
    }

    [Fact]
    public void IsScalarMatrix_ScalarMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 5; matrix[1, 1] = 5; matrix[2, 2] = 5;

        Assert.True(matrix.IsScalarMatrix());
    }

    [Fact]
    public void IsTridiagonalMatrix_Tridiagonal_ReturnsTrue()
    {
        var matrix = new Matrix<double>(4, 4);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4; matrix[1, 2] = 5;
        matrix[2, 1] = 6; matrix[2, 2] = 7; matrix[2, 3] = 8;
        matrix[3, 2] = 9; matrix[3, 3] = 10;

        Assert.True(matrix.IsTridiagonalMatrix());
    }

    [Fact]
    public void IsUpperBidiagonalMatrix_UpperBidiagonal_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 1] = 3; matrix[1, 2] = 4;
        matrix[2, 2] = 5;

        Assert.True(matrix.IsUpperBidiagonalMatrix());
    }

    [Fact]
    public void IsLowerBidiagonalMatrix_LowerBidiagonal_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1;
        matrix[1, 0] = 2; matrix[1, 1] = 3;
        matrix[2, 1] = 4; matrix[2, 2] = 5;

        Assert.True(matrix.IsLowerBidiagonalMatrix());
    }

    [Fact]
    public void IsSkewSymmetricMatrix_SkewSymmetric_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 0; matrix[0, 1] = 2; matrix[0, 2] = -3;
        matrix[1, 0] = -2; matrix[1, 1] = 0; matrix[1, 2] = 4;
        matrix[2, 0] = 3; matrix[2, 1] = -4; matrix[2, 2] = 0;

        Assert.True(matrix.IsSkewSymmetricMatrix());
    }

    [Fact]
    public void IsPermutationMatrix_PermutationMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 1] = 1; // row 0 -> column 1
        matrix[1, 2] = 1; // row 1 -> column 2
        matrix[2, 0] = 1; // row 2 -> column 0

        Assert.True(matrix.IsPermutationMatrix());
    }

    [Fact]
    public void IsStochasticMatrix_StochasticMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 0.2; matrix[0, 1] = 0.3; matrix[0, 2] = 0.5;
        matrix[1, 0] = 0.4; matrix[1, 1] = 0.4; matrix[1, 2] = 0.2;
        matrix[2, 0] = 0.1; matrix[2, 1] = 0.6; matrix[2, 2] = 0.3;

        Assert.True(matrix.IsStochasticMatrix());
    }

    [Fact]
    public void IsDoublyStochasticMatrix_DoublyStochastic_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 0.5; matrix[0, 1] = 0.25; matrix[0, 2] = 0.25;
        matrix[1, 0] = 0.25; matrix[1, 1] = 0.5; matrix[1, 2] = 0.25;
        matrix[2, 0] = 0.25; matrix[2, 1] = 0.25; matrix[2, 2] = 0.5;

        Assert.True(matrix.IsDoublyStochasticMatrix());
    }

    [Fact]
    public void IsToeplitzMatrix_ToeplitzMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 4; matrix[1, 1] = 1; matrix[1, 2] = 2;
        matrix[2, 0] = 5; matrix[2, 1] = 4; matrix[2, 2] = 1;

        Assert.True(matrix.IsToeplitzMatrix());
    }

    [Fact]
    public void IsHankelMatrix_HankelMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 2; matrix[1, 1] = 3; matrix[1, 2] = 4;
        matrix[2, 0] = 3; matrix[2, 1] = 4; matrix[2, 2] = 5;

        Assert.True(matrix.IsHankelMatrix());
    }

    [Fact]
    public void IsCirculantMatrix_CirculantMatrix_ReturnsTrue()
    {
        // Implementation uses left circular shift convention
        // Row i, col j should equal Row 0, col (j + i) % cols
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 2; matrix[1, 1] = 3; matrix[1, 2] = 1; // left shift by 1
        matrix[2, 0] = 3; matrix[2, 1] = 1; matrix[2, 2] = 2; // left shift by 2

        Assert.True(matrix.IsCirculantMatrix());
    }

    #endregion

    #region Matrix Operations Tests

    [Fact]
    public void Transpose_ReturnsTransposedMatrix()
    {
        var matrix = new Matrix<double>(2, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;

        var result = matrix.Transpose();

        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
        Assert.Equal(1, result[0, 0]);
        Assert.Equal(4, result[0, 1]);
        Assert.Equal(2, result[1, 0]);
        Assert.Equal(5, result[1, 1]);
        Assert.Equal(3, result[2, 0]);
        Assert.Equal(6, result[2, 1]);
    }

    [Fact]
    public void Negate_NegatesAllElements()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1; matrix[0, 1] = -2;
        matrix[1, 0] = 3; matrix[1, 1] = -4;

        var result = matrix.Negate();

        Assert.Equal(-1, result[0, 0]);
        Assert.Equal(2, result[0, 1]);
        Assert.Equal(-3, result[1, 0]);
        Assert.Equal(4, result[1, 1]);
    }

    [Fact]
    public void FrobeniusNorm_ReturnsCorrectNorm()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;

        var norm = matrix.FrobeniusNorm();

        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        Assert.Equal(Math.Sqrt(30), norm, Tolerance);
    }

    [Fact]
    public void Determinant_2x2Matrix_ReturnsCorrectValue()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;

        var det = matrix.Determinant();

        // det = 1*4 - 2*3 = -2
        Assert.Equal(-2, det, Tolerance);
    }

    [Fact]
    public void Determinant_3x3Matrix_ReturnsCorrectValue()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;
        matrix[2, 0] = 7; matrix[2, 1] = 8; matrix[2, 2] = 9;

        var det = matrix.Determinant();

        // This matrix is singular (rows are linearly dependent)
        Assert.Equal(0, det, 1e-6);
    }

    #endregion

    #region Inverse Tests

    [Fact]
    public void Inverse_2x2Matrix_ReturnsCorrectInverse()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 4; matrix[0, 1] = 7;
        matrix[1, 0] = 2; matrix[1, 1] = 6;

        var inverse = matrix.Inverse();
        var product = matrix * inverse;

        // Verify product is identity
        Assert.Equal(1, product[0, 0], Tolerance);
        Assert.Equal(0, product[0, 1], Tolerance);
        Assert.Equal(0, product[1, 0], Tolerance);
        Assert.Equal(1, product[1, 1], Tolerance);
    }

    [Fact]
    public void InvertDiagonalMatrix_ReturnsCorrectInverse()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 2; matrix[1, 1] = 4; matrix[2, 2] = 5;

        var inverse = matrix.InvertDiagonalMatrix();

        Assert.Equal(0.5, inverse[0, 0], Tolerance);
        Assert.Equal(0.25, inverse[1, 1], Tolerance);
        Assert.Equal(0.2, inverse[2, 2], Tolerance);
    }

    [Fact]
    public void InvertUpperTriangularMatrix_ReturnsCorrectInverse()
    {
        // Upper triangular matrix
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 0; matrix[1, 1] = 4; matrix[1, 2] = 5;
        matrix[2, 0] = 0; matrix[2, 1] = 0; matrix[2, 2] = 6;

        var inverse = matrix.InvertUpperTriangularMatrix();
        var product = matrix * inverse;

        // Product should be identity matrix
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, product[i, j], 1e-10);
            }
        }
    }

    #endregion

    #region PointwiseMultiply Tests

    [Fact]
    public void PointwiseMultiply_TwoMatrices_ReturnsHadamardProduct()
    {
        var a = new Matrix<double>(2, 2);
        a[0, 0] = 1; a[0, 1] = 2;
        a[1, 0] = 3; a[1, 1] = 4;

        var b = new Matrix<double>(2, 2);
        b[0, 0] = 5; b[0, 1] = 6;
        b[1, 0] = 7; b[1, 1] = 8;

        var result = a.PointwiseMultiply(b);

        Assert.Equal(5, result[0, 0]);
        Assert.Equal(12, result[0, 1]);
        Assert.Equal(21, result[1, 0]);
        Assert.Equal(32, result[1, 1]);
    }

    [Fact]
    public void PointwiseMultiply_MatrixAndVector_ScalesRowsByVectorElements()
    {
        // Vector length must match rows - scales each row by corresponding vector element
        var matrix = new Matrix<double>(3, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;
        matrix[2, 0] = 5; matrix[2, 1] = 6;

        var vector = new Vector<double>([2, 3, 4]);
        var result = matrix.PointwiseMultiply(vector);

        Assert.Equal(2, result[0, 0]);  // 1*2
        Assert.Equal(4, result[0, 1]);  // 2*2
        Assert.Equal(9, result[1, 0]);  // 3*3
        Assert.Equal(12, result[1, 1]); // 4*3
        Assert.Equal(20, result[2, 0]); // 5*4
        Assert.Equal(24, result[2, 1]); // 6*4
    }

    #endregion

    #region Submatrix Tests

    [Fact]
    public void Submatrix_ExtractsCorrectRegion()
    {
        var matrix = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                matrix[i, j] = i * 4 + j;

        var sub = matrix.Submatrix(1, 1, 2, 2);

        Assert.Equal(2, sub.Rows);
        Assert.Equal(2, sub.Columns);
        Assert.Equal(5, sub[0, 0]);
        Assert.Equal(6, sub[0, 1]);
        Assert.Equal(9, sub[1, 0]);
        Assert.Equal(10, sub[1, 1]);
    }

    [Fact]
    public void Submatrix_ByIndices_ExtractsRows()
    {
        var matrix = new Matrix<double>(4, 3);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                matrix[i, j] = i * 3 + j;

        var sub = matrix.Submatrix([0, 2, 3]);

        Assert.Equal(3, sub.Rows);
        Assert.Equal(3, sub.Columns);
        Assert.Equal(0, sub[0, 0]); // row 0
        Assert.Equal(6, sub[1, 0]); // row 2
        Assert.Equal(9, sub[2, 0]); // row 3
    }

    #endregion

    #region AddColumn Tests

    [Fact]
    public void AddColumn_AppendsColumn()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;

        var col = new Vector<double>([5, 6]);
        var result = matrix.AddColumn(col);

        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
        Assert.Equal(5, result[0, 2]);
        Assert.Equal(6, result[1, 2]);
    }

    #endregion

    #region GetColumns Tests

    [Fact]
    public void GetColumns_ExtractsSelectedColumns()
    {
        var matrix = new Matrix<double>(3, 4);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                matrix[i, j] = i * 4 + j;

        var cols = matrix.GetColumns([1, 3]);

        // Extracting 2 columns from 3x4 matrix should give 3x2 matrix
        Assert.Equal(3, cols.Rows);
        Assert.Equal(2, cols.Columns);
        Assert.Equal(1, cols[0, 0]); // matrix[0,1]
        Assert.Equal(3, cols[0, 1]); // matrix[0,3]
        Assert.Equal(5, cols[1, 0]); // matrix[1,1]
        Assert.Equal(7, cols[1, 1]); // matrix[1,3]
    }

    #endregion

    #region GetRow Tests

    [Fact]
    public void GetRow_ReturnsCorrectRow()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;
        matrix[2, 0] = 7; matrix[2, 1] = 8; matrix[2, 2] = 9;

        var row = matrix.GetRow(1);

        Assert.Equal(3, row.Length);
        Assert.Equal(4, row[0]);
        Assert.Equal(5, row[1]);
        Assert.Equal(6, row[2]);
    }

    #endregion

    #region GetRowRange Tests

    [Fact]
    public void GetRowRange_ExtractsRowSubset()
    {
        var matrix = new Matrix<double>(4, 3);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                matrix[i, j] = i * 3 + j;

        var sub = matrix.GetRowRange(1, 2);

        Assert.Equal(2, sub.Rows);
        Assert.Equal(3, sub.Columns);
        Assert.Equal(3, sub[0, 0]); // row 1
        Assert.Equal(6, sub[1, 0]); // row 2
    }

    #endregion

    #region SwapRows Tests

    [Fact]
    public void SwapRows_SwapsRowsInPlace()
    {
        var matrix = new Matrix<double>(3, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;
        matrix[2, 0] = 5; matrix[2, 1] = 6;

        matrix.SwapRows(0, 2);

        Assert.Equal(5, matrix[0, 0]);
        Assert.Equal(6, matrix[0, 1]);
        Assert.Equal(1, matrix[2, 0]);
        Assert.Equal(2, matrix[2, 1]);
    }

    #endregion

    #region KroneckerProduct Tests

    [Fact]
    public void KroneckerProduct_ReturnsCorrectResult()
    {
        var a = new Matrix<double>(2, 2);
        a[0, 0] = 1; a[0, 1] = 2;
        a[1, 0] = 3; a[1, 1] = 4;

        var b = new Matrix<double>(2, 2);
        b[0, 0] = 0; b[0, 1] = 5;
        b[1, 0] = 6; b[1, 1] = 7;

        var result = a.KroneckerProduct(b);

        Assert.Equal(4, result.Rows);
        Assert.Equal(4, result.Columns);
        // First block (a[0,0] * b)
        Assert.Equal(0, result[0, 0]);  // 1 * 0
        Assert.Equal(5, result[0, 1]);  // 1 * 5
        Assert.Equal(6, result[1, 0]);  // 1 * 6
        Assert.Equal(7, result[1, 1]);  // 1 * 7
    }

    #endregion

    #region Flatten Tests

    [Fact]
    public void Flatten_FlattensToVector()
    {
        var matrix = new Matrix<double>(2, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;

        var flat = matrix.Flatten();

        Assert.Equal(6, flat.Length);
        Assert.Equal(1, flat[0]);
        Assert.Equal(6, flat[5]);
    }

    #endregion

    #region Reshape Tests

    [Fact]
    public void Reshape_ChangesMatrixShape()
    {
        var matrix = new Matrix<double>(2, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;

        var reshaped = matrix.Reshape(3, 2);

        Assert.Equal(3, reshaped.Rows);
        Assert.Equal(2, reshaped.Columns);
        Assert.Equal(1, reshaped[0, 0]);
        Assert.Equal(2, reshaped[0, 1]);
        Assert.Equal(3, reshaped[1, 0]);
        Assert.Equal(4, reshaped[1, 1]);
    }

    #endregion

    #region Extract Tests

    [Fact]
    public void Extract_ExtractsTopLeftCorner()
    {
        var matrix = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                matrix[i, j] = i * 4 + j;

        var extracted = matrix.Extract(2, 3);

        Assert.Equal(2, extracted.Rows);
        Assert.Equal(3, extracted.Columns);
        Assert.Equal(0, extracted[0, 0]);
        Assert.Equal(1, extracted[0, 1]);
        Assert.Equal(2, extracted[0, 2]);
    }

    #endregion

    #region RowWiseArgmax Tests

    [Fact]
    public void RowWiseArgmax_ReturnsIndicesOfMaxInEachRow()
    {
        var matrix = new Matrix<double>(3, 4);
        matrix[0, 0] = 1; matrix[0, 1] = 5; matrix[0, 2] = 2; matrix[0, 3] = 3;
        matrix[1, 0] = 7; matrix[1, 1] = 2; matrix[1, 2] = 3; matrix[1, 3] = 1;
        matrix[2, 0] = 1; matrix[2, 1] = 2; matrix[2, 2] = 3; matrix[2, 3] = 9;

        var argmax = matrix.RowWiseArgmax();

        Assert.Equal(3, argmax.Length);
        Assert.Equal(1, argmax[0]); // max at column 1
        Assert.Equal(0, argmax[1]); // max at column 0
        Assert.Equal(3, argmax[2]); // max at column 3
    }

    #endregion

    #region SetSubmatrix Tests

    [Fact]
    public void SetSubmatrix_SetsValuesInPlace()
    {
        var matrix = new Matrix<double>(4, 4);
        var sub = new Matrix<double>(2, 2);
        sub[0, 0] = 9; sub[0, 1] = 8;
        sub[1, 0] = 7; sub[1, 1] = 6;

        matrix.SetSubmatrix(1, 1, sub);

        Assert.Equal(9, matrix[1, 1]);
        Assert.Equal(8, matrix[1, 2]);
        Assert.Equal(7, matrix[2, 1]);
        Assert.Equal(6, matrix[2, 2]);
    }

    #endregion

    #region GetSubColumn Tests

    [Fact]
    public void GetSubColumn_ExtractsColumnPortion()
    {
        var matrix = new Matrix<double>(5, 3);
        for (int i = 0; i < 5; i++)
            matrix[i, 1] = i * 10;

        var sub = matrix.GetSubColumn(1, 2, 2);

        Assert.Equal(2, sub.Length);
        Assert.Equal(20, sub[0]); // matrix[2,1]
        Assert.Equal(30, sub[1]); // matrix[3,1]
    }

    #endregion

    #region GetRank Tests

    [Fact]
    public void GetRank_IdentityMatrix_ReturnsFullRank()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[1, 1] = 1; matrix[2, 2] = 1;

        var rank = matrix.GetRank(1e-10);

        Assert.Equal(3, rank);
    }

    [Fact]
    public void GetRank_RankDeficientMatrix_ReturnsCorrectRank()
    {
        // All rows are multiples of the first row: rank should be 1
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
        matrix[1, 0] = 2; matrix[1, 1] = 4; matrix[1, 2] = 6;
        matrix[2, 0] = 3; matrix[2, 1] = 6; matrix[2, 2] = 9;

        var rank = matrix.GetRank(1e-10);

        Assert.Equal(1, rank);
    }

    [Fact]
    public void GetRank_ZeroMatrix_ReturnsZero()
    {
        var matrix = new Matrix<double>(3, 3);

        var rank = matrix.GetRank(1e-10);

        Assert.Equal(0, rank);
    }

    #endregion

    #region Complex Matrix Tests

    [Fact]
    public void ToComplexMatrix_ConvertsRealToComplex()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4;

        var complex = matrix.ToComplexMatrix();

        Assert.Equal(2, complex.Rows);
        Assert.Equal(2, complex.Columns);
        Assert.Equal(1, complex[0, 0].Real);
        Assert.Equal(0, complex[0, 0].Imaginary);
    }

    [Fact]
    public void ToRealMatrix_ExtractsRealPart()
    {
        var complex = new Matrix<Complex<double>>(2, 2);
        complex[0, 0] = new Complex<double>(1, 2);
        complex[0, 1] = new Complex<double>(3, 4);
        complex[1, 0] = new Complex<double>(5, 6);
        complex[1, 1] = new Complex<double>(7, 8);

        var real = complex.ToRealMatrix();

        Assert.Equal(1, real[0, 0]);
        Assert.Equal(3, real[0, 1]);
        Assert.Equal(5, real[1, 0]);
        Assert.Equal(7, real[1, 1]);
    }

    [Fact]
    public void ConjugateTranspose_ReturnsHermitianTranspose()
    {
        var matrix = new Matrix<Complex<double>>(2, 2);
        matrix[0, 0] = new Complex<double>(1, 2);
        matrix[0, 1] = new Complex<double>(3, 4);
        matrix[1, 0] = new Complex<double>(5, 6);
        matrix[1, 1] = new Complex<double>(7, 8);

        var result = matrix.ConjugateTranspose();

        Assert.Equal(1, result[0, 0].Real);
        Assert.Equal(-2, result[0, 0].Imaginary); // conjugate
        Assert.Equal(5, result[0, 1].Real); // transposed
        Assert.Equal(-6, result[0, 1].Imaginary); // conjugate
    }

    #endregion

    #region IsBandMatrix Tests

    [Fact]
    public void IsBandMatrix_TridiagonalMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<double>(4, 4);
        matrix[0, 0] = 1; matrix[0, 1] = 2;
        matrix[1, 0] = 3; matrix[1, 1] = 4; matrix[1, 2] = 5;
        matrix[2, 1] = 6; matrix[2, 2] = 7; matrix[2, 3] = 8;
        matrix[3, 2] = 9; matrix[3, 3] = 10;

        Assert.True(matrix.IsBandMatrix(1, 1));
    }

    [Fact]
    public void IsBandMatrix_DiagonalMatrix_ReturnsTrue()
    {
        var matrix = Matrix<double>.CreateDiagonal(new Vector<double>([1, 2, 3]));

        Assert.True(matrix.IsBandMatrix(0, 0));
    }

    #endregion

    #region IsHermitianMatrix Tests

    [Fact]
    public void IsHermitianMatrix_HermitianMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<Complex<double>>(2, 2);
        matrix[0, 0] = new Complex<double>(1, 0);  // Real diagonal
        matrix[0, 1] = new Complex<double>(2, 3);
        matrix[1, 0] = new Complex<double>(2, -3); // Conjugate of [0,1]
        matrix[1, 1] = new Complex<double>(4, 0);  // Real diagonal

        Assert.True(matrix.IsHermitianMatrix());
    }

    [Fact]
    public void IsHermitianMatrix_NonHermitianMatrix_ReturnsFalse()
    {
        var matrix = new Matrix<Complex<double>>(2, 2);
        matrix[0, 0] = new Complex<double>(1, 1);  // Complex diagonal - not Hermitian
        matrix[0, 1] = new Complex<double>(2, 3);
        matrix[1, 0] = new Complex<double>(2, 3);  // Same as [0,1], not conjugate
        matrix[1, 1] = new Complex<double>(4, 0);

        Assert.False(matrix.IsHermitianMatrix());
    }

    #endregion

    #region IsSkewHermitianMatrix Tests

    [Fact]
    public void IsSkewHermitianMatrix_SkewHermitianMatrix_ReturnsTrue()
    {
        var matrix = new Matrix<Complex<double>>(2, 2);
        matrix[0, 0] = new Complex<double>(0, 1);   // Pure imaginary diagonal
        matrix[0, 1] = new Complex<double>(2, 3);
        matrix[1, 0] = new Complex<double>(-2, 3);  // Negative conjugate of [0,1]
        matrix[1, 1] = new Complex<double>(0, -2);  // Pure imaginary diagonal

        Assert.True(matrix.IsSkewHermitianMatrix());
    }

    #endregion

    #region IsSingularMatrix Tests

    [Fact]
    public void IsSingularMatrix_ZeroDeterminant_ReturnsTrue()
    {
        // Matrix with linearly dependent rows
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 2, 4, 6 },  // Row 2 = 2 * Row 1
            { 1, 1, 1 }
        });

        Assert.True(matrix.IsSingularMatrix());
    }

    [Fact]
    public void IsSingularMatrix_IdentityMatrix_ReturnsFalse()
    {
        var matrix = Matrix<double>.CreateIdentity(3);

        Assert.False(matrix.IsSingularMatrix());
    }

    #endregion

    #region IsNonSingularMatrix Tests

    [Fact]
    public void IsNonSingularMatrix_IdentityMatrix_ReturnsTrue()
    {
        var matrix = Matrix<double>.CreateIdentity(3);

        Assert.True(matrix.IsNonSingularMatrix());
    }

    [Fact]
    public void IsNonSingularMatrix_SingularMatrix_ReturnsFalse()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 2, 4 }  // Row 2 = 2 * Row 1
        });

        Assert.False(matrix.IsNonSingularMatrix());
    }

    #endregion

    #region IsPositiveDefiniteMatrix Tests

    [Fact]
    public void IsPositiveDefiniteMatrix_IdentityMatrix_ReturnsTrue()
    {
        var matrix = Matrix<double>.CreateIdentity(3);

        Assert.True(matrix.IsPositiveDefiniteMatrix());
    }

    [Fact]
    public void IsPositiveDefiniteMatrix_SymmetricPD_ReturnsTrue()
    {
        // A positive definite matrix: A^T * A for full-rank A
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 2 },
            { 2, 3 }
        });

        Assert.True(matrix.IsPositiveDefiniteMatrix());
    }

    #endregion

    #region IsPositiveSemiDefiniteMatrix Tests

    [Fact]
    public void IsPositiveSemiDefiniteMatrix_ZeroMatrix_ReturnsTrue()
    {
        var matrix = Matrix<double>.CreateZeros(3, 3);

        Assert.True(matrix.IsPositiveSemiDefiniteMatrix());
    }

    [Fact]
    public void IsPositiveSemiDefiniteMatrix_PositiveDefinite_ReturnsTrue()
    {
        var matrix = Matrix<double>.CreateIdentity(3);

        Assert.True(matrix.IsPositiveSemiDefiniteMatrix());
    }

    #endregion

    #region IsIdempotentMatrix Tests

    [Fact]
    public void IsIdempotentMatrix_IdentityMatrix_ReturnsTrue()
    {
        // I * I = I
        var matrix = Matrix<double>.CreateIdentity(3);

        Assert.True(matrix.IsIdempotentMatrix());
    }

    [Fact]
    public void IsIdempotentMatrix_ZeroMatrix_ReturnsTrue()
    {
        // 0 * 0 = 0
        var matrix = Matrix<double>.CreateZeros(3, 3);

        Assert.True(matrix.IsIdempotentMatrix());
    }

    [Fact]
    public void IsIdempotentMatrix_ProjectionMatrix_ReturnsTrue()
    {
        // A simple 2D projection matrix onto x-axis
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 0 },
            { 0, 0 }
        });

        Assert.True(matrix.IsIdempotentMatrix());
    }

    #endregion

    #region IsInvolutoryMatrix Tests

    [Fact]
    public void IsInvolutoryMatrix_IdentityMatrix_ReturnsTrue()
    {
        // I * I = I
        var matrix = Matrix<double>.CreateIdentity(3);

        Assert.True(matrix.IsInvolutoryMatrix());
    }

    [Fact]
    public void IsInvolutoryMatrix_ReflectionMatrix_ReturnsTrue()
    {
        // Reflection matrix: A^2 = I
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 0 },
            { 0, -1 }  // Reflection across x-axis
        });

        Assert.True(matrix.IsInvolutoryMatrix());
    }

    #endregion

    #region IsOrthogonalProjectionMatrix Tests

    [Fact]
    public void IsOrthogonalProjectionMatrix_SimpleProjection_ReturnsTrue()
    {
        // Projection onto x-axis (symmetric and idempotent)
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 0 },
            { 0, 0 }
        });

        Assert.True(matrix.IsOrthogonalProjectionMatrix());
    }

    #endregion

    #region IsInvertible Tests

    [Fact]
    public void IsInvertible_IdentityMatrix_ReturnsTrue()
    {
        var matrix = Matrix<double>.CreateIdentity(3);

        Assert.True(matrix.IsInvertible());
    }

    [Fact]
    public void IsInvertible_SingularMatrix_ReturnsFalse()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 2, 4 }
        });

        Assert.False(matrix.IsInvertible());
    }

    #endregion

    #region IsVandermondeMatrix Tests

    [Fact]
    public void IsVandermondeMatrix_VandermondeMatrix_ReturnsTrue()
    {
        // Vandermonde matrix for nodes [1, 2, 3]
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 1, 1 },
            { 1, 2, 4 },
            { 1, 3, 9 }
        });

        Assert.True(matrix.IsVandermondeMatrix());
    }

    #endregion

    #region IsHilbertMatrix Tests

    [Fact]
    public void IsHilbertMatrix_HilbertMatrix_ReturnsTrue()
    {
        // H[i,j] = 1 / (i + j + 1) for 0-indexed
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0/2.0, 1.0/3.0 },
            { 1.0/2.0, 1.0/3.0, 1.0/4.0 },
            { 1.0/3.0, 1.0/4.0, 1.0/5.0 }
        });

        Assert.True(matrix.IsHilbertMatrix());
    }

    #endregion

    #region IsCauchyMatrix Tests

    [Fact]
    public void IsCauchyMatrix_CauchyMatrix_ReturnsTrue()
    {
        // C[i,j] = 1 / (x[i] - y[j])
        // For x = [1, 2, 3] and y = [4, 5, 6]
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0/(1-4), 1.0/(1-5), 1.0/(1-6) },
            { 1.0/(2-4), 1.0/(2-5), 1.0/(2-6) },
            { 1.0/(3-4), 1.0/(3-5), 1.0/(3-6) }
        });

        Assert.True(matrix.IsCauchyMatrix());
    }

    #endregion

    #region IsCompanionMatrix Tests

    [Fact]
    public void IsCompanionMatrix_CompanionMatrix_ReturnsTrue()
    {
        // Companion matrix for polynomial coefficients
        var matrix = new Matrix<double>(new double[,]
        {
            { 0, 0, -1 },
            { 1, 0, -2 },
            { 0, 1, -3 }
        });

        Assert.True(matrix.IsCompanionMatrix());
    }

    #endregion

    #region IsAdjacencyMatrix Tests

    [Fact]
    public void IsAdjacencyMatrix_SymmetricBinaryMatrix_ReturnsTrue()
    {
        // Adjacency matrix: symmetric, binary (0 or 1), zero diagonal
        var matrix = new Matrix<double>(new double[,]
        {
            { 0, 1, 1 },
            { 1, 0, 0 },
            { 1, 0, 0 }
        });

        Assert.True(matrix.IsAdjacencyMatrix());
    }

    [Fact]
    public void IsAdjacencyMatrix_NonSymmetric_ReturnsFalse()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 0, 1, 0 },
            { 0, 0, 1 },
            { 0, 0, 0 }
        });

        Assert.False(matrix.IsAdjacencyMatrix());
    }

    #endregion

    #region IsLaplacianMatrix Tests

    [Fact]
    public void IsLaplacianMatrix_SimpleLaplacian_ReturnsTrue()
    {
        // Laplacian: L = D - A, where D is degree matrix, A is adjacency
        // For a simple graph with 3 nodes
        var matrix = new Matrix<double>(new double[,]
        {
            { 2, -1, -1 },
            { -1, 1, 0 },
            { -1, 0, 1 }
        });

        Assert.True(matrix.IsLaplacianMatrix());
    }

    #endregion

    #region IsIncidenceMatrix Tests

    [Fact]
    public void IsIncidenceMatrix_IncidenceMatrix_ReturnsTrue()
    {
        // Incidence matrix: each column has exactly one +1 and one -1
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 1, 0 },
            { -1, 0, 1 },
            { 0, -1, -1 }
        });

        Assert.True(matrix.IsIncidenceMatrix());
    }

    #endregion

    #region InvertLowerTriangularMatrix Tests

    [Fact]
    public void InvertLowerTriangularMatrix_ReturnsCorrectInverse()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 2, 0, 0 },
            { 1, 3, 0 },
            { 2, 1, 4 }
        });

        var inverse = matrix.InvertLowerTriangularMatrix();

        // Verify L * L^-1 = I
        var product = matrix.Multiply(inverse);
        Assert.Equal(1, product[0, 0], Tolerance);
        Assert.Equal(1, product[1, 1], Tolerance);
        Assert.Equal(1, product[2, 2], Tolerance);
        Assert.Equal(0, product[0, 1], Tolerance);
    }

    #endregion

    #region Inverse Variants Tests

    [Fact]
    public void InverseGaussianJordanElimination_ReturnsCorrectInverse()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 7 },
            { 2, 6 }
        });

        var inverse = matrix.InverseGaussianJordanElimination();

        // Verify A * A^-1 = I
        var product = matrix.Multiply(inverse);
        Assert.Equal(1, product[0, 0], 1e-6);
        Assert.Equal(1, product[1, 1], 1e-6);
    }

    [Fact]
    public void InverseNewton_ReturnsCorrectInverse()
    {
        // Use a well-conditioned diagonally dominant matrix for Newton-Schulz
        // Newton-Schulz converges slowly for ill-conditioned matrices
        // This matrix has eigenvalues 4 and 2, giving condition number 2
        var matrix = new Matrix<double>(new double[,]
        {
            { 3, 1 },
            { 1, 3 }
        });

        var inverse = matrix.InverseNewton(maxIterations: 100);

        // Verify A * A^-1 ≈ I
        var product = matrix.Multiply(inverse);
        Assert.Equal(1, product[0, 0], 1e-4);
        Assert.Equal(1, product[1, 1], 1e-4);
        Assert.Equal(0, product[0, 1], 1e-4);
        Assert.Equal(0, product[1, 0], 1e-4);
    }

    [Fact]
    public void InverseStrassen_ReturnsCorrectInverse()
    {
        // Strassen requires power of 2 dimension
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 7 },
            { 2, 6 }
        });

        var inverse = matrix.InverseStrassen();

        // Verify A * A^-1 ≈ I
        var product = matrix.Multiply(inverse);
        Assert.Equal(1, product[0, 0], 1e-6);
        Assert.Equal(1, product[1, 1], 1e-6);
    }

    #endregion

    #region GetNullity Tests

    [Fact]
    public void GetNullity_FullRankMatrix_ReturnsZero()
    {
        var matrix = Matrix<double>.CreateIdentity(3);

        Assert.Equal(0, matrix.GetNullity());
    }

    [Fact]
    public void GetNullity_RankDeficientMatrix_ReturnsCorrectNullity()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 2, 4, 6 },  // Row 2 = 2 * Row 1
            { 3, 6, 9 }   // Row 3 = 3 * Row 1
        });

        // Nullity = n - rank = 3 - 1 = 2
        Assert.Equal(2, matrix.GetNullity());
    }

    #endregion

    #region LogDeterminant Tests

    [Fact]
    public void LogDeterminant_IdentityMatrix_ReturnsZero()
    {
        var matrix = Matrix<double>.CreateIdentity(3);

        var logDet = matrix.LogDeterminant();

        Assert.Equal(0, logDet, 1e-6); // log(1) = 0
    }

    [Fact]
    public void LogDeterminant_PositiveDefiniteMatrix_ReturnsCorrectValue()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 2 },
            { 2, 3 }
        });

        var logDet = matrix.LogDeterminant();
        var expected = Math.Log(4 * 3 - 2 * 2); // log(8) ≈ 2.079

        Assert.Equal(expected, logDet, 1e-6);
    }

    #endregion

    #region GetColumnVectors Tests

    [Fact]
    public void GetColumnVectors_ReturnsCorrectColumns()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var columns = matrix.GetColumnVectors(new[] { 0, 2 });

        Assert.Equal(2, columns.Length);
        Assert.Equal(1, columns[0][0]);
        Assert.Equal(4, columns[0][1]);
        Assert.Equal(3, columns[1][0]);
        Assert.Equal(6, columns[1][1]);
    }

    #endregion

    #region Max with Selector Tests

    [Fact]
    public void Max_WithSelector_ReturnsMaxOfTransformedValues()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { -5, 2 },
            { 3, -4 }
        });

        var max = matrix.Max(x => Math.Abs(x));

        Assert.Equal(5, max);
    }

    #endregion

    #region ToComplexVector Tests

    [Fact]
    public void ToComplexVector_ConvertsRealToComplex()
    {
        var vector = new Vector<double>([1, 2, 3]);

        var complexVector = vector.ToComplexVector();

        Assert.Equal(3, complexVector.Length);
        Assert.Equal(1, complexVector[0].Real);
        Assert.Equal(0, complexVector[0].Imaginary);
    }

    #endregion

    #region CreateComplexMatrix Tests

    [Fact]
    public void CreateComplexMatrix_CreatesCorrectDimensions()
    {
        var template = new Matrix<Complex<double>>(2, 3);

        var result = template.CreateComplexMatrix(4, 5);

        Assert.Equal(4, result.Rows);
        Assert.Equal(5, result.Columns);
    }

    #endregion

    #region IsBlockMatrix Tests

    [Fact]
    public void IsBlockMatrix_UniformBlocks_ReturnsTrue()
    {
        // 4x4 matrix with 2x2 blocks
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 1, 2, 2 },
            { 1, 1, 2, 2 },
            { 3, 3, 4, 4 },
            { 3, 3, 4, 4 }
        });

        Assert.True(matrix.IsBlockMatrix(2, 2));
    }

    #endregion

    #region InvertUnitaryMatrix Tests

    [Fact]
    public void InvertUnitaryMatrix_ReturnsConjugateTranspose()
    {
        // A simple unitary matrix: diagonal with unit complex numbers
        var matrix = new Matrix<Complex<double>>(2, 2);
        matrix[0, 0] = new Complex<double>(1, 0);
        matrix[0, 1] = new Complex<double>(0, 0);
        matrix[1, 0] = new Complex<double>(0, 0);
        matrix[1, 1] = new Complex<double>(0, 1); // i

        var inverse = matrix.InvertUnitaryMatrix();

        // Inverse of unitary is conjugate transpose
        Assert.Equal(1, inverse[0, 0].Real);
        Assert.Equal(0, inverse[0, 0].Imaginary);
        Assert.Equal(0, inverse[1, 1].Real);
        Assert.Equal(-1, inverse[1, 1].Imaginary); // conjugate of i
    }

    #endregion

    #region GetDeterminant Tests

    [Fact]
    public void GetDeterminant_SameAsDeterminant()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });

        var det1 = matrix.Determinant();
        var det2 = matrix.GetDeterminant();

        Assert.Equal(det1, det2, Tolerance);
    }

    #endregion
}
