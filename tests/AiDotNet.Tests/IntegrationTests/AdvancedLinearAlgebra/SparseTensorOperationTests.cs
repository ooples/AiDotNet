using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Additional integration tests for SparseTensor operations including
/// Transpose, Coalesce, ToDense, and format conversions.
/// </summary>
public class SparseTensorOperationTests
{
    private const double Tolerance = 1e-14;

    #region Transpose Tests

    [Fact]
    public void SparseTensor_Transpose_SwapsRowsAndColumns()
    {
        // Arrange - 3x4 matrix
        int[] rowIndices = { 0, 0, 1, 1, 2, 2 };
        int[] colIndices = { 0, 2, 1, 3, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var sparse = new SparseTensor<double>(3, 4, rowIndices, colIndices, values);

        // Act
        var transposed = sparse.Transpose();

        // Assert - Should be 4x3
        Assert.Equal(4, transposed.Rows);
        Assert.Equal(3, transposed.Columns);
        Assert.Equal(6, transposed.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_Transpose_VerifyValues()
    {
        // Arrange - Simple 2x3 matrix
        // [1 2 3]
        // [4 0 5]
        int[] rowIndices = { 0, 0, 0, 1, 1 };
        int[] colIndices = { 0, 1, 2, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var sparse = new SparseTensor<double>(2, 3, rowIndices, colIndices, values);

        // Act
        var transposed = sparse.Transpose();
        var denseTrans = transposed.ToDense();

        // Assert - Transposed matrix should be 3x2:
        // [1 4]
        // [2 0]
        // [3 5]
        Assert.True(Math.Abs(denseTrans[0, 0] - 1.0) < Tolerance);
        Assert.True(Math.Abs(denseTrans[0, 1] - 4.0) < Tolerance);
        Assert.True(Math.Abs(denseTrans[1, 0] - 2.0) < Tolerance);
        Assert.True(Math.Abs(denseTrans[1, 1]) < Tolerance); // 0
        Assert.True(Math.Abs(denseTrans[2, 0] - 3.0) < Tolerance);
        Assert.True(Math.Abs(denseTrans[2, 1] - 5.0) < Tolerance);
    }

    [Fact]
    public void SparseTensor_Transpose_Csr_PreservesValues()
    {
        // Arrange
        int[] rowPointers = { 0, 2, 3, 5 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var csr = SparseTensor<double>.FromCsr(3, 3, rowPointers, colIndices, values);

        // Act
        var transposed = csr.Transpose();

        // Assert
        Assert.Equal(3, transposed.Rows);
        Assert.Equal(3, transposed.Columns);
        Assert.Equal(5, transposed.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_Transpose_Csc_PreservesValues()
    {
        // Arrange
        int[] colPointers = { 0, 2, 3, 5 };
        int[] rowIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 4.0, 3.0, 2.0, 5.0 };
        var csc = SparseTensor<double>.FromCsc(3, 3, colPointers, rowIndices, values);

        // Act
        var transposed = csc.Transpose();

        // Assert
        Assert.Equal(3, transposed.Rows);
        Assert.Equal(3, transposed.Columns);
        Assert.Equal(5, transposed.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_DoubleTranspose_RestoresOriginal()
    {
        // Arrange
        int[] rowIndices = { 0, 1, 2, 0, 2 };
        int[] colIndices = { 0, 1, 2, 2, 0 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var original = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);
        var originalDense = original.ToDense();

        // Act
        var doubleTransposed = original.Transpose().Transpose();
        var doubleDense = doubleTransposed.ToDense();

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(originalDense[i, j] - doubleDense[i, j]) < Tolerance,
                    $"Double transpose mismatch at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void SparseTensor_Transpose_SquareMatrix_VerifyEntries()
    {
        // Arrange - 3x3 non-symmetric matrix
        // [1 2 0]
        // [0 3 4]
        // [5 0 6]
        int[] rowIndices = { 0, 0, 1, 1, 2, 2 };
        int[] colIndices = { 0, 1, 1, 2, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var transposed = sparse.Transpose();
        var originalDense = sparse.ToDense();
        var transposedDense = transposed.ToDense();

        // Assert - transposed[j,i] should equal original[i,j]
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(originalDense[i, j] - transposedDense[j, i]) < Tolerance,
                    $"Transpose failed: original[{i},{j}]={originalDense[i, j]}, transposed[{j},{i}]={transposedDense[j, i]}");
            }
        }
    }

    #endregion

    #region Coalesce Tests

    [Fact]
    public void SparseTensor_Coalesce_MergesDuplicateEntries()
    {
        // Arrange - Matrix with duplicate entries at (0,0)
        int[] rowIndices = { 0, 0, 1 };
        int[] colIndices = { 0, 0, 1 };
        double[] values = { 1.0, 2.0, 3.0 }; // (0,0) has 1+2=3

        var sparse = new SparseTensor<double>(2, 2, rowIndices, colIndices, values);

        // Act
        var coalesced = sparse.Coalesce();

        // Assert - Should have 2 entries after merging duplicates
        Assert.Equal(2, coalesced.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_Coalesce_SumsDuplicateValues()
    {
        // Arrange - Multiple entries at same position
        int[] rowIndices = { 0, 0, 0, 1, 1 };
        int[] colIndices = { 0, 0, 0, 1, 1 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 }; // (0,0)=1+2+3=6, (1,1)=4+5=9

        var sparse = new SparseTensor<double>(2, 2, rowIndices, colIndices, values);

        // Act
        var coalesced = sparse.Coalesce();
        var dense = coalesced.ToDense();

        // Assert
        Assert.Equal(2, coalesced.NonZeroCount);
        Assert.True(Math.Abs(dense[0, 0] - 6.0) < Tolerance, $"Expected 6.0, got {dense[0, 0]}");
        Assert.True(Math.Abs(dense[1, 1] - 9.0) < Tolerance, $"Expected 9.0, got {dense[1, 1]}");
    }

    [Fact]
    public void SparseTensor_Coalesce_RemovesZeros()
    {
        // Arrange - Entries that sum to zero
        int[] rowIndices = { 0, 0, 1 };
        int[] colIndices = { 0, 0, 1 };
        double[] values = { 5.0, -5.0, 3.0 }; // (0,0) sums to 0

        var sparse = new SparseTensor<double>(2, 2, rowIndices, colIndices, values);

        // Act
        var coalesced = sparse.Coalesce();

        // Assert - (0,0) should be removed since it's zero
        Assert.Equal(1, coalesced.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_Coalesce_NoDuplicates_NoChange()
    {
        // Arrange - No duplicate entries
        int[] rowIndices = { 0, 1, 2 };
        int[] colIndices = { 0, 1, 2 };
        double[] values = { 1.0, 2.0, 3.0 };

        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var coalesced = sparse.Coalesce();

        // Assert
        Assert.Equal(3, coalesced.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_Coalesce_Empty_ReturnsEmpty()
    {
        // Arrange
        var sparse = new SparseTensor<double>(3, 3, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>());

        // Act
        var coalesced = sparse.Coalesce();

        // Assert
        Assert.Equal(0, coalesced.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_Coalesce_SortsEntries()
    {
        // Arrange - Entries not in sorted order
        int[] rowIndices = { 2, 0, 1, 0 };
        int[] colIndices = { 2, 0, 1, 2 };
        double[] values = { 3.0, 1.0, 2.0, 4.0 };

        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var coalesced = sparse.Coalesce();

        // Assert - Should be in row-major order after coalesce
        Assert.Equal(4, coalesced.NonZeroCount);
        // First entry should be at lowest row
        Assert.True(coalesced.RowIndices[0] <= coalesced.RowIndices[1]);
    }

    [Fact]
    public void SparseTensor_Coalesce_ManyDuplicates_CorrectSum()
    {
        // Arrange - Many duplicates at same position
        int[] rowIndices = { 1, 1, 1, 1, 1 };
        int[] colIndices = { 1, 1, 1, 1, 1 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 }; // Sum = 15

        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var coalesced = sparse.Coalesce();
        var dense = coalesced.ToDense();

        // Assert
        Assert.Equal(1, coalesced.NonZeroCount);
        Assert.True(Math.Abs(dense[1, 1] - 15.0) < Tolerance);
    }

    #endregion

    #region ToDense Tests

    [Fact]
    public void SparseTensor_ToDense_CorrectValues()
    {
        // Arrange
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        int[] rowIndices = { 0, 0, 1, 2, 2 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var dense = sparse.ToDense();

        // Assert
        Assert.Equal(3, dense.Shape[0]);
        Assert.Equal(3, dense.Shape[1]);

        // Check all values
        Assert.True(Math.Abs(dense[0, 0] - 1.0) < Tolerance);
        Assert.True(Math.Abs(dense[0, 1]) < Tolerance); // 0
        Assert.True(Math.Abs(dense[0, 2] - 2.0) < Tolerance);
        Assert.True(Math.Abs(dense[1, 0]) < Tolerance); // 0
        Assert.True(Math.Abs(dense[1, 1] - 3.0) < Tolerance);
        Assert.True(Math.Abs(dense[1, 2]) < Tolerance); // 0
        Assert.True(Math.Abs(dense[2, 0] - 4.0) < Tolerance);
        Assert.True(Math.Abs(dense[2, 1]) < Tolerance); // 0
        Assert.True(Math.Abs(dense[2, 2] - 5.0) < Tolerance);
    }

    [Fact]
    public void SparseTensor_ToDense_EmptyMatrix_AllZeros()
    {
        // Arrange
        var sparse = new SparseTensor<double>(3, 3, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>());

        // Act
        var dense = sparse.ToDense();

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(dense[i, j]) < Tolerance, $"Expected 0 at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void SparseTensor_ToDense_Csr_CorrectValues()
    {
        // Arrange
        int[] rowPointers = { 0, 2, 3, 5 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var csr = SparseTensor<double>.FromCsr(3, 3, rowPointers, colIndices, values);

        // Act
        var dense = csr.ToDense();

        // Assert
        Assert.True(Math.Abs(dense[0, 0] - 1.0) < Tolerance);
        Assert.True(Math.Abs(dense[0, 2] - 2.0) < Tolerance);
        Assert.True(Math.Abs(dense[1, 1] - 3.0) < Tolerance);
        Assert.True(Math.Abs(dense[2, 0] - 4.0) < Tolerance);
        Assert.True(Math.Abs(dense[2, 2] - 5.0) < Tolerance);
    }

    [Fact]
    public void SparseTensor_ToDense_Csc_CorrectValues()
    {
        // Arrange - CSC of same matrix
        int[] colPointers = { 0, 2, 3, 5 };
        int[] rowIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 4.0, 3.0, 2.0, 5.0 };
        var csc = SparseTensor<double>.FromCsc(3, 3, colPointers, rowIndices, values);

        // Act
        var dense = csc.ToDense();

        // Assert
        Assert.True(Math.Abs(dense[0, 0] - 1.0) < Tolerance);
        Assert.True(Math.Abs(dense[0, 2] - 2.0) < Tolerance);
        Assert.True(Math.Abs(dense[1, 1] - 3.0) < Tolerance);
        Assert.True(Math.Abs(dense[2, 0] - 4.0) < Tolerance);
        Assert.True(Math.Abs(dense[2, 2] - 5.0) < Tolerance);
    }

    [Fact]
    public void SparseTensor_ToDense_RectangularMatrix()
    {
        // Arrange - 2x4 matrix
        int[] rowIndices = { 0, 0, 1, 1 };
        int[] colIndices = { 0, 3, 1, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0 };
        var sparse = new SparseTensor<double>(2, 4, rowIndices, colIndices, values);

        // Act
        var dense = sparse.ToDense();

        // Assert
        Assert.Equal(2, dense.Shape[0]);
        Assert.Equal(4, dense.Shape[1]);
        Assert.True(Math.Abs(dense[0, 0] - 1.0) < Tolerance);
        Assert.True(Math.Abs(dense[0, 3] - 2.0) < Tolerance);
        Assert.True(Math.Abs(dense[1, 1] - 3.0) < Tolerance);
        Assert.True(Math.Abs(dense[1, 2] - 4.0) < Tolerance);
    }

    #endregion

    #region FromDense Advanced Tests

    [Fact]
    public void SparseTensor_FromDense_WithTolerance_FiltersSmallValues()
    {
        // Arrange - Matrix with small values that should be filtered
        var dense = new Tensor<double>(new[] { 3, 3 });
        dense[0, 0] = 1.0;
        dense[0, 1] = 1e-10; // Should be filtered with tolerance 1e-8
        dense[1, 1] = 2.0;
        dense[2, 2] = 3.0;

        // Act
        var sparse = SparseTensor<double>.FromDense(dense, 1e-8);

        // Assert
        Assert.Equal(3, sparse.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_FromDense_ToDense_RoundTrip()
    {
        // Arrange
        var dense = new Tensor<double>(new[] { 4, 4 });
        dense[0, 0] = 1.0; dense[0, 3] = 2.0;
        dense[1, 1] = 3.0; dense[1, 2] = 4.0;
        dense[2, 0] = 5.0; dense[2, 3] = 6.0;
        dense[3, 1] = 7.0; dense[3, 2] = 8.0;

        // Act
        var sparse = SparseTensor<double>.FromDense(dense);
        var roundTrip = sparse.ToDense();

        // Assert
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.True(Math.Abs(dense[i, j] - roundTrip[i, j]) < Tolerance,
                    $"Mismatch at [{i},{j}]: expected {dense[i, j]}, got {roundTrip[i, j]}");
            }
        }
    }

    [Fact]
    public void SparseTensor_FromDense_AllZeros_EmptySparse()
    {
        // Arrange
        var dense = new Tensor<double>(new[] { 5, 5 }); // All zeros

        // Act
        var sparse = SparseTensor<double>.FromDense(dense);

        // Assert
        Assert.Equal(0, sparse.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_FromDense_FullMatrix_AllNonZeros()
    {
        // Arrange
        var dense = new Tensor<double>(new[] { 2, 2 });
        dense[0, 0] = 1.0; dense[0, 1] = 2.0;
        dense[1, 0] = 3.0; dense[1, 1] = 4.0;

        // Act
        var sparse = SparseTensor<double>.FromDense(dense);

        // Assert
        Assert.Equal(4, sparse.NonZeroCount);
    }

    #endregion

    #region Large Sparse Matrix Tests

    [Fact]
    public void SparseTensor_LargeSparseMatrix_HandlesCorrectly()
    {
        // Arrange - 100x100 matrix with 10% density
        int size = 100;
        int nnz = size * size / 10;
        var random = new Random(42);

        var rowIndices = new int[nnz];
        var colIndices = new int[nnz];
        var values = new double[nnz];

        for (int i = 0; i < nnz; i++)
        {
            rowIndices[i] = random.Next(size);
            colIndices[i] = random.Next(size);
            values[i] = random.NextDouble() * 10;
        }

        // Act
        var sparse = new SparseTensor<double>(size, size, rowIndices, colIndices, values);
        var csr = sparse.ToCsr();
        var csc = sparse.ToCsc();

        // Assert
        Assert.Equal(size, sparse.Rows);
        Assert.Equal(size, sparse.Columns);
        Assert.Equal(SparseStorageFormat.Csr, csr.Format);
        Assert.Equal(SparseStorageFormat.Csc, csc.Format);
    }

    [Fact]
    public void SparseTensor_VerySparseLargeMatrix_HandlesCorrectly()
    {
        // Arrange - 1000x1000 matrix with only 50 non-zeros (0.005% density)
        int size = 1000;
        int nnz = 50;

        var rowIndices = new int[nnz];
        var colIndices = new int[nnz];
        var values = new double[nnz];

        for (int i = 0; i < nnz; i++)
        {
            rowIndices[i] = i * (size / nnz);
            colIndices[i] = i * (size / nnz);
            values[i] = i + 1.0;
        }

        // Act
        var sparse = new SparseTensor<double>(size, size, rowIndices, colIndices, values);
        var csr = sparse.ToCsr();

        // Assert
        Assert.Equal(size, sparse.Rows);
        Assert.Equal(nnz, sparse.NonZeroCount);
        Assert.Equal(nnz, csr.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_DiagonalLargeMatrix_HandlesCorrectly()
    {
        // Arrange - 500x500 diagonal matrix
        int size = 500;
        var rowIndices = new int[size];
        var colIndices = new int[size];
        var values = new double[size];

        for (int i = 0; i < size; i++)
        {
            rowIndices[i] = i;
            colIndices[i] = i;
            values[i] = i + 1.0;
        }

        // Act
        var sparse = new SparseTensor<double>(size, size, rowIndices, colIndices, values);
        var transposed = sparse.Transpose();

        // Assert - Transpose of diagonal is same matrix
        Assert.Equal(size, sparse.NonZeroCount);
        Assert.Equal(size, transposed.NonZeroCount);
    }

    #endregion

    #region Format Conversion Value Verification Tests

    [Fact]
    public void SparseTensor_AllFormats_ProduceSameDense()
    {
        // Arrange
        int[] rowIndices = { 0, 0, 1, 2, 2 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var coo = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var cooDense = coo.ToDense();
        var csrDense = coo.ToCsr().ToDense();
        var cscDense = coo.ToCsc().ToDense();

        // Assert - All formats should produce identical dense matrices
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(cooDense[i, j] - csrDense[i, j]) < Tolerance,
                    $"COO vs CSR mismatch at [{i},{j}]");
                Assert.True(Math.Abs(cooDense[i, j] - cscDense[i, j]) < Tolerance,
                    $"COO vs CSC mismatch at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void SparseTensor_CsrToCsc_PreservesValues()
    {
        // Arrange
        int[] rowPointers = { 0, 2, 3, 5 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var csr = SparseTensor<double>.FromCsr(3, 3, rowPointers, colIndices, values);

        // Act
        var csc = csr.ToCsc();
        var csrDense = csr.ToDense();
        var cscDense = csc.ToDense();

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(csrDense[i, j] - cscDense[i, j]) < Tolerance,
                    $"CSR vs CSC mismatch at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void SparseTensor_CscToCsr_PreservesValues()
    {
        // Arrange
        int[] colPointers = { 0, 2, 3, 5 };
        int[] rowIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 4.0, 3.0, 2.0, 5.0 };
        var csc = SparseTensor<double>.FromCsc(3, 3, colPointers, rowIndices, values);

        // Act
        var csr = csc.ToCsr();
        var cscDense = csc.ToDense();
        var csrDense = csr.ToDense();

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(cscDense[i, j] - csrDense[i, j]) < Tolerance,
                    $"CSC vs CSR mismatch at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Symmetry and Pattern Tests

    [Fact]
    public void SparseTensor_SymmetricMatrix_TransposeEqualsSelf()
    {
        // Arrange - Symmetric matrix
        // [1 2 3]
        // [2 4 5]
        // [3 5 6]
        int[] rowIndices = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
        int[] colIndices = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
        double[] values = { 1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0 };
        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var transposed = sparse.Transpose();
        var originalDense = sparse.ToDense();
        var transposedDense = transposed.ToDense();

        // Assert - Should be equal to transpose
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(originalDense[i, j] - transposedDense[i, j]) < Tolerance,
                    $"Symmetric matrix should equal its transpose at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void SparseTensor_TridiagonalMatrix_CorrectSparsity()
    {
        // Arrange - Tridiagonal 5x5 matrix
        var rowIndices = new List<int>();
        var colIndices = new List<int>();
        var values = new List<double>();

        for (int i = 0; i < 5; i++)
        {
            // Subdiagonal
            if (i > 0)
            {
                rowIndices.Add(i);
                colIndices.Add(i - 1);
                values.Add(1.0);
            }
            // Diagonal
            rowIndices.Add(i);
            colIndices.Add(i);
            values.Add(2.0);
            // Superdiagonal
            if (i < 4)
            {
                rowIndices.Add(i);
                colIndices.Add(i + 1);
                values.Add(1.0);
            }
        }

        var sparse = new SparseTensor<double>(5, 5, rowIndices.ToArray(), colIndices.ToArray(), values.ToArray());

        // Act & Assert
        // 5 diagonal + 4 sub + 4 super = 13 non-zeros
        Assert.Equal(13, sparse.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_BandMatrix_CorrectStorage()
    {
        // Arrange - Banded matrix with bandwidth 2
        var rowIndices = new List<int>();
        var colIndices = new List<int>();
        var values = new List<double>();

        int size = 6;
        int bandwidth = 2;

        for (int i = 0; i < size; i++)
        {
            for (int j = Math.Max(0, i - bandwidth); j <= Math.Min(size - 1, i + bandwidth); j++)
            {
                rowIndices.Add(i);
                colIndices.Add(j);
                values.Add(i + j + 1.0);
            }
        }

        var sparse = new SparseTensor<double>(size, size, rowIndices.ToArray(), colIndices.ToArray(), values.ToArray());

        // Act
        var csr = sparse.ToCsr();

        // Assert
        Assert.Equal(SparseStorageFormat.Csr, csr.Format);
        Assert.True(csr.NonZeroCount > 0);
    }

    #endregion

    #region Stress Tests

    [Fact]
    public void SparseTensor_MultipleConversions_NoDataLoss()
    {
        // Arrange
        int[] rowIndices = { 0, 0, 1, 1, 2 };
        int[] colIndices = { 0, 2, 1, 2, 0 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var original = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);
        var originalDense = original.ToDense();

        // Act - Multiple round-trip conversions
        var result = original
            .ToCsr()
            .ToCoo()
            .ToCsc()
            .ToCoo()
            .ToCsr()
            .ToCsc()
            .ToCoo();
        var resultDense = result.ToDense();

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(originalDense[i, j] - resultDense[i, j]) < Tolerance,
                    $"Data loss after multiple conversions at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void SparseTensor_TransposeAndConversions_Consistent()
    {
        // Arrange
        int[] rowIndices = { 0, 1, 2, 0, 1 };
        int[] colIndices = { 1, 0, 2, 2, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act - Transpose in different formats
        var cooTransposed = sparse.Transpose();
        var csrTransposed = sparse.ToCsr().Transpose();
        var cscTransposed = sparse.ToCsc().Transpose();

        // Assert - All transposes should produce same dense matrix
        var cooDense = cooTransposed.ToDense();
        var csrDense = csrTransposed.ToDense();
        var cscDense = cscTransposed.ToDense();

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(cooDense[i, j] - csrDense[i, j]) < Tolerance);
                Assert.True(Math.Abs(cooDense[i, j] - cscDense[i, j]) < Tolerance);
            }
        }
    }

    [Fact]
    public void SparseTensor_CoalesceAndTranspose_Consistent()
    {
        // Arrange - Matrix with duplicates
        int[] rowIndices = { 0, 0, 1, 1, 0 };
        int[] colIndices = { 0, 0, 1, 1, 1 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act - Coalesce first, then transpose
        var coalescedThenTransposed = sparse.Coalesce().Transpose();

        // Also transpose first, then coalesce
        var transposedThenCoalesced = sparse.Transpose().Coalesce();

        // Get dense representations
        var method1Dense = coalescedThenTransposed.ToDense();
        var method2Dense = transposedThenCoalesced.ToDense();

        // Assert - Both methods should give same result
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.True(Math.Abs(method1Dense[i, j] - method2Dense[i, j]) < Tolerance,
                    $"Coalesce/Transpose order mismatch at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Numeric Type Tests

    [Fact]
    public void SparseTensor_IntType_WorksCorrectly()
    {
        // Arrange
        int[] rowIndices = { 0, 1, 2 };
        int[] colIndices = { 1, 0, 2 };
        int[] values = { 10, 20, 30 };

        // Act
        var sparse = new SparseTensor<int>(3, 3, rowIndices, colIndices, values);

        // Assert
        Assert.Equal(3, sparse.NonZeroCount);
        Assert.Equal(10, sparse.Values[0]);
        Assert.Equal(20, sparse.Values[1]);
        Assert.Equal(30, sparse.Values[2]);
    }

    [Fact]
    public void SparseTensor_IntType_Transpose()
    {
        // Arrange
        int[] rowIndices = { 0, 1 };
        int[] colIndices = { 1, 0 };
        int[] values = { 10, 20 };
        var sparse = new SparseTensor<int>(2, 2, rowIndices, colIndices, values);

        // Act
        var transposed = sparse.Transpose();
        var dense = transposed.ToDense();

        // Assert
        Assert.Equal(10, dense[1, 0]);
        Assert.Equal(20, dense[0, 1]);
    }

    [Fact]
    public void SparseTensor_IntType_Coalesce()
    {
        // Arrange - Duplicates
        int[] rowIndices = { 0, 0, 1 };
        int[] colIndices = { 0, 0, 1 };
        int[] values = { 5, 7, 10 };
        var sparse = new SparseTensor<int>(2, 2, rowIndices, colIndices, values);

        // Act
        var coalesced = sparse.Coalesce();
        var dense = coalesced.ToDense();

        // Assert
        Assert.Equal(2, coalesced.NonZeroCount);
        Assert.Equal(12, dense[0, 0]); // 5 + 7
        Assert.Equal(10, dense[1, 1]);
    }

    #endregion
}
