using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for SparseTensor that verify storage format conversions
/// and sparse-dense operations work correctly.
/// </summary>
public class SparseTensorIntegrationTests
{
    private const double Tolerance = 1e-14;

    #region COO Format Tests

    [Fact]
    public void SparseTensor_Coo_Construction_StoresCorrectValues()
    {
        // Arrange - Create a simple 3x3 sparse matrix with known values
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        int[] rowIndices = { 0, 0, 1, 2, 2 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Assert
        Assert.Equal(3, sparse.Rows);
        Assert.Equal(3, sparse.Columns);
        Assert.Equal(5, sparse.NonZeroCount);
        Assert.Equal(SparseStorageFormat.Coo, sparse.Format);
    }

    [Fact]
    public void SparseTensor_Coo_FromDense_ExtractsNonZeros()
    {
        // Arrange - Create a dense tensor with some zeros
        var dense = new Tensor<double>(new[] { 3, 3 });
        dense[0, 0] = 1.0; dense[0, 2] = 2.0;
        dense[1, 1] = 3.0;
        dense[2, 0] = 4.0; dense[2, 2] = 5.0;

        // Act
        var sparse = SparseTensor<double>.FromDense(dense);

        // Assert
        Assert.Equal(5, sparse.NonZeroCount);
        Assert.Equal(SparseStorageFormat.Coo, sparse.Format);
    }

    [Fact]
    public void SparseTensor_Coo_EmptyMatrix_HasZeroNonZeros()
    {
        // Arrange & Act
        var sparse = new SparseTensor<double>(5, 5, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>());

        // Assert
        Assert.Equal(0, sparse.NonZeroCount);
        Assert.Equal(5, sparse.Rows);
        Assert.Equal(5, sparse.Columns);
    }

    #endregion

    #region CSR Format Tests

    [Fact]
    public void SparseTensor_FromCsr_ValidConstruction()
    {
        // Arrange - CSR representation of:
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        int[] rowPointers = { 0, 2, 3, 5 }; // 3 rows, so 4 pointers
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var sparse = SparseTensor<double>.FromCsr(3, 3, rowPointers, colIndices, values);

        // Assert
        Assert.Equal(3, sparse.Rows);
        Assert.Equal(3, sparse.Columns);
        Assert.Equal(5, sparse.NonZeroCount);
        Assert.Equal(SparseStorageFormat.Csr, sparse.Format);
    }

    [Fact]
    public void SparseTensor_CooToCsr_Conversion_PreservesData()
    {
        // Arrange
        int[] rowIndices = { 0, 0, 1, 2, 2 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var coo = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var csr = coo.ToCsr();

        // Assert
        Assert.Equal(SparseStorageFormat.Csr, csr.Format);
        Assert.Equal(5, csr.NonZeroCount);

        // Convert back to COO to verify data integrity
        var cooAgain = csr.ToCoo();
        Assert.Equal(5, cooAgain.NonZeroCount);

        // Verify values are preserved (order might differ due to format conversion)
        var originalValues = values.OrderBy(v => v).ToArray();
        var roundTripValues = cooAgain.Values.OrderBy(v => v).ToArray();
        for (int i = 0; i < originalValues.Length; i++)
        {
            Assert.True(Math.Abs(originalValues[i] - roundTripValues[i]) < Tolerance,
                $"Value mismatch at index {i}: expected {originalValues[i]}, got {roundTripValues[i]}");
        }
    }

    [Fact]
    public void SparseTensor_CsrToCoo_Conversion_PreservesData()
    {
        // Arrange
        int[] rowPointers = { 0, 2, 3, 5 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var csr = SparseTensor<double>.FromCsr(3, 3, rowPointers, colIndices, values);

        // Act
        var coo = csr.ToCoo();

        // Assert
        Assert.Equal(SparseStorageFormat.Coo, coo.Format);
        Assert.Equal(5, coo.NonZeroCount);

        // Verify values are preserved (order might differ due to format conversion)
        var originalValues = values.OrderBy(v => v).ToArray();
        var resultValues = coo.Values.OrderBy(v => v).ToArray();
        for (int i = 0; i < originalValues.Length; i++)
        {
            Assert.True(Math.Abs(originalValues[i] - resultValues[i]) < Tolerance,
                $"Value mismatch at index {i}: expected {originalValues[i]}, got {resultValues[i]}");
        }
    }

    #endregion

    #region CSC Format Tests

    [Fact]
    public void SparseTensor_FromCsc_ValidConstruction()
    {
        // Arrange - CSC representation of:
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        int[] colPointers = { 0, 2, 3, 5 }; // 3 columns, so 4 pointers
        int[] rowIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 4.0, 3.0, 2.0, 5.0 };

        // Act
        var sparse = SparseTensor<double>.FromCsc(3, 3, colPointers, rowIndices, values);

        // Assert
        Assert.Equal(3, sparse.Rows);
        Assert.Equal(3, sparse.Columns);
        Assert.Equal(5, sparse.NonZeroCount);
        Assert.Equal(SparseStorageFormat.Csc, sparse.Format);
    }

    [Fact]
    public void SparseTensor_CooToCsc_Conversion_PreservesData()
    {
        // Arrange
        int[] rowIndices = { 0, 0, 1, 2, 2 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var coo = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var csc = coo.ToCsc();

        // Assert
        Assert.Equal(SparseStorageFormat.Csc, csc.Format);
        Assert.Equal(5, csc.NonZeroCount);

        // Verify values are preserved (order might differ due to format conversion)
        var originalValues = values.OrderBy(v => v).ToArray();
        var resultValues = csc.Values.OrderBy(v => v).ToArray();
        for (int i = 0; i < originalValues.Length; i++)
        {
            Assert.True(Math.Abs(originalValues[i] - resultValues[i]) < Tolerance,
                $"Value mismatch at index {i}: expected {originalValues[i]}, got {resultValues[i]}");
        }
    }

    #endregion

    #region Format Round-Trip Tests

    [Fact]
    public void SparseTensor_CooToCsrToCoo_RoundTrip_PreservesValues()
    {
        // Arrange
        int[] rowIndices = { 0, 0, 1, 2, 2 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var original = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var csr = original.ToCsr();
        var roundTrip = csr.ToCoo();

        // Assert
        Assert.Equal(original.NonZeroCount, roundTrip.NonZeroCount);

        // Values should be preserved (order might differ)
        var originalValues = original.Values.OrderBy(v => v).ToArray();
        var roundTripValues = roundTrip.Values.OrderBy(v => v).ToArray();

        for (int i = 0; i < originalValues.Length; i++)
        {
            Assert.True(Math.Abs(originalValues[i] - roundTripValues[i]) < Tolerance,
                $"Value mismatch at index {i}");
        }
    }

    [Fact]
    public void SparseTensor_CooToCscToCoo_RoundTrip_PreservesValues()
    {
        // Arrange
        int[] rowIndices = { 0, 0, 1, 2, 2 };
        int[] colIndices = { 0, 2, 1, 0, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var original = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var csc = original.ToCsc();
        var roundTrip = csc.ToCoo();

        // Assert
        Assert.Equal(original.NonZeroCount, roundTrip.NonZeroCount);

        // Verify values are preserved (order might differ)
        var originalValues = original.Values.OrderBy(v => v).ToArray();
        var roundTripValues = roundTrip.Values.OrderBy(v => v).ToArray();
        for (int i = 0; i < originalValues.Length; i++)
        {
            Assert.True(Math.Abs(originalValues[i] - roundTripValues[i]) < Tolerance,
                $"Value mismatch at index {i}: expected {originalValues[i]}, got {roundTripValues[i]}");
        }
    }

    [Fact]
    public void SparseTensor_AllFormats_SameNonZeroCount()
    {
        // Arrange
        int[] rowIndices = { 0, 1, 2, 0, 2 };
        int[] colIndices = { 0, 1, 2, 2, 0 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var coo = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var csr = coo.ToCsr();
        var csc = coo.ToCsc();

        // Assert
        Assert.Equal(coo.NonZeroCount, csr.NonZeroCount);
        Assert.Equal(coo.NonZeroCount, csc.NonZeroCount);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void SparseTensor_SingleElement_AllFormats()
    {
        // Arrange - Single element matrix
        int[] rowIndices = { 1 };
        int[] colIndices = { 2 };
        double[] values = { 42.0 };
        var coo = new SparseTensor<double>(5, 5, rowIndices, colIndices, values);

        // Act
        var csr = coo.ToCsr();
        var csc = coo.ToCsc();

        // Assert
        Assert.Equal(1, coo.NonZeroCount);
        Assert.Equal(1, csr.NonZeroCount);
        Assert.Equal(1, csc.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_DiagonalMatrix_CorrectStorage()
    {
        // Arrange - 4x4 diagonal matrix
        int[] rowIndices = { 0, 1, 2, 3 };
        int[] colIndices = { 0, 1, 2, 3 };
        double[] values = { 1.0, 2.0, 3.0, 4.0 };
        var coo = new SparseTensor<double>(4, 4, rowIndices, colIndices, values);

        // Act
        var csr = coo.ToCsr();
        var csc = coo.ToCsc();

        // Assert
        Assert.Equal(4, csr.NonZeroCount);
        Assert.Equal(4, csc.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_LowerTriangular_CorrectStorage()
    {
        // Arrange - Lower triangular 3x3
        // [1 0 0]
        // [2 3 0]
        // [4 5 6]
        int[] rowIndices = { 0, 1, 1, 2, 2, 2 };
        int[] colIndices = { 0, 0, 1, 0, 1, 2 };
        double[] values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var coo = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act & Assert
        Assert.Equal(6, coo.NonZeroCount);
        var csr = coo.ToCsr();
        Assert.Equal(6, csr.NonZeroCount);
    }

    [Fact]
    public void SparseTensor_TallMatrix_ValidStorage()
    {
        // Arrange - 5x2 matrix
        int[] rowIndices = { 0, 2, 4 };
        int[] colIndices = { 0, 1, 0 };
        double[] values = { 1.0, 2.0, 3.0 };
        var coo = new SparseTensor<double>(5, 2, rowIndices, colIndices, values);

        // Act
        var csr = coo.ToCsr();
        var csc = coo.ToCsc();

        // Assert
        Assert.Equal(5, csr.Rows);
        Assert.Equal(2, csr.Columns);
        Assert.Equal(5, csc.Rows);
        Assert.Equal(2, csc.Columns);
    }

    [Fact]
    public void SparseTensor_WideMatrix_ValidStorage()
    {
        // Arrange - 2x5 matrix
        int[] rowIndices = { 0, 0, 1 };
        int[] colIndices = { 0, 4, 2 };
        double[] values = { 1.0, 2.0, 3.0 };
        var coo = new SparseTensor<double>(2, 5, rowIndices, colIndices, values);

        // Act
        var csr = coo.ToCsr();
        var csc = coo.ToCsc();

        // Assert
        Assert.Equal(2, csr.Rows);
        Assert.Equal(5, csr.Columns);
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void SparseTensor_Coo_NegativeRows_ThrowsException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SparseTensor<double>(-1, 3, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>()));
    }

    [Fact]
    public void SparseTensor_Coo_NegativeColumns_ThrowsException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SparseTensor<double>(3, -1, Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>()));
    }

    [Fact]
    public void SparseTensor_Coo_MismatchedArrayLengths_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() =>
            new SparseTensor<double>(3, 3, new[] { 0, 1 }, new[] { 0 }, new[] { 1.0, 2.0 }));
    }

    [Fact]
    public void SparseTensor_Coo_NullRowIndices_ThrowsException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SparseTensor<double>(3, 3, null!, new[] { 0 }, new[] { 1.0 }));
    }

    [Fact]
    public void SparseTensor_Coo_NullColumnIndices_ThrowsException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SparseTensor<double>(3, 3, new[] { 0 }, null!, new[] { 1.0 }));
    }

    [Fact]
    public void SparseTensor_Coo_NullValues_ThrowsException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SparseTensor<double>(3, 3, new[] { 0 }, new[] { 0 }, null!));
    }

    [Fact]
    public void SparseTensor_Csr_InvalidRowPointers_ThrowsException()
    {
        // RowPointers should have length = rows + 1
        Assert.Throws<ArgumentException>(() =>
            SparseTensor<double>.FromCsr(3, 3, new[] { 0, 1 }, new[] { 0 }, new[] { 1.0 }));
    }

    [Fact]
    public void SparseTensor_Csc_InvalidColumnPointers_ThrowsException()
    {
        // ColumnPointers should have length = columns + 1
        Assert.Throws<ArgumentException>(() =>
            SparseTensor<double>.FromCsc(3, 3, new[] { 0, 1 }, new[] { 0 }, new[] { 1.0 }));
    }

    [Fact]
    public void SparseTensor_FromDense_NonRank2_ThrowsException()
    {
        // SparseTensor only supports 2D tensors
        var tensor3d = new Tensor<double>(new[] { 2, 2, 2 });

        Assert.Throws<ArgumentException>(() =>
            SparseTensor<double>.FromDense(tensor3d));
    }

    [Fact]
    public void SparseTensor_FromDense_Null_ThrowsException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            SparseTensor<double>.FromDense(null!));
    }

    #endregion

    #region Numeric Type Tests

    [Fact]
    public void SparseTensor_FloatType_WorksCorrectly()
    {
        // Arrange
        int[] rowIndices = { 0, 1 };
        int[] colIndices = { 0, 1 };
        float[] values = { 1.5f, 2.5f };

        // Act
        var sparse = new SparseTensor<float>(2, 2, rowIndices, colIndices, values);

        // Assert
        Assert.Equal(2, sparse.NonZeroCount);
        Assert.Equal(1.5f, sparse.Values[0]);
        Assert.Equal(2.5f, sparse.Values[1]);
    }

    #endregion
}
