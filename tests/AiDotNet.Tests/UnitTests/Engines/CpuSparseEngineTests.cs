using Xunit;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Engines;

/// <summary>
/// Unit tests for CpuSparseEngine operations.
/// </summary>
public class CpuSparseEngineTests
{
    private readonly CpuSparseEngine _engine = CpuSparseEngine.Instance;

    #region SpMV Tests

    [Fact]
    public void SpMV_IdentityMatrix_ReturnsOriginalVector()
    {
        // Arrange: 3x3 identity as sparse
        var rowIndices = new[] { 0, 1, 2 };
        var colIndices = new[] { 0, 1, 2 };
        var values = new[] { 1.0, 1.0, 1.0 };
        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);
        var dense = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        // Act
        var result = _engine.SpMV(sparse, dense);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(2.0, result[0], precision: 10);
        Assert.Equal(3.0, result[1], precision: 10);
        Assert.Equal(4.0, result[2], precision: 10);
    }

    [Fact]
    public void SpMV_SingleRowSparse_ComputesDotProduct()
    {
        // Arrange: 1x3 sparse matrix with values [1, 2, 3]
        var rowIndices = new[] { 0, 0, 0 };
        var colIndices = new[] { 0, 1, 2 };
        var values = new[] { 1.0, 2.0, 3.0 };
        var sparse = new SparseTensor<double>(1, 3, rowIndices, colIndices, values);
        var dense = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = _engine.SpMV(sparse, dense);

        // Assert: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        Assert.Equal(1, result.Length);
        Assert.Equal(32.0, result[0], precision: 10);
    }

    [Fact]
    public void SpMV_EmptySparse_ReturnsZeroVector()
    {
        // Arrange: Empty 3x3 sparse matrix
        var sparse = new SparseTensor<double>(3, 3,
            Array.Empty<int>(), Array.Empty<int>(), Array.Empty<double>());
        var dense = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = _engine.SpMV(sparse, dense);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(0.0, result[0]);
        Assert.Equal(0.0, result[1]);
        Assert.Equal(0.0, result[2]);
    }

    #endregion

    #region SpMM Tests

    [Fact]
    public void SpMM_IdentitySparse_ReturnsDenseMatrix()
    {
        // Arrange: 2x2 identity as sparse
        var rowIndices = new[] { 0, 1 };
        var colIndices = new[] { 0, 1 };
        var values = new[] { 1.0, 1.0 };
        var sparse = new SparseTensor<double>(2, 2, rowIndices, colIndices, values);

        var dense = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        // Act
        var result = _engine.SpMM(sparse, dense);

        // Assert: Identity * Dense = Dense
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
        Assert.Equal(1.0, result[0, 0], precision: 10);
        Assert.Equal(2.0, result[0, 1], precision: 10);
        Assert.Equal(3.0, result[1, 0], precision: 10);
        Assert.Equal(4.0, result[1, 1], precision: 10);
    }

    [Fact]
    public void SpMM_ScalingSparse_ScalesMatrix()
    {
        // Arrange: 2x2 diagonal with 2 on diagonal
        var rowIndices = new[] { 0, 1 };
        var colIndices = new[] { 0, 1 };
        var values = new[] { 2.0, 2.0 };
        var sparse = new SparseTensor<double>(2, 2, rowIndices, colIndices, values);

        var dense = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        // Act
        var result = _engine.SpMM(sparse, dense);

        // Assert: 2I * Dense = 2 * Dense
        Assert.Equal(2.0, result[0, 0], precision: 10);
        Assert.Equal(4.0, result[0, 1], precision: 10);
        Assert.Equal(6.0, result[1, 0], precision: 10);
        Assert.Equal(8.0, result[1, 1], precision: 10);
    }

    #endregion

    #region SparseToDense Tests

    [Fact]
    public void SparseToDense_SimpleMatrix_ConvertsCorrectly()
    {
        // Arrange
        var rowIndices = new[] { 0, 0, 1, 2 };
        var colIndices = new[] { 0, 2, 1, 2 };
        var values = new[] { 1.0, 2.0, 3.0, 4.0 };
        var sparse = new SparseTensor<double>(3, 3, rowIndices, colIndices, values);

        // Act
        var result = _engine.SparseToDense(sparse);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
        Assert.Equal(1.0, result[0, 0]);
        Assert.Equal(0.0, result[0, 1]);
        Assert.Equal(2.0, result[0, 2]);
        Assert.Equal(0.0, result[1, 0]);
        Assert.Equal(3.0, result[1, 1]);
        Assert.Equal(0.0, result[1, 2]);
        Assert.Equal(0.0, result[2, 0]);
        Assert.Equal(0.0, result[2, 1]);
        Assert.Equal(4.0, result[2, 2]);
    }

    #endregion

    #region DenseToSparse Tests

    [Fact]
    public void DenseToSparse_SparseMatrix_ConvertsCorrectly()
    {
        // Arrange
        var dense = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0, 2.0 },
            { 0.0, 3.0, 0.0 },
            { 0.0, 0.0, 4.0 }
        });

        // Act
        var result = _engine.DenseToSparse(dense, 0.0);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
        Assert.Equal(4, result.NonZeroCount);
    }

    #endregion

    #region Coalesce Tests

    [Fact]
    public void Coalesce_DuplicateEntries_SumsValues()
    {
        // Arrange: Two entries at position (0,0) with values 1.0 and 2.0
        var rowIndices = new[] { 0, 0, 1 };
        var colIndices = new[] { 0, 0, 1 };
        var values = new[] { 1.0, 2.0, 3.0 };
        var sparse = new SparseTensor<double>(2, 2, rowIndices, colIndices, values);

        // Act
        var result = _engine.Coalesce(sparse);

        // Assert: (0,0) should have value 3.0 (1.0 + 2.0)
        Assert.Equal(2, result.NonZeroCount);
        var dense = _engine.SparseToDense(result);
        Assert.Equal(3.0, dense[0, 0], precision: 10);
        Assert.Equal(3.0, dense[1, 1], precision: 10);
    }

    #endregion

    #region SparseTranspose Tests

    [Fact]
    public void SparseTranspose_RectangularMatrix_TransposesCorrectly()
    {
        // Arrange: 2x3 matrix
        var rowIndices = new[] { 0, 0, 1 };
        var colIndices = new[] { 0, 2, 1 };
        var values = new[] { 1.0, 2.0, 3.0 };
        var sparse = new SparseTensor<double>(2, 3, rowIndices, colIndices, values);

        // Act
        var result = _engine.SparseTranspose(sparse);

        // Assert: Should be 3x2
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
        var dense = _engine.SparseToDense(result);
        Assert.Equal(1.0, dense[0, 0]);
        Assert.Equal(2.0, dense[2, 0]);
        Assert.Equal(3.0, dense[1, 1]);
    }

    #endregion

    #region AddSparseDense Tests

    [Fact]
    public void AddSparseDense_AddsCorrectly()
    {
        // Arrange
        var rowIndices = new[] { 0, 1 };
        var colIndices = new[] { 0, 1 };
        var values = new[] { 1.0, 2.0 };
        var sparse = new SparseTensor<double>(2, 2, rowIndices, colIndices, values);
        var dense = new Matrix<double>(new double[,] { { 3.0, 4.0 }, { 5.0, 6.0 } });

        // Act
        var result = _engine.AddSparseDense(sparse, dense);

        // Assert
        Assert.Equal(4.0, result[0, 0], precision: 10);  // 1 + 3
        Assert.Equal(4.0, result[0, 1], precision: 10);  // 0 + 4
        Assert.Equal(5.0, result[1, 0], precision: 10);  // 0 + 5
        Assert.Equal(8.0, result[1, 1], precision: 10);  // 2 + 6
    }

    #endregion
}
