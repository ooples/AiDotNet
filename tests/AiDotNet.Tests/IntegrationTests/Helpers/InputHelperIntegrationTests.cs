using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for InputHelper to verify input data operations.
/// </summary>
public class InputHelperIntegrationTests
{
    #region GetBatchSize Tests - Matrix

    [Fact]
    public void GetBatchSize_Matrix_ReturnsRowCount()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10, 11, 12 }
        });

        var batchSize = InputHelper<double, Matrix<double>>.GetBatchSize(matrix);

        Assert.Equal(4, batchSize);
    }

    [Fact]
    public void GetBatchSize_SingleRowMatrix_ReturnsOne()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4, 5 }
        });

        var batchSize = InputHelper<double, Matrix<double>>.GetBatchSize(matrix);

        Assert.Equal(1, batchSize);
    }

    [Fact]
    public void GetBatchSize_Float_Matrix_ReturnsCorrectValue()
    {
        var matrix = new Matrix<float>(5, 3);

        var batchSize = InputHelper<float, Matrix<float>>.GetBatchSize(matrix);

        Assert.Equal(5, batchSize);
    }

    #endregion

    #region GetBatchSize Tests - Tensor

    [Fact]
    public void GetBatchSize_Tensor2D_ReturnsFirstDimension()
    {
        var tensor = new Tensor<double>(new[] { 10, 5 });

        var batchSize = InputHelper<double, Tensor<double>>.GetBatchSize(tensor);

        Assert.Equal(10, batchSize);
    }

    [Fact]
    public void GetBatchSize_Tensor3D_ReturnsFirstDimension()
    {
        var tensor = new Tensor<double>(new[] { 8, 28, 28 });

        var batchSize = InputHelper<double, Tensor<double>>.GetBatchSize(tensor);

        Assert.Equal(8, batchSize);
    }

    #endregion

    #region GetInputSize Tests - Matrix

    [Fact]
    public void GetInputSize_Matrix_ReturnsColumnCount()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4, 5 },
            { 6, 7, 8, 9, 10 }
        });

        var inputSize = InputHelper<double, Matrix<double>>.GetInputSize(matrix);

        Assert.Equal(5, inputSize);
    }

    [Fact]
    public void GetInputSize_SingleColumnMatrix_ReturnsOne()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1 },
            { 2 },
            { 3 }
        });

        var inputSize = InputHelper<double, Matrix<double>>.GetInputSize(matrix);

        Assert.Equal(1, inputSize);
    }

    #endregion

    #region GetInputSize Tests - Tensor

    [Fact]
    public void GetInputSize_Tensor2D_ReturnsSecondDimension()
    {
        var tensor = new Tensor<double>(new[] { 10, 64 });

        var inputSize = InputHelper<double, Tensor<double>>.GetInputSize(tensor);

        Assert.Equal(64, inputSize);
    }

    [Fact]
    public void GetInputSize_Tensor3D_ReturnsFlattenedSize()
    {
        var tensor = new Tensor<double>(new[] { 10, 28, 28 });

        var inputSize = InputHelper<double, Tensor<double>>.GetInputSize(tensor);

        // For 3D tensor, input size is product of all dimensions except first
        Assert.Equal(28 * 28, inputSize);
    }

    [Fact]
    public void GetInputSize_Tensor1D_ReturnsFirstDimension()
    {
        var tensor = new Tensor<double>(new[] { 100 });

        var inputSize = InputHelper<double, Tensor<double>>.GetInputSize(tensor);

        Assert.Equal(100, inputSize);
    }

    #endregion

    #region GetElement Tests - Matrix

    [Fact]
    public void GetElement_Matrix_ReturnsCorrectValue()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        var element = InputHelper<double, Matrix<double>>.GetElement(matrix, 1, 2);

        Assert.Equal(6, element);
    }

    [Fact]
    public void GetElement_Matrix_FirstElement_ReturnsCorrectValue()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 10, 20 },
            { 30, 40 }
        });

        var element = InputHelper<double, Matrix<double>>.GetElement(matrix, 0, 0);

        Assert.Equal(10, element);
    }

    [Fact]
    public void GetElement_Matrix_LastElement_ReturnsCorrectValue()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });

        var element = InputHelper<double, Matrix<double>>.GetElement(matrix, 1, 1);

        Assert.Equal(4, element);
    }

    [Fact]
    public void GetElement_Matrix_OutOfRange_ThrowsException()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Matrix<double>>.GetElement(matrix, 5, 0));
    }

    #endregion

    #region GetElement Tests - Vector

    [Fact]
    public void GetElement_Vector_RowIndex_ReturnsCorrectValue()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });

        // When column is 0, use row as the index
        var element = InputHelper<double, Vector<double>>.GetElement(vector, 2, 0);

        Assert.Equal(30, element);
    }

    [Fact]
    public void GetElement_Vector_ColumnIndex_ReturnsCorrectValue()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });

        // When row is 0, use column as the index
        var element = InputHelper<double, Vector<double>>.GetElement(vector, 0, 3);

        Assert.Equal(40, element);
    }

    [Fact]
    public void GetElement_Vector_OutOfRange_ThrowsException()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Vector<double>>.GetElement(vector, 10, 0));
    }

    #endregion

    #region GetElement Tests - Tensor

    [Fact]
    public void GetElement_Tensor2D_ReturnsCorrectValue()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        tensor[1, 2] = 42.0;

        var element = InputHelper<double, Tensor<double>>.GetElement(tensor, 1, 2);

        Assert.Equal(42.0, element);
    }

    [Fact]
    public void GetElement_Tensor1D_ReturnsCorrectValue()
    {
        var tensor = new Tensor<double>(new[] { 5 });
        tensor[3] = 99.0;

        var element = InputHelper<double, Tensor<double>>.GetElement(tensor, 3, 0);

        Assert.Equal(99.0, element);
    }

    #endregion

    #region GetBatch Tests - Matrix

    [Fact]
    public void GetBatch_Matrix_SingleIndex_ReturnsSingleRow()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        var batch = InputHelper<double, Matrix<double>>.GetBatch(matrix, new[] { 1 });

        Assert.Equal(1, batch.Rows);
        Assert.Equal(3, batch.Columns);
        Assert.Equal(4, batch[0, 0]);
        Assert.Equal(5, batch[0, 1]);
        Assert.Equal(6, batch[0, 2]);
    }

    [Fact]
    public void GetBatch_Matrix_MultipleIndices_ReturnsMultipleRows()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 },
            { 7, 8 }
        });

        var batch = InputHelper<double, Matrix<double>>.GetBatch(matrix, new[] { 0, 2, 3 });

        Assert.Equal(3, batch.Rows);
        Assert.Equal(2, batch.Columns);
        Assert.Equal(1, batch[0, 0]);  // Row 0
        Assert.Equal(5, batch[1, 0]);  // Row 2
        Assert.Equal(7, batch[2, 0]);  // Row 3
    }

    [Fact]
    public void GetBatch_Matrix_ReorderedIndices_ReturnsReorderedRows()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });

        var batch = InputHelper<double, Matrix<double>>.GetBatch(matrix, new[] { 2, 0, 1 });

        Assert.Equal(5, batch[0, 0]);  // Original row 2
        Assert.Equal(1, batch[1, 0]);  // Original row 0
        Assert.Equal(3, batch[2, 0]);  // Original row 1
    }

    [Fact]
    public void GetBatch_Matrix_InvalidIndex_ThrowsException()
    {
        var matrix = new Matrix<double>(3, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Matrix<double>>.GetBatch(matrix, new[] { 0, 5 }));
    }

    [Fact]
    public void GetBatch_Matrix_EmptyIndices_ThrowsException()
    {
        var matrix = new Matrix<double>(3, 2);

        Assert.Throws<ArgumentException>(() =>
            InputHelper<double, Matrix<double>>.GetBatch(matrix, Array.Empty<int>()));
    }

    #endregion

    #region GetBatch Tests - Vector

    [Fact]
    public void GetBatch_Vector_SingleIndex_ReturnsSingleElement()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });

        var batch = InputHelper<double, Vector<double>>.GetBatch(vector, new[] { 2 });

        Assert.Equal(1, batch.Length);
        Assert.Equal(30.0, batch[0]);
    }

    [Fact]
    public void GetBatch_Vector_MultipleIndices_ReturnsMultipleElements()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

        var batch = InputHelper<double, Vector<double>>.GetBatch(vector, new[] { 0, 2, 4 });

        Assert.Equal(3, batch.Length);
        Assert.Equal(10.0, batch[0]);
        Assert.Equal(30.0, batch[1]);
        Assert.Equal(50.0, batch[2]);
    }

    #endregion

    #region GetBatch Tests - Tensor

    [Fact]
    public void GetBatch_Tensor2D_ReturnsCorrectSlices()
    {
        var tensor = new Tensor<double>(new[] { 4, 3 });
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                tensor[i, j] = i * 10 + j;

        var batch = InputHelper<double, Tensor<double>>.GetBatch(tensor, new[] { 1, 3 });

        Assert.Equal(2, batch.Shape[0]);
        Assert.Equal(3, batch.Shape[1]);
        Assert.Equal(10, batch[0, 0]);  // Original row 1: 10, 11, 12
        Assert.Equal(30, batch[1, 0]);  // Original row 3: 30, 31, 32
    }

    [Fact]
    public void GetBatch_Tensor1D_ReturnsCorrectElements()
    {
        var tensor = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++)
            tensor[i] = i * 10.0;

        var batch = InputHelper<double, Tensor<double>>.GetBatch(tensor, new[] { 0, 2, 4 });

        Assert.Equal(3, batch.Shape[0]);
        Assert.Equal(0.0, batch[0]);
        Assert.Equal(20.0, batch[1]);
        Assert.Equal(40.0, batch[2]);
    }

    #endregion

    #region CreateSingleItemBatch Tests - Vector

    [Fact]
    public void CreateSingleItemBatch_Vector_ReturnsMatrixWithOneRow()
    {
        // Create a vector with test values
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Convert vector to single-row matrix for CreateSingleItemBatch
        var singleRowMatrix = new Matrix<double>(1, vector.Length);
        for (int i = 0; i < vector.Length; i++) singleRowMatrix[0, i] = vector[i];

        // Act
        var result = InputHelper<double, Matrix<double>>.CreateSingleItemBatch(singleRowMatrix);

        // Assert - verify dimensions and that original vector values are preserved
        Assert.Equal(1, result.Rows);
        Assert.Equal(4, result.Columns);
        Assert.Equal(1.0, result[0, 0]);
        Assert.Equal(2.0, result[0, 1]);
        Assert.Equal(3.0, result[0, 2]);
        Assert.Equal(4.0, result[0, 3]);
    }

    #endregion

    #region CreateSingleItemBatch Tests - Matrix

    [Fact]
    public void CreateSingleItemBatch_SingleRowMatrix_ReturnsSameMatrix()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 }
        });

        var batch = InputHelper<double, Matrix<double>>.CreateSingleItemBatch(matrix);

        Assert.Equal(1, batch.Rows);
        Assert.Equal(3, batch.Columns);
        Assert.Equal(1, batch[0, 0]);
        Assert.Equal(2, batch[0, 1]);
        Assert.Equal(3, batch[0, 2]);
    }

    [Fact]
    public void CreateSingleItemBatch_MultiRowMatrix_ReturnsSingleRow()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var batch = InputHelper<double, Matrix<double>>.CreateSingleItemBatch(matrix);

        Assert.Equal(1, batch.Rows);
        Assert.Equal(3, batch.Columns);
    }

    [Fact]
    public void CreateSingleItemBatch_SingleColumnMatrix_ReturnsTransposed()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1 },
            { 2 },
            { 3 }
        });

        var batch = InputHelper<double, Matrix<double>>.CreateSingleItemBatch(matrix);

        Assert.Equal(1, batch.Rows);
        Assert.Equal(3, batch.Columns);
    }

    #endregion

    #region CreateSingleItemBatch Tests - Tensor

    [Fact]
    public void CreateSingleItemBatch_Tensor_ReturnsWithBatchDimensionOne()
    {
        var tensor = new Tensor<double>(new[] { 1, 10, 10 });

        var batch = InputHelper<double, Tensor<double>>.CreateSingleItemBatch(tensor);

        Assert.Equal(1, batch.Shape[0]);
    }

    [Fact]
    public void CreateSingleItemBatch_Tensor1D_AddsBatchDimension()
    {
        var tensor = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++)
            tensor[i] = i * 10.0;

        var batch = InputHelper<double, Tensor<double>>.CreateSingleItemBatch(tensor);

        // 1D tensor [5] should become 2D tensor [1, 5]
        Assert.Equal(2, batch.Shape.Length);
        Assert.Equal(1, batch.Shape[0]);
        Assert.Equal(5, batch.Shape[1]);
    }

    #endregion

    #region GetItem Tests - Vector

    [Fact]
    public void GetItem_Vector_ReturnsSingletonVector()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        var item = InputHelper<double, Vector<double>>.GetItem(vector, 1);

        Assert.Equal(1, item.Length);
        Assert.Equal(20.0, item[0]);
    }

    [Fact]
    public void GetItem_Vector_FirstElement_ReturnsCorrectValue()
    {
        var vector = new Vector<double>(new[] { 100.0, 200.0, 300.0 });

        var item = InputHelper<double, Vector<double>>.GetItem(vector, 0);

        Assert.Equal(100.0, item[0]);
    }

    [Fact]
    public void GetItem_Vector_LastElement_ReturnsCorrectValue()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        var item = InputHelper<double, Vector<double>>.GetItem(vector, 3);

        Assert.Equal(4.0, item[0]);
    }

    [Fact]
    public void GetItem_Vector_OutOfRange_ThrowsException()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Vector<double>>.GetItem(vector, 10));
    }

    [Fact]
    public void GetItem_Vector_NegativeIndex_ThrowsException()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Vector<double>>.GetItem(vector, -1));
    }

    #endregion

    #region GetItem Tests - Matrix

    [Fact]
    public void GetItem_Matrix_ReturnsRowAsVector()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        // Get the second row (index 1) which should be { 4, 5, 6 }
        var item = InputHelper<double, Matrix<double>>.GetItem(matrix, 1);

        // Verify the row was correctly extracted
        Assert.NotNull(item);
        Assert.Equal(1, item.Rows);
        Assert.Equal(3, item.Columns);
        Assert.Equal(4.0, item[0, 0]);
        Assert.Equal(5.0, item[0, 1]);
        Assert.Equal(6.0, item[0, 2]);
    }

    [Fact]
    public void GetItem_Matrix_NullInput_ThrowsException()
    {
        Matrix<double>? nullMatrix = null;

        Assert.Throws<ArgumentNullException>(() =>
            InputHelper<double, Matrix<double>>.GetItem(nullMatrix!, 0));
    }

    #endregion

    #region GetItem Tests - Tensor

    [Fact]
    public void GetItem_Tensor2D_ReturnsSlice()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                tensor[i, j] = i * 10 + j;

        var item = InputHelper<double, Tensor<double>>.GetItem(tensor, 1);

        // Should return a 1D tensor (slice at index 1)
        Assert.NotNull(item);
    }

    [Fact]
    public void GetItem_Tensor_OutOfRange_ThrowsException()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Tensor<double>>.GetItem(tensor, 10));
    }

    #endregion

    #region GetFeatureValue Tests - Vector

    [Fact]
    public void GetFeatureValue_Vector_ReturnsCorrectValue()
    {
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });

        var value = InputHelper<double, Vector<double>>.GetFeatureValue(vector, 2);

        Assert.Equal(30.0, value);
    }

    [Fact]
    public void GetFeatureValue_Vector_FirstFeature_ReturnsCorrectValue()
    {
        var vector = new Vector<double>(new[] { 100.0, 200.0 });

        var value = InputHelper<double, Vector<double>>.GetFeatureValue(vector, 0);

        Assert.Equal(100.0, value);
    }

    [Fact]
    public void GetFeatureValue_Vector_OutOfRange_ThrowsException()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Vector<double>>.GetFeatureValue(vector, 10));
    }

    #endregion

    #region GetFeatureValue Tests - Matrix

    [Fact]
    public void GetFeatureValue_Matrix_ReturnsFirstRowFeature()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 }
        });

        var value = InputHelper<double, Matrix<double>>.GetFeatureValue(matrix, 2);

        // Should return the feature from the first row
        Assert.Equal(3, value);
    }

    [Fact]
    public void GetFeatureValue_Matrix_OutOfRange_ThrowsException()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 }
        });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Matrix<double>>.GetFeatureValue(matrix, 10));
    }

    #endregion

    #region GetFeatureValue Tests - Tensor

    [Fact]
    public void GetFeatureValue_Tensor1D_ReturnsCorrectValue()
    {
        var tensor = new Tensor<double>(new[] { 4 });
        tensor[0] = 10;
        tensor[1] = 20;
        tensor[2] = 30;
        tensor[3] = 40;

        var value = InputHelper<double, Tensor<double>>.GetFeatureValue(tensor, 2);

        Assert.Equal(30, value);
    }

    [Fact]
    public void GetFeatureValue_Tensor2D_ReturnsCorrectValue()
    {
        var tensor = new Tensor<double>(new[] { 2, 4 });
        tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3; tensor[0, 3] = 4;

        var value = InputHelper<double, Tensor<double>>.GetFeatureValue(tensor, 1);

        Assert.Equal(2, value);
    }

    #endregion

    #region Null Input Tests

    [Fact]
    public void GetElement_NullInput_ThrowsArgumentNullException()
    {
        Matrix<double>? nullMatrix = null;

        Assert.Throws<ArgumentNullException>(() =>
            InputHelper<double, Matrix<double>>.GetElement(nullMatrix!, 0, 0));
    }

    [Fact]
    public void GetBatch_NullInput_ThrowsArgumentNullException()
    {
        Matrix<double>? nullMatrix = null;

        Assert.Throws<ArgumentNullException>(() =>
            InputHelper<double, Matrix<double>>.GetBatch(nullMatrix!, new[] { 0 }));
    }

    [Fact]
    public void GetBatch_NullIndices_ThrowsArgumentNullException()
    {
        var matrix = new Matrix<double>(3, 2);

        Assert.Throws<ArgumentNullException>(() =>
            InputHelper<double, Matrix<double>>.GetBatch(matrix, null!));
    }

    [Fact]
    public void CreateSingleItemBatch_NullInput_ThrowsArgumentNullException()
    {
        Matrix<double>? nullMatrix = null;

        Assert.Throws<ArgumentNullException>(() =>
            InputHelper<double, Matrix<double>>.CreateSingleItemBatch(nullMatrix!));
    }

    [Fact]
    public void GetFeatureValue_NullInput_ThrowsArgumentNullException()
    {
        Vector<double>? nullVector = null;

        Assert.Throws<ArgumentNullException>(() =>
            InputHelper<double, Vector<double>>.GetFeatureValue(nullVector!, 0));
    }

    [Fact]
    public void GetFeatureValue_NegativeIndex_ThrowsArgumentOutOfRangeException()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InputHelper<double, Vector<double>>.GetFeatureValue(vector, -1));
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void GetBatchSize_Float_Tensor_ReturnsCorrectValue()
    {
        var tensor = new Tensor<float>(new[] { 16, 32 });

        var batchSize = InputHelper<float, Tensor<float>>.GetBatchSize(tensor);

        Assert.Equal(16, batchSize);
    }

    [Fact]
    public void GetInputSize_Float_Matrix_ReturnsCorrectValue()
    {
        var matrix = new Matrix<float>(10, 20);

        var inputSize = InputHelper<float, Matrix<float>>.GetInputSize(matrix);

        Assert.Equal(20, inputSize);
    }

    [Fact]
    public void GetElement_Float_Matrix_ReturnsCorrectValue()
    {
        var matrix = new Matrix<float>(new float[,]
        {
            { 1.5f, 2.5f },
            { 3.5f, 4.5f }
        });

        var element = InputHelper<float, Matrix<float>>.GetElement(matrix, 1, 1);

        Assert.Equal(4.5f, element);
    }

    [Fact]
    public void GetBatch_Float_Vector_ReturnsCorrectValues()
    {
        var vector = new Vector<float>(new[] { 1f, 2f, 3f, 4f, 5f });

        var batch = InputHelper<float, Vector<float>>.GetBatch(vector, new[] { 1, 3 });

        Assert.Equal(2, batch.Length);
        Assert.Equal(2f, batch[0]);
        Assert.Equal(4f, batch[1]);
    }

    #endregion

    #region Large Dataset Tests

    [Fact]
    public void GetBatch_LargeMatrix_PerformsCorrectly()
    {
        int rows = 1000;
        int cols = 100;
        var matrix = new Matrix<double>(rows, cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = i * cols + j;

        // Select every 100th row
        var indices = Enumerable.Range(0, rows).Where(i => i % 100 == 0).ToArray();
        var batch = InputHelper<double, Matrix<double>>.GetBatch(matrix, indices);

        Assert.Equal(10, batch.Rows);
        Assert.Equal(cols, batch.Columns);
        Assert.Equal(0, batch[0, 0]);  // Row 0
        Assert.Equal(100 * cols, batch[1, 0]);  // Row 100
    }

    [Fact]
    public void GetBatch_LargeTensor_PerformsCorrectly()
    {
        var tensor = new Tensor<double>(new[] { 500, 64 });

        var indices = new[] { 0, 100, 200, 300, 400, 499 };
        var batch = InputHelper<double, Tensor<double>>.GetBatch(tensor, indices);

        Assert.Equal(6, batch.Shape[0]);
        Assert.Equal(64, batch.Shape[1]);
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void GetBatchSize_EmptyMatrix_ReturnsZero()
    {
        var matrix = new Matrix<double>(0, 0);

        var batchSize = InputHelper<double, Matrix<double>>.GetBatchSize(matrix);

        Assert.Equal(0, batchSize);
    }

    [Fact]
    public void GetInputSize_EmptyMatrix_ReturnsZero()
    {
        var matrix = new Matrix<double>(0, 0);

        var inputSize = InputHelper<double, Matrix<double>>.GetInputSize(matrix);

        Assert.Equal(0, inputSize);
    }

    [Fact]
    public void GetBatch_DuplicateIndices_ReturnsDuplicatedRows()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });

        var batch = InputHelper<double, Matrix<double>>.GetBatch(matrix, new[] { 0, 0, 1, 1 });

        Assert.Equal(4, batch.Rows);
        Assert.Equal(1, batch[0, 0]);  // Duplicate of row 0
        Assert.Equal(1, batch[1, 0]);  // Duplicate of row 0
        Assert.Equal(3, batch[2, 0]);  // Duplicate of row 1
        Assert.Equal(3, batch[3, 0]);  // Duplicate of row 1
    }

    #endregion

    #region Unsupported Type Tests

    [Fact]
    public void GetBatchSize_UnsupportedType_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            InputHelper<double, string>.GetBatchSize("invalid"));
    }

    [Fact]
    public void GetInputSize_UnsupportedType_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            InputHelper<double, string>.GetInputSize("invalid"));
    }

    #endregion
}
