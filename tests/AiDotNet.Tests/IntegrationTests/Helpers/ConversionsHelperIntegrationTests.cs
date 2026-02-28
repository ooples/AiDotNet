using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for ConversionsHelper:
/// ConvertToMatrix, ConvertToVector, ConvertToScalar,
/// ConvertObjectToVector, TensorToMatrix, MatrixToTensor,
/// VectorToTensor, ConvertToTensor, ConvertVectorToInput,
/// ConvertVectorToInputWithoutReference, GetSampleCount.
/// </summary>
public class ConversionsHelperIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region ConvertToMatrix

    [Fact]
    public void ConvertToMatrix_Matrix_ReturnsSameMatrix()
    {
        var matrix = new Matrix<double>(2, 3);
        matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
        matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

        var result = ConversionsHelper.ConvertToMatrix<double, Matrix<double>>(matrix);
        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
        Assert.Equal(1.0, result[0, 0], Tolerance);
    }

    [Fact]
    public void ConvertToMatrix_2DTensor_ConvertsCorrectly()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        var result = ConversionsHelper.ConvertToMatrix<double, Tensor<double>>(tensor);
        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void ConvertToMatrix_1DTensor_CreatesRowMatrix()
    {
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
        var result = ConversionsHelper.ConvertToMatrix<double, Tensor<double>>(tensor);
        Assert.Equal(1, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void ConvertToMatrix_UnsupportedType_Throws()
    {
        Assert.Throws<InvalidOperationException>(() =>
            ConversionsHelper.ConvertToMatrix<double, string>("hello"));
    }

    #endregion

    #region ConvertToVector

    [Fact]
    public void ConvertToVector_Vector_ReturnsSameVector()
    {
        var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var result = ConversionsHelper.ConvertToVector<double, Vector<double>>(vector);
        Assert.Equal(3, result.Length);
        Assert.Equal(1.0, result[0], Tolerance);
    }

    [Fact]
    public void ConvertToVector_Tensor_FlattensToVector()
    {
        var tensor = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 1, 2, 3, 4 }));
        var result = ConversionsHelper.ConvertToVector<double, Tensor<double>>(tensor);
        Assert.Equal(4, result.Length);
    }

    [Fact]
    public void ConvertToVector_Array_WrapsInVector()
    {
        var array = new double[] { 1.0, 2.0, 3.0 };
        var result = ConversionsHelper.ConvertToVector<double, double[]>(array);
        Assert.Equal(3, result.Length);
        Assert.Equal(1.0, result[0], Tolerance);
    }

    [Fact]
    public void ConvertToVector_Scalar_CreatesBinaryVector()
    {
        // Scalar 0.7 -> [0.3, 0.7] for binary classification
        var result = ConversionsHelper.ConvertToVector<double, double>(0.7);
        Assert.Equal(2, result.Length);
        Assert.Equal(0.3, result[0], Tolerance);
        Assert.Equal(0.7, result[1], Tolerance);
    }

    [Fact]
    public void ConvertToVector_ScalarOutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ConversionsHelper.ConvertToVector<double, double>(1.5));
    }

    #endregion

    #region ConvertToScalar

    [Fact]
    public void ConvertToScalar_Scalar_ReturnsSame()
    {
        var result = ConversionsHelper.ConvertToScalar<double, double>(42.0);
        Assert.Equal(42.0, result, Tolerance);
    }

    [Fact]
    public void ConvertToScalar_Vector_ReturnsFirstElement()
    {
        var vector = new Vector<double>(new double[] { 7.0, 8.0, 9.0 });
        var result = ConversionsHelper.ConvertToScalar<double, Vector<double>>(vector);
        Assert.Equal(7.0, result, Tolerance);
    }

    [Fact]
    public void ConvertToScalar_EmptyVector_Throws()
    {
        var vector = new Vector<double>(0);
        Assert.Throws<InvalidOperationException>(() =>
            ConversionsHelper.ConvertToScalar<double, Vector<double>>(vector));
    }

    [Fact]
    public void ConvertToScalar_Matrix_ReturnsFirstElement()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 99.0;
        var result = ConversionsHelper.ConvertToScalar<double, Matrix<double>>(matrix);
        Assert.Equal(99.0, result, Tolerance);
    }

    #endregion

    #region ConvertObjectToVector

    [Fact]
    public void ConvertObjectToVector_Null_ReturnsNull()
    {
        var result = ConversionsHelper.ConvertObjectToVector<double>(null);
        Assert.Null(result);
    }

    [Fact]
    public void ConvertObjectToVector_Vector_ReturnsSame()
    {
        var vector = new Vector<double>(new double[] { 1.0, 2.0 });
        var result = ConversionsHelper.ConvertObjectToVector<double>(vector);
        Assert.NotNull(result);
        Assert.Equal(2, result.Length);
    }

    [Fact]
    public void ConvertObjectToVector_Tensor_Converts()
    {
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
        var result = ConversionsHelper.ConvertObjectToVector<double>(tensor);
        Assert.NotNull(result);
        Assert.Equal(3, result.Length);
    }

    [Fact]
    public void ConvertObjectToVector_UnsupportedType_Throws()
    {
        Assert.Throws<InvalidOperationException>(() =>
            ConversionsHelper.ConvertObjectToVector<double>("unsupported"));
    }

    #endregion

    #region TensorToMatrix

    [Fact]
    public void TensorToMatrix_ValidDimensions_Converts()
    {
        var tensor = new Tensor<double>(new[] { 6 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        var result = ConversionsHelper.TensorToMatrix<double>(tensor, 2, 3);
        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void TensorToMatrix_MismatchedSize_Throws()
    {
        var tensor = new Tensor<double>(new[] { 6 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        Assert.Throws<ArgumentException>(() =>
            ConversionsHelper.TensorToMatrix<double>(tensor, 3, 3)); // 9 != 6
    }

    #endregion

    #region MatrixToTensor

    [Fact]
    public void MatrixToTensor_ValidShape_Converts()
    {
        var matrix = new Matrix<double>(2, 3);
        var result = ConversionsHelper.MatrixToTensor<double>(matrix, new[] { 2, 3 });
        Assert.Equal(new[] { 2, 3 }, result.Shape);
    }

    [Fact]
    public void MatrixToTensor_MismatchedSize_Throws()
    {
        var matrix = new Matrix<double>(2, 3); // 6 elements
        Assert.Throws<ArgumentException>(() =>
            ConversionsHelper.MatrixToTensor<double>(matrix, new[] { 3, 3 })); // 9 != 6
    }

    #endregion

    #region VectorToTensor

    [Fact]
    public void VectorToTensor_ValidShape_Converts()
    {
        var vector = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
        var result = ConversionsHelper.VectorToTensor<double>(vector, new[] { 2, 3 });
        Assert.Equal(new[] { 2, 3 }, result.Shape);
        Assert.Equal(6, result.Length);
    }

    [Fact]
    public void VectorToTensor_MismatchedLength_Throws()
    {
        var vector = new Vector<double>(new double[] { 1, 2, 3 });
        Assert.Throws<ArgumentException>(() =>
            ConversionsHelper.VectorToTensor<double>(vector, new[] { 2, 3 })); // 6 != 3
    }

    #endregion

    #region ConvertToTensor

    [Fact]
    public void ConvertToTensor_Tensor_ReturnsSame()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        var result = ConversionsHelper.ConvertToTensor<double>(tensor);
        Assert.Equal(new[] { 2, 3 }, result.Shape);
    }

    [Fact]
    public void ConvertToTensor_Matrix_Converts()
    {
        var matrix = new Matrix<double>(2, 3);
        var result = ConversionsHelper.ConvertToTensor<double>(matrix);
        Assert.Equal(6, result.Length);
    }

    [Fact]
    public void ConvertToTensor_Vector_Converts()
    {
        var vector = new Vector<double>(new double[] { 1, 2, 3 });
        var result = ConversionsHelper.ConvertToTensor<double>(vector);
        Assert.Equal(3, result.Length);
    }

    [Fact]
    public void ConvertToTensor_UnsupportedType_Throws()
    {
        Assert.Throws<InvalidOperationException>(() =>
            ConversionsHelper.ConvertToTensor<double>("unsupported"));
    }

    #endregion

    #region ConvertVectorToInputWithoutReference

    [Fact]
    public void ConvertVectorToInputWithoutReference_ToVector_ReturnsSame()
    {
        var vector = new Vector<double>(new double[] { 1, 2, 3 });
        var result = ConversionsHelper.ConvertVectorToInputWithoutReference<double, Vector<double>>(vector);
        Assert.Equal(3, result.Length);
    }

    [Fact]
    public void ConvertVectorToInputWithoutReference_ToTensor_CreatesBatch1()
    {
        var vector = new Vector<double>(new double[] { 1, 2, 3 });
        var result = ConversionsHelper.ConvertVectorToInputWithoutReference<double, Tensor<double>>(vector);
        Assert.Equal(new[] { 1, 3 }, result.Shape);
    }

    [Fact]
    public void ConvertVectorToInputWithoutReference_ToArray_ConvertsCorrectly()
    {
        var vector = new Vector<double>(new double[] { 1, 2, 3 });
        var result = ConversionsHelper.ConvertVectorToInputWithoutReference<double, double[]>(vector);
        Assert.Equal(3, result.Length);
        Assert.Equal(1.0, result[0], Tolerance);
    }

    [Fact]
    public void ConvertVectorToInputWithoutReference_ToScalar_ReturnsFirst()
    {
        var vector = new Vector<double>(new double[] { 42.0 });
        var result = ConversionsHelper.ConvertVectorToInputWithoutReference<double, double>(vector);
        Assert.Equal(42.0, result, Tolerance);
    }

    #endregion

    #region GetSampleCount

    [Fact]
    public void GetSampleCount_Matrix_ReturnsRows()
    {
        var matrix = new Matrix<double>(5, 3);
        int count = ConversionsHelper.GetSampleCount<double, Matrix<double>>(matrix);
        Assert.Equal(5, count);
    }

    [Fact]
    public void GetSampleCount_Vector_ReturnsLength()
    {
        var vector = new Vector<double>(new double[] { 1, 2, 3 });
        int count = ConversionsHelper.GetSampleCount<double, Vector<double>>(vector);
        Assert.Equal(3, count);
    }

    [Fact]
    public void GetSampleCount_Tensor_ReturnsFirstDim()
    {
        var tensor = new Tensor<double>(new[] { 4, 3, 2 });
        int count = ConversionsHelper.GetSampleCount<double, Tensor<double>>(tensor);
        Assert.Equal(4, count);
    }

    [Fact]
    public void GetSampleCount_Null_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ConversionsHelper.GetSampleCount<double, Matrix<double>>(null!));
    }

    #endregion
}
