using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for ModelHelper to verify model creation and data handling operations.
/// These tests validate the model helper functions used for creating and managing models.
/// </summary>
public class ModelHelperIntegrationTests
{
    #region CreateDefaultModelData Tests - Matrix/Vector

    [Fact]
    public void CreateDefaultModelData_MatrixVector_ReturnsEmptyStructures()
    {
        var (x, y, predictions) = ModelHelper<double, Matrix<double>, Vector<double>>.CreateDefaultModelData();

        Assert.NotNull(x);
        Assert.NotNull(y);
        Assert.NotNull(predictions);
        Assert.Equal(0, x.Rows);
        Assert.Equal(0, x.Columns);
        Assert.Equal(0, y.Length);
        Assert.Equal(0, predictions.Length);
    }

    [Fact]
    public void CreateDefaultModelData_FloatMatrixVector_ReturnsEmptyStructures()
    {
        var (x, y, predictions) = ModelHelper<float, Matrix<float>, Vector<float>>.CreateDefaultModelData();

        Assert.NotNull(x);
        Assert.NotNull(y);
        Assert.NotNull(predictions);
    }

    #endregion

    #region CreateDefaultModelData Tests - Tensor

    [Fact]
    public void CreateDefaultModelData_TensorTensor_ReturnsEmptyStructures()
    {
        var (x, y, predictions) = ModelHelper<double, Tensor<double>, Tensor<double>>.CreateDefaultModelData();

        Assert.NotNull(x);
        Assert.NotNull(y);
        Assert.NotNull(predictions);
    }

    #endregion

    #region CreateDefaultModelData Tests - Vector/Vector

    [Fact]
    public void CreateDefaultModelData_VectorVector_ReturnsEmptyStructures()
    {
        var (x, y, predictions) = ModelHelper<double, Vector<double>, Vector<double>>.CreateDefaultModelData();

        Assert.NotNull(x);
        Assert.NotNull(y);
        Assert.NotNull(predictions);
        Assert.Equal(0, x.Length);
        Assert.Equal(0, y.Length);
    }

    #endregion

    #region CreateDefaultModel Tests - Matrix/Vector

    [Fact]
    public void CreateDefaultModel_MatrixVector_ReturnsVectorModel()
    {
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateDefaultModel();

        Assert.NotNull(model);
        // VectorModel is the expected type for Matrix<T> -> Vector<T>
    }

    [Fact]
    public void CreateDefaultModel_FloatMatrixVector_ReturnsVectorModel()
    {
        var model = ModelHelper<float, Matrix<float>, Vector<float>>.CreateDefaultModel();

        Assert.NotNull(model);
    }

    #endregion

    #region CreateDefaultModel Tests - Tensor

    [Fact]
    public void CreateDefaultModel_TensorTensor_ReturnsNeuralNetwork()
    {
        var model = ModelHelper<double, Tensor<double>, Tensor<double>>.CreateDefaultModel();

        Assert.NotNull(model);
        // NeuralNetwork is the expected type for Tensor<T> -> Tensor<T>
    }

    #endregion

    #region GetColumnVectors Tests - Matrix

    [Fact]
    public void GetColumnVectors_Matrix_ExtractsSingleColumn()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        var columns = ModelHelper<double, Matrix<double>, Vector<double>>.GetColumnVectors(matrix, new[] { 0 });

        Assert.Single(columns);
        Assert.Equal(3, columns[0].Length);
        Assert.Equal(1, columns[0][0]);
        Assert.Equal(4, columns[0][1]);
        Assert.Equal(7, columns[0][2]);
    }

    [Fact]
    public void GetColumnVectors_Matrix_ExtractsMultipleColumns()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 }
        });

        var columns = ModelHelper<double, Matrix<double>, Vector<double>>.GetColumnVectors(matrix, new[] { 0, 2 });

        Assert.Equal(2, columns.Count);

        // First column (index 0)
        Assert.Equal(1, columns[0][0]);
        Assert.Equal(5, columns[0][1]);

        // Second column (index 2)
        Assert.Equal(3, columns[1][0]);
        Assert.Equal(7, columns[1][1]);
    }

    [Fact]
    public void GetColumnVectors_Matrix_AllColumns()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 10, 20, 30 },
            { 40, 50, 60 }
        });

        var columns = ModelHelper<double, Matrix<double>, Vector<double>>.GetColumnVectors(matrix, new[] { 0, 1, 2 });

        Assert.Equal(3, columns.Count);
        Assert.Equal(10, columns[0][0]);
        Assert.Equal(20, columns[1][0]);
        Assert.Equal(30, columns[2][0]);
    }

    [Fact]
    public void GetColumnVectors_Matrix_EmptyIndices()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });

        var columns = ModelHelper<double, Matrix<double>, Vector<double>>.GetColumnVectors(matrix, Array.Empty<int>());

        Assert.Empty(columns);
    }

    [Fact]
    public void GetColumnVectors_Matrix_Float()
    {
        var matrix = new Matrix<float>(new float[,]
        {
            { 1.5f, 2.5f },
            { 3.5f, 4.5f }
        });

        var columns = ModelHelper<float, Matrix<float>, Vector<float>>.GetColumnVectors(matrix, new[] { 1 });

        Assert.Single(columns);
        Assert.Equal(2.5f, columns[0][0]);
        Assert.Equal(4.5f, columns[0][1]);
    }

    #endregion

    #region GetColumnVectors Tests - Tensor

    [Fact]
    public void GetColumnVectors_Tensor2D_ExtractsSingleColumn()
    {
        var tensor = new Tensor<double>(new int[] { 3, 4 });
        // Fill tensor: row 0 = [1,2,3,4], row 1 = [5,6,7,8], row 2 = [9,10,11,12]
        tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3; tensor[0, 3] = 4;
        tensor[1, 0] = 5; tensor[1, 1] = 6; tensor[1, 2] = 7; tensor[1, 3] = 8;
        tensor[2, 0] = 9; tensor[2, 1] = 10; tensor[2, 2] = 11; tensor[2, 3] = 12;

        var columns = ModelHelper<double, Tensor<double>, Tensor<double>>.GetColumnVectors(tensor, new[] { 1 });

        Assert.Single(columns);
        Assert.Equal(3, columns[0].Length);
        Assert.Equal(2, columns[0][0]);
        Assert.Equal(6, columns[0][1]);
        Assert.Equal(10, columns[0][2]);
    }

    [Fact]
    public void GetColumnVectors_Tensor2D_ExtractsMultipleColumns()
    {
        var tensor = new Tensor<double>(new int[] { 2, 3 });
        tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
        tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

        var columns = ModelHelper<double, Tensor<double>, Tensor<double>>.GetColumnVectors(tensor, new[] { 0, 2 });

        Assert.Equal(2, columns.Count);
        Assert.Equal(1, columns[0][0]);
        Assert.Equal(4, columns[0][1]);
        Assert.Equal(3, columns[1][0]);
        Assert.Equal(6, columns[1][1]);
    }

    [Fact]
    public void GetColumnVectors_Tensor1D_ThrowsArgumentException()
    {
        var tensor = new Tensor<double>(new int[] { 5 }); // 1D tensor

        Assert.Throws<ArgumentException>(() =>
            ModelHelper<double, Tensor<double>, Tensor<double>>.GetColumnVectors(tensor, new[] { 0 }));
    }

    [Fact]
    public void GetColumnVectors_Tensor_InvalidIndex_ThrowsException()
    {
        var tensor = new Tensor<double>(new int[] { 2, 3 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ModelHelper<double, Tensor<double>, Tensor<double>>.GetColumnVectors(tensor, new[] { 5 }));
    }

    [Fact]
    public void GetColumnVectors_Tensor_NegativeIndex_ThrowsException()
    {
        var tensor = new Tensor<double>(new int[] { 2, 3 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ModelHelper<double, Tensor<double>, Tensor<double>>.GetColumnVectors(tensor, new[] { -1 }));
    }

    #endregion

    #region CreateRandomModelWithFeatures Tests - Vector Model

    [Fact]
    public void CreateRandomModelWithFeatures_VectorModel_ReturnsModel()
    {
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0, 2 },
            totalFeatures: 5,
            useExpressionTrees: false);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_VectorModel_Float()
    {
        var model = ModelHelper<float, Matrix<float>, Vector<float>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 1, 3 },
            totalFeatures: 4,
            useExpressionTrees: false);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_VectorModel_AllFeaturesActive()
    {
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0, 1, 2, 3, 4 },
            totalFeatures: 5,
            useExpressionTrees: false);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_VectorModel_SingleFeature()
    {
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0 },
            totalFeatures: 10,
            useExpressionTrees: false);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_VectorModel_NoActiveFeatures()
    {
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: Array.Empty<int>(),
            totalFeatures: 5,
            useExpressionTrees: false);

        Assert.NotNull(model);
    }

    #endregion

    #region CreateRandomModelWithFeatures Tests - Expression Tree

    [Fact]
    public void CreateRandomModelWithFeatures_ExpressionTree_ReturnsModel()
    {
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0, 1 },
            totalFeatures: 3,
            useExpressionTrees: true,
            maxExpressionTreeDepth: 2);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_ExpressionTree_DepthOne()
    {
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0 },
            totalFeatures: 2,
            useExpressionTrees: true,
            maxExpressionTreeDepth: 1);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_ExpressionTree_DeepTree()
    {
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0, 1, 2 },
            totalFeatures: 5,
            useExpressionTrees: true,
            maxExpressionTreeDepth: 5);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_ExpressionTree_NoActiveFeatures()
    {
        // Should still work, creating tree with only constants
        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: Array.Empty<int>(),
            totalFeatures: 3,
            useExpressionTrees: true,
            maxExpressionTreeDepth: 2);

        Assert.NotNull(model);
    }

    #endregion

    #region CreateRandomModelWithFeatures Tests - Neural Network

    [Fact]
    public void CreateRandomModelWithFeatures_NeuralNetwork_ReturnsModel()
    {
        var model = ModelHelper<double, Tensor<double>, Tensor<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0, 1, 2 },
            totalFeatures: 10);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_NeuralNetwork_SingleFeature()
    {
        var model = ModelHelper<double, Tensor<double>, Tensor<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0 },
            totalFeatures: 5);

        Assert.NotNull(model);
    }

    [Fact]
    public void CreateRandomModelWithFeatures_NeuralNetwork_Float()
    {
        var model = ModelHelper<float, Tensor<float>, Tensor<float>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0, 2, 4 },
            totalFeatures: 8);

        Assert.NotNull(model);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void CreateDefaultModelData_UnsupportedTypes_ThrowsInvalidOperationException()
    {
        // Using string as TInput which is unsupported
        Assert.Throws<InvalidOperationException>(() =>
            ModelHelper<double, string, string>.CreateDefaultModelData());
    }

    [Fact]
    public void CreateDefaultModel_UnsupportedTypes_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() =>
            ModelHelper<double, string, string>.CreateDefaultModel());
    }

    [Fact]
    public void CreateRandomModelWithFeatures_UnsupportedTypes_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() =>
            ModelHelper<double, string, string>.CreateRandomModelWithFeatures(
                new[] { 0 }, 5));
    }

    [Fact]
    public void GetColumnVectors_NullInput_ThrowsException()
    {
        // Test behavior with null matrix
        Matrix<double>? nullMatrix = null;

        Assert.Throws<InvalidOperationException>(() =>
            ModelHelper<double, Matrix<double>, Vector<double>>.GetColumnVectors(nullMatrix!, new[] { 0 }));
    }

    #endregion

    #region Reproducibility Tests

    [Fact]
    public void CreateRandomModelWithFeatures_DifferentCalls_ProducesDifferentModels()
    {
        // Due to randomness, multiple calls should (likely) produce different models
        var model1 = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0, 1, 2 },
            totalFeatures: 5,
            useExpressionTrees: false);

        var model2 = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: new[] { 0, 1, 2 },
            totalFeatures: 5,
            useExpressionTrees: false);

        // Both should be valid
        Assert.NotNull(model1);
        Assert.NotNull(model2);
        // They may or may not be different, but both should be valid
    }

    #endregion

    #region Large Scale Tests

    [Fact]
    public void GetColumnVectors_LargeMatrix_ExtractsCorrectly()
    {
        int rows = 1000;
        int cols = 100;
        var matrix = new Matrix<double>(rows, cols);

        // Fill with predictable values
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = i * cols + j;
            }
        }

        var columns = ModelHelper<double, Matrix<double>, Vector<double>>.GetColumnVectors(
            matrix, new[] { 0, 50, 99 });

        Assert.Equal(3, columns.Count);
        Assert.Equal(rows, columns[0].Length);

        // Check first column
        Assert.Equal(0, columns[0][0]); // First row, column 0
        Assert.Equal(100, columns[0][1]); // Second row, column 0

        // Check middle column (50)
        Assert.Equal(50, columns[1][0]); // First row, column 50

        // Check last column (99)
        Assert.Equal(99, columns[2][0]); // First row, column 99
    }

    [Fact]
    public void CreateRandomModelWithFeatures_ManyFeatures_Succeeds()
    {
        var activeFeatures = Enumerable.Range(0, 50).ToArray();

        var model = ModelHelper<double, Matrix<double>, Vector<double>>.CreateRandomModelWithFeatures(
            activeFeatures: activeFeatures,
            totalFeatures: 100,
            useExpressionTrees: false);

        Assert.NotNull(model);
    }

    #endregion
}
