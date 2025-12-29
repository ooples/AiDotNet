using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Results;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for OptimizerHelper to verify optimization utility operations.
/// </summary>
public class OptimizerHelperIntegrationTests
{
    #region SelectFeatures Tests - Matrix with Indices

    [Fact]
    public void SelectFeatures_Matrix_SingleFeature_ReturnsCorrectColumn()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        });

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.SelectFeatures(
            matrix, new List<int> { 1 });

        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
        Assert.Equal(2, result[0, 0]);
        Assert.Equal(6, result[1, 0]);
        Assert.Equal(10, result[2, 0]);
    }

    [Fact]
    public void SelectFeatures_Matrix_MultipleFeatures_ReturnsCorrectColumns()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 }
        });

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.SelectFeatures(
            matrix, new List<int> { 0, 2 });

        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
        Assert.Equal(1, result[0, 0]);
        Assert.Equal(3, result[0, 1]);
        Assert.Equal(5, result[1, 0]);
        Assert.Equal(7, result[1, 1]);
    }

    [Fact]
    public void SelectFeatures_Matrix_AllFeatures_ReturnsSameShape()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.SelectFeatures(
            matrix, new List<int> { 0, 1, 2 });

        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void SelectFeatures_Matrix_ReorderedFeatures_ReordersColumns()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 }
        });

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.SelectFeatures(
            matrix, new List<int> { 2, 0, 1 });

        Assert.Equal(3, result[0, 0]); // Originally column 2
        Assert.Equal(1, result[0, 1]); // Originally column 0
        Assert.Equal(2, result[0, 2]); // Originally column 1
    }

    [Fact]
    public void SelectFeatures_Matrix_Float_WorksCorrectly()
    {
        var matrix = new Matrix<float>(new float[,]
        {
            { 1.5f, 2.5f, 3.5f }
        });

        var result = OptimizerHelper<float, Matrix<float>, Vector<float>>.SelectFeatures(
            matrix, new List<int> { 1 });

        Assert.Equal(2.5f, result[0, 0]);
    }

    #endregion

    #region SelectFeatures Tests - Tensor with Indices

    [Fact]
    public void SelectFeatures_Tensor2D_SingleFeature_ReturnsCorrectColumn()
    {
        var tensor = new Tensor<double>(new[] { 2, 4 });
        tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3; tensor[0, 3] = 4;
        tensor[1, 0] = 5; tensor[1, 1] = 6; tensor[1, 2] = 7; tensor[1, 3] = 8;

        var result = OptimizerHelper<double, Tensor<double>, Tensor<double>>.SelectFeatures(
            tensor, new List<int> { 2 });

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(1, result.Shape[1]);
        Assert.Equal(3, result[0, 0]);
        Assert.Equal(7, result[1, 0]);
    }

    [Fact]
    public void SelectFeatures_Tensor2D_MultipleFeatures_ReturnsCorrectColumns()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3;
        tensor[1, 0] = 4; tensor[1, 1] = 5; tensor[1, 2] = 6;

        var result = OptimizerHelper<double, Tensor<double>, Tensor<double>>.SelectFeatures(
            tensor, new List<int> { 0, 2 });

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(2, result.Shape[1]);
        Assert.Equal(1, result[0, 0]);
        Assert.Equal(3, result[0, 1]);
    }

    [Fact]
    public void SelectFeatures_Tensor1D_ThrowsArgumentException()
    {
        var tensor = new Tensor<double>(new[] { 5 }); // 1D tensor

        Assert.Throws<ArgumentException>(() =>
            OptimizerHelper<double, Tensor<double>, Tensor<double>>.SelectFeatures(
                tensor, new List<int> { 0 }));
    }

    #endregion

    #region SelectFeatures Tests - With Vector Features

    [Fact]
    public void SelectFeatures_MatrixWithVectors_CreatesNewMatrixFromColumns()
    {
        var col1 = new Vector<double>(new[] { 1.0, 4.0, 7.0 });
        var col2 = new Vector<double>(new[] { 3.0, 6.0, 9.0 });
        var selectedFeatures = new List<Vector<double>> { col1, col2 };

        var dummyMatrix = new Matrix<double>(3, 3); // Used for type inference only

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.SelectFeatures(
            dummyMatrix, (IEnumerable<Vector<double>>)selectedFeatures);

        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
        Assert.Equal(1.0, result[0, 0]);
        Assert.Equal(3.0, result[0, 1]);
    }

    [Fact]
    public void SelectFeatures_TensorWithVectors_CreatesTensorFromColumns()
    {
        var col1 = new Vector<double>(new[] { 1.0, 2.0 });
        var col2 = new Vector<double>(new[] { 3.0, 4.0 });
        var selectedFeatures = new List<Vector<double>> { col1, col2 };

        var dummyTensor = new Tensor<double>(new[] { 2, 2 });

        var result = OptimizerHelper<double, Tensor<double>, Tensor<double>>.SelectFeatures(
            dummyTensor, (IEnumerable<Vector<double>>)selectedFeatures);

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(2, result.Shape[1]);
    }

    [Fact]
    public void SelectFeatures_WithVectors_EmptyList_ReturnsEmptyResult()
    {
        var selectedFeatures = new List<Vector<double>>();
        var dummyTensor = new Tensor<double>(new[] { 2, 2 });

        var result = OptimizerHelper<double, Tensor<double>, Tensor<double>>.SelectFeatures(
            dummyTensor, (IEnumerable<Vector<double>>)selectedFeatures);

        Assert.Equal(0, result.Shape[0]);
        Assert.Equal(0, result.Shape[1]);
    }

    [Fact]
    public void SelectFeatures_WithVectors_MismatchedLengths_ThrowsException()
    {
        var col1 = new Vector<double>(new[] { 1.0, 2.0 });
        var col2 = new Vector<double>(new[] { 3.0, 4.0, 5.0 }); // Different length
        var selectedFeatures = new List<Vector<double>> { col1, col2 };

        var dummyTensor = new Tensor<double>(new[] { 2, 2 });

        Assert.Throws<ArgumentException>(() =>
            OptimizerHelper<double, Tensor<double>, Tensor<double>>.SelectFeatures(
                dummyTensor, (IEnumerable<Vector<double>>)selectedFeatures));
    }

    #endregion

    #region CreateOptimizationInputData Tests

    [Fact]
    public void CreateOptimizationInputData_MatrixVector_CreatesValidObject()
    {
        var xTrain = new Matrix<double>(10, 5);
        var yTrain = new Vector<double>(10);
        var xVal = new Matrix<double>(5, 5);
        var yVal = new Vector<double>(5);
        var xTest = new Matrix<double>(3, 5);
        var yTest = new Vector<double>(3);

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.CreateOptimizationInputData(
            xTrain, yTrain, xVal, yVal, xTest, yTest);

        Assert.NotNull(result);
        Assert.Equal(xTrain, result.XTrain);
        Assert.Equal(yTrain, result.YTrain);
        Assert.Equal(xVal, result.XValidation);
        Assert.Equal(yVal, result.YValidation);
        Assert.Equal(xTest, result.XTest);
        Assert.Equal(yTest, result.YTest);
    }

    [Fact]
    public void CreateOptimizationInputData_TensorTensor_CreatesValidObject()
    {
        var xTrain = new Tensor<double>(new[] { 10, 5 });
        var yTrain = new Tensor<double>(new[] { 10, 1 });
        var xVal = new Tensor<double>(new[] { 5, 5 });
        var yVal = new Tensor<double>(new[] { 5, 1 });
        var xTest = new Tensor<double>(new[] { 3, 5 });
        var yTest = new Tensor<double>(new[] { 3, 1 });

        var result = OptimizerHelper<double, Tensor<double>, Tensor<double>>.CreateOptimizationInputData(
            xTrain, yTrain, xVal, yVal, xTest, yTest);

        Assert.NotNull(result);
        Assert.Equal(xTrain, result.XTrain);
        Assert.Equal(yTrain, result.YTrain);
    }

    [Fact]
    public void CreateOptimizationInputData_Float_WorksCorrectly()
    {
        var xTrain = new Matrix<float>(5, 3);
        var yTrain = new Vector<float>(5);
        var xVal = new Matrix<float>(2, 3);
        var yVal = new Vector<float>(2);
        var xTest = new Matrix<float>(1, 3);
        var yTest = new Vector<float>(1);

        var result = OptimizerHelper<float, Matrix<float>, Vector<float>>.CreateOptimizationInputData(
            xTrain, yTrain, xVal, yVal, xTest, yTest);

        Assert.NotNull(result);
    }

    #endregion

    #region CreateDatasetResult Tests

    [Fact]
    public void CreateDatasetResult_WithNullStats_UsesEmptyStats()
    {
        var predictions = new Vector<double>(5);
        var features = new Matrix<double>(5, 3);
        var y = new Vector<double>(5);

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.CreateDatasetResult(
            predictions,
            errorStats: null,
            actualBasicStats: null,
            predictedBasicStats: null,
            predictionStats: null,
            features,
            y);

        Assert.NotNull(result);
        Assert.NotNull(result.ErrorStats);
        Assert.NotNull(result.ActualBasicStats);
        Assert.NotNull(result.PredictedBasicStats);
        Assert.NotNull(result.PredictionStats);
    }

    [Fact]
    public void CreateDatasetResult_WithValidStats_PreservesStats()
    {
        var predictions = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var features = new Matrix<double>(3, 2);
        var y = new Vector<double>(new[] { 1.1, 2.1, 2.9 });
        var errorStats = ErrorStats<double>.Empty();
        var actualStats = BasicStats<double>.Empty();
        var predictedStats = BasicStats<double>.Empty();
        var predStats = PredictionStats<double>.Empty();

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.CreateDatasetResult(
            predictions, errorStats, actualStats, predictedStats, predStats, features, y);

        Assert.NotNull(result);
        Assert.Equal(predictions, result.Predictions);
        Assert.Equal(features, result.X);
        Assert.Equal(y, result.Y);
    }

    #endregion

    #region CreateOptimizationResult Tests

    [Fact]
    public void CreateOptimizationResult_CreatesValidResultObject()
    {
        var bestSolution = new VectorModel<double>(new Vector<double>(new[] { 0.5, 0.3, 0.2 }));
        var bestFitness = 0.95;
        var fitnessHistory = new List<double> { 0.5, 0.7, 0.85, 0.95 };
        var selectedFeatures = new List<Vector<double>>
        {
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0, 0.0 })
        };

        var predictions = new Vector<double>(5);
        var features = new Matrix<double>(5, 3);
        var y = new Vector<double>(5);

        var trainingResult = OptimizerHelper<double, Matrix<double>, Vector<double>>.CreateDatasetResult(
            predictions, null, null, null, null, features, y);
        var validationResult = OptimizerHelper<double, Matrix<double>, Vector<double>>.CreateDatasetResult(
            predictions, null, null, null, null, features, y);
        var testResult = OptimizerHelper<double, Matrix<double>, Vector<double>>.CreateDatasetResult(
            predictions, null, null, null, null, features, y);

        var fitDetectionResult = new FitDetectorResult<double>(FitType.GoodFit, 0.9);

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.CreateOptimizationResult(
            bestSolution,
            bestFitness,
            fitnessHistory,
            selectedFeatures,
            trainingResult,
            validationResult,
            testResult,
            fitDetectionResult,
            iterationCount: 100);

        Assert.NotNull(result);
        Assert.Equal(bestSolution, result.BestSolution);
        Assert.Equal(bestFitness, result.BestFitnessScore);
        Assert.Equal(100, result.Iterations);
        Assert.Equal(4, result.FitnessHistory.Length);
        Assert.Equal(2, result.SelectedFeatures.Count);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void SelectFeatures_Matrix_EmptyList_ReturnsEmptyMatrix()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.SelectFeatures(
            matrix, new List<int>());

        Assert.Equal(2, result.Rows);
        Assert.Equal(0, result.Columns);
    }

    [Fact]
    public void SelectFeatures_UnsupportedType_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            OptimizerHelper<double, string, string>.SelectFeatures(
                "invalid", new List<int> { 0 }));
    }

    [Fact]
    public void SelectFeatures_Matrix_LargeDataset_PerformsCorrectly()
    {
        int rows = 1000;
        int cols = 100;
        var matrix = new Matrix<double>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = i * cols + j;
            }
        }

        var selectedIndices = new List<int> { 0, 25, 50, 75, 99 };
        var result = OptimizerHelper<double, Matrix<double>, Vector<double>>.SelectFeatures(
            matrix, selectedIndices);

        Assert.Equal(rows, result.Rows);
        Assert.Equal(5, result.Columns);

        // Verify first row values
        Assert.Equal(0, result[0, 0]);   // Column 0
        Assert.Equal(25, result[0, 1]);  // Column 25
        Assert.Equal(50, result[0, 2]);  // Column 50
        Assert.Equal(75, result[0, 3]);  // Column 75
        Assert.Equal(99, result[0, 4]);  // Column 99
    }

    #endregion
}
