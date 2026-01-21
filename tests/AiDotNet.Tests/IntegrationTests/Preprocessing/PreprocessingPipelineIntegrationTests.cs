using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Integration tests for PreprocessingPipeline.
/// Tests sequential transformations, fit/transform/fit_transform, and inverse_transform.
/// </summary>
public class PreprocessingPipelineIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(double[,] data)
    {
        return new Matrix<double>(data);
    }

    private static void AssertMatrixEqual(Matrix<double> expected, Matrix<double> actual, double tolerance = Tolerance)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);

        for (int i = 0; i < expected.Rows; i++)
        {
            for (int j = 0; j < expected.Columns; j++)
            {
                Assert.True(
                    Math.Abs(expected[i, j] - actual[i, j]) < tolerance,
                    $"Mismatch at [{i},{j}]: expected {expected[i, j]}, actual {actual[i, j]}");
            }
        }
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_CreatesEmptyPipeline()
    {
        // Act
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

        // Assert
        Assert.False(pipeline.IsFitted);
        Assert.Equal(0, pipeline.Count);
        Assert.Empty(pipeline.Steps);
    }

    #endregion

    #region Add Step Tests

    [Fact]
    public void Add_WithoutName_AddsStepWithAutomaticName()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler = new StandardScaler<double>();

        // Act
        pipeline.Add(scaler);

        // Assert
        Assert.Equal(1, pipeline.Count);
        Assert.Equal("step_0", pipeline.Steps[0].Name);
    }

    [Fact]
    public void Add_WithName_AddsStepWithSpecifiedName()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler = new StandardScaler<double>();

        // Act
        pipeline.Add("my_scaler", scaler);

        // Assert
        Assert.Equal(1, pipeline.Count);
        Assert.Equal("my_scaler", pipeline.Steps[0].Name);
    }

    [Fact]
    public void Add_DuplicateName_ThrowsArgumentException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler1 = new StandardScaler<double>();
        var scaler2 = new MinMaxScaler<double>();
        pipeline.Add("scaler", scaler1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => pipeline.Add("scaler", scaler2));
    }

    [Fact]
    public void Add_NullTransformer_ThrowsArgumentNullException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => pipeline.Add("step", null!));
    }

    [Fact]
    public void Add_EmptyName_ThrowsArgumentException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => pipeline.Add("", scaler));
    }

    [Fact]
    public void Add_WhitespaceName_ThrowsArgumentException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => pipeline.Add("   ", scaler));
    }

    [Fact]
    public void Add_ReturnsThisForChaining()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler = new StandardScaler<double>();

        // Act
        var result = pipeline.Add(scaler);

        // Assert
        Assert.Same(pipeline, result);
    }

    #endregion

    #region GetStep Tests

    [Fact]
    public void GetStep_ExistingName_ReturnsTransformer()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler = new StandardScaler<double>();
        pipeline.Add("my_scaler", scaler);

        // Act
        var retrieved = pipeline.GetStep("my_scaler");

        // Assert
        Assert.Same(scaler, retrieved);
    }

    [Fact]
    public void GetStep_NonExistingName_ReturnsNull()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler = new StandardScaler<double>();
        pipeline.Add("my_scaler", scaler);

        // Act
        var retrieved = pipeline.GetStep("nonexistent");

        // Assert
        Assert.Null(retrieved);
    }

    #endregion

    #region Fit Tests

    [Fact]
    public void Fit_WithValidData_SetsFittedToTrue()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        pipeline.Fit(data);

        // Assert
        Assert.True(pipeline.IsFitted);
    }

    [Fact]
    public void Fit_NullData_ThrowsArgumentNullException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => pipeline.Fit(null!));
    }

    [Fact]
    public void Fit_FitsAllTransformersInSequence()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler1 = new StandardScaler<double>();
        var scaler2 = new MinMaxScaler<double>();
        pipeline.Add(scaler1);
        pipeline.Add(scaler2);
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        pipeline.Fit(data);

        // Assert
        Assert.True(scaler1.IsFitted);
        Assert.True(scaler2.IsFitted);
    }

    #endregion

    #region Transform Tests

    [Fact]
    public void Transform_WithoutFit_ThrowsInvalidOperationException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => pipeline.Transform(data));
    }

    [Fact]
    public void Transform_NullData_ThrowsArgumentNullException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });
        pipeline.Fit(data);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => pipeline.Transform(null!));
    }

    [Fact]
    public void Transform_SingleScaler_TransformsCorrectly()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        // Data with known mean and std
        // Col 0: [1, 3, 5] -> mean=3, std=sqrt(8/3)
        // Col 1: [2, 4, 6] -> mean=4, std=sqrt(8/3)
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert - Check that mean is ~0 for each column
        double col0Mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        double col1Mean = (result[0, 1] + result[1, 1] + result[2, 1]) / 3;

        Assert.True(Math.Abs(col0Mean) < 1e-10, $"Column 0 mean should be ~0, got {col0Mean}");
        Assert.True(Math.Abs(col1Mean) < 1e-10, $"Column 1 mean should be ~0, got {col1Mean}");
    }

    [Fact]
    public void Transform_MultipleScalers_AppliesInSequence()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        pipeline.Add(new MinMaxScaler<double>());

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert - After StandardScaler then MinMaxScaler, values should be in [0, 1]
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.True(result[i, j] >= -1e-10 && result[i, j] <= 1.0 + 1e-10,
                    $"Value at [{i},{j}] = {result[i, j]} should be in [0, 1]");
            }
        }
    }

    #endregion

    #region FitTransform Tests

    [Fact]
    public void FitTransform_EquivalentToFitThenTransform()
    {
        // Arrange
        var pipeline1 = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline1.Add(new StandardScaler<double>());

        var pipeline2 = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline2.Add(new StandardScaler<double>());

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result1 = pipeline1.FitTransform(data);

        pipeline2.Fit(data);
        var result2 = pipeline2.Transform(data);

        // Assert
        AssertMatrixEqual(result1, result2);
    }

    [Fact]
    public void FitTransform_NullData_ThrowsArgumentNullException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => pipeline.FitTransform(null!));
    }

    #endregion

    #region InverseTransform Tests

    [Fact]
    public void InverseTransform_WithSupportingTransformers_ReversesTransformation()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var transformed = pipeline.FitTransform(data);
        var inversed = pipeline.InverseTransform(transformed);

        // Assert
        AssertMatrixEqual(data, inversed, 1e-9);
    }

    [Fact]
    public void InverseTransform_MultipleScalers_ReversesInCorrectOrder()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        pipeline.Add(new MinMaxScaler<double>());

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var transformed = pipeline.FitTransform(data);
        var inversed = pipeline.InverseTransform(transformed);

        // Assert
        AssertMatrixEqual(data, inversed, 1e-9);
    }

    [Fact]
    public void InverseTransform_WithoutFit_ThrowsInvalidOperationException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => pipeline.InverseTransform(data));
    }

    [Fact]
    public void SupportsInverseTransform_AllSupportingTransformers_ReturnsTrue()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        pipeline.Add(new MinMaxScaler<double>());

        // Assert
        Assert.True(pipeline.SupportsInverseTransform);
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void Clone_CreatesNewPipelineWithSameStructure()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add("scaler1", new StandardScaler<double>());
        pipeline.Add("scaler2", new MinMaxScaler<double>());

        // Act
        var clone = pipeline.Clone();

        // Assert
        Assert.Equal(pipeline.Count, clone.Count);
        Assert.False(clone.IsFitted); // Clone should not be fitted
    }

    [Fact]
    public void Clone_DoesNotShareFittedState()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });
        pipeline.Fit(data);

        // Act
        var clone = pipeline.Clone();

        // Assert
        Assert.True(pipeline.IsFitted);
        Assert.False(clone.IsFitted);
    }

    #endregion

    #region GetFeatureNamesOut Tests

    [Fact]
    public void GetFeatureNamesOut_PassesThroughInputNames()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        var inputNames = new[] { "feature_a", "feature_b" };

        // Act
        var outputNames = pipeline.GetFeatureNamesOut(inputNames);

        // Assert
        Assert.Equal(inputNames, outputNames);
    }

    [Fact]
    public void GetFeatureNamesOut_NoInputNames_ReturnsEmptyArray()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        // Act
        var outputNames = pipeline.GetFeatureNamesOut(null);

        // Assert
        Assert.Empty(outputNames);
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void FitTransform_SingleRow_WorksCorrectly()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new MinMaxScaler<double>()); // StandardScaler may have issues with single row (std=0)

        var data = CreateTestMatrix(new double[,] { { 1, 2, 3 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert - Single row should map to middle of range
        Assert.Equal(1, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void FitTransform_SingleColumn_WorksCorrectly()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        var data = CreateTestMatrix(new double[,] { { 1 }, { 3 }, { 5 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);

        // Check mean is ~0
        double mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(mean) < 1e-10);
    }

    [Fact]
    public void FitTransform_ConstantColumn_HandlesGracefully()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        // All values in column 0 are the same
        var data = CreateTestMatrix(new double[,] { { 5, 1 }, { 5, 2 }, { 5, 3 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert - Constant column should not cause NaN or infinity
        for (int i = 0; i < result.Rows; i++)
        {
            Assert.False(double.IsNaN(result[i, 0]), $"NaN found at [{i},0]");
            Assert.False(double.IsInfinity(result[i, 0]), $"Infinity found at [{i},0]");
        }
    }

    [Fact]
    public void FitTransform_EmptyPipeline_PassesThroughData()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert
        AssertMatrixEqual(data, result);
    }

    [Fact]
    public void FitTransform_LargeMatrix_WorksCorrectly()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        // Create a 1000x100 matrix
        var data = new double[1000, 100];
        var random = new Random(42);
        for (int i = 0; i < 1000; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                data[i, j] = random.NextDouble() * 100;
            }
        }
        var matrix = CreateTestMatrix(data);

        // Act
        var result = pipeline.FitTransform(matrix);

        // Assert - Check dimensions and no NaN values
        Assert.Equal(1000, result.Rows);
        Assert.Equal(100, result.Columns);

        for (int i = 0; i < 10; i++) // Check first 10 rows
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.False(double.IsNaN(result[i, j]), $"NaN found at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void Transform_NewData_UseFittedParameters()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        var trainData = CreateTestMatrix(new double[,] { { 0, 0 }, { 10, 10 } }); // mean=5, range=10
        var testData = CreateTestMatrix(new double[,] { { 5, 5 } }); // Should map to 0 (mean)

        // Act
        pipeline.Fit(trainData);
        var result = pipeline.Transform(testData);

        // Assert - Value of 5 (the mean) should transform to 0
        Assert.True(Math.Abs(result[0, 0]) < 1e-10, $"Expected ~0, got {result[0, 0]}");
        Assert.True(Math.Abs(result[0, 1]) < 1e-10, $"Expected ~0, got {result[0, 1]}");
    }

    #endregion

    #region Data Leakage Tests

    [Fact]
    public void Transform_UsesOnlyFittedParameters_NoDataLeakage()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new MinMaxScaler<double>());

        var trainData = CreateTestMatrix(new double[,] { { 0, 0 }, { 10, 10 } }); // min=0, max=10
        var testData = CreateTestMatrix(new double[,] { { 20, 20 } }); // Outside training range

        // Act
        pipeline.Fit(trainData);
        var result = pipeline.Transform(testData);

        // Assert - Values outside training range should extrapolate beyond [0, 1]
        Assert.True(result[0, 0] > 1.0, $"Expected value > 1 for extrapolation, got {result[0, 0]}");
        Assert.True(result[0, 1] > 1.0, $"Expected value > 1 for extrapolation, got {result[0, 1]}");
    }

    #endregion
}
