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

    #region Nested Pipeline Tests - Pipeline with ColumnTransformer

    [Fact]
    public void Pipeline_WithColumnTransformer_TransformsCorrectly()
    {
        // Arrange - Pipeline containing a ColumnTransformer
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

        var columnTransformer = new ColumnTransformer<double>();
        columnTransformer.Add("scale_first", new StandardScaler<double>(), new[] { 0 });
        columnTransformer.Add("scale_second", new MinMaxScaler<double>(), new[] { 1 });

        pipeline.Add("column_transformer", columnTransformer);

        var data = CreateTestMatrix(new double[,] { { 1, 0 }, { 2, 5 }, { 3, 10 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert - Should have 2 columns
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Column 0 should be standardized (mean ~0)
        double col0Mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(col0Mean) < Tolerance, "Column 0 should be centered");

        // Column 1 should be min-max scaled to [0, 1]
        Assert.True(Math.Abs(result[0, 1] - 0) < Tolerance, "Min value should be 0");
        Assert.True(Math.Abs(result[2, 1] - 1) < Tolerance, "Max value should be 1");
    }

    [Fact]
    public void Pipeline_WithColumnTransformer_DoesNotSupportInverseTransform()
    {
        // Arrange - ColumnTransformer does not support inverse transform
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

        var columnTransformer = new ColumnTransformer<double>();
        columnTransformer.Add("scaler", new StandardScaler<double>(), new[] { 0, 1 });

        pipeline.Add(columnTransformer);

        var data = CreateTestMatrix(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 } });

        // Act
        var transformed = pipeline.FitTransform(data);

        // Assert - ColumnTransformer doesn't support inverse transform
        Assert.False(pipeline.SupportsInverseTransform);
        Assert.Throws<NotSupportedException>(() => pipeline.InverseTransform(transformed));
    }

    #endregion

    #region Nested Pipeline Tests - Pipeline with FeatureUnion

    [Fact]
    public void Pipeline_WithFeatureUnion_ConcatenatesFeatures()
    {
        // Arrange - Pipeline containing a FeatureUnion
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

        var featureUnion = new FeatureUnion<double>();
        featureUnion.Add("standard", new StandardScaler<double>());
        featureUnion.Add("minmax", new MinMaxScaler<double>());

        pipeline.Add("feature_union", featureUnion);

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert - Should have 4 columns (2 from StandardScaler + 2 from MinMaxScaler)
        Assert.Equal(3, result.Rows);
        Assert.Equal(4, result.Columns);
    }

    [Fact]
    public void Pipeline_WithFeatureUnion_InverseTransformNotSupported()
    {
        // Arrange - FeatureUnion doesn't support inverse transform
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

        var featureUnion = new FeatureUnion<double>();
        featureUnion.Add("standard", new StandardScaler<double>());

        pipeline.Add(featureUnion);

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var transformed = pipeline.FitTransform(data);

        // Assert - Should not support inverse transform
        Assert.False(pipeline.SupportsInverseTransform);
    }

    #endregion

    #region Complex Nested Pipeline Tests

    [Fact]
    public void Pipeline_MultipleStepsWithColumnTransformer_TransformsSequentially()
    {
        // Arrange - Pipeline with multiple steps including ColumnTransformer
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

        // First step: StandardScaler on all columns
        pipeline.Add("global_scaler", new StandardScaler<double>());

        // Second step: ColumnTransformer for ALL columns to preserve output width
        var columnTransformer = new ColumnTransformer<double>();
        columnTransformer.Add("rescale", new MinMaxScaler<double>(), new[] { 0, 1 }); // Both columns

        pipeline.Add("column_transformer", columnTransformer);

        var data = CreateTestMatrix(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 } });

        // Act
        var result = pipeline.FitTransform(data);

        // Assert - Should process both steps sequentially
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void Pipeline_NestedPipelineInPipeline_TransformsCorrectly()
    {
        // Arrange - A pipeline containing another pipeline
        var innerPipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        innerPipeline.Add("inner_scaler", new StandardScaler<double>());

        var outerPipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        outerPipeline.Add("inner_pipeline", innerPipeline);
        outerPipeline.Add("outer_scaler", new MinMaxScaler<double>());

        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        // Act
        var result = outerPipeline.FitTransform(data);

        // Assert - Should process inner pipeline first, then outer scaler
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // After StandardScaler + MinMaxScaler, values should be in [0, 1]
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.True(result[i, j] >= -Tolerance && result[i, j] <= 1 + Tolerance,
                    $"Value at [{i},{j}] = {result[i, j]} should be in [0, 1]");
            }
        }
    }

    #endregion

    #region PreprocessingInfo Tests

    [Fact]
    public void PreprocessingInfo_DefaultConstructor_CreatesUnfittedInstance()
    {
        // Act
        var info = new PreprocessingInfo<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.Null(info.Pipeline);
        Assert.Null(info.TargetPipeline);
        Assert.False(info.IsFitted);
        Assert.False(info.IsTargetFitted);
    }

    [Fact]
    public void PreprocessingInfo_WithPipeline_StoresPipelineCorrectly()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        pipeline.Fit(data);

        // Act
        var info = new PreprocessingInfo<double, Matrix<double>, Vector<double>>(pipeline);

        // Assert
        Assert.NotNull(info.Pipeline);
        Assert.True(info.IsFitted);
        Assert.Null(info.TargetPipeline);
        Assert.False(info.IsTargetFitted);
    }

    [Fact]
    public void PreprocessingInfo_TransformFeatures_TransformsCorrectly()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());

        var trainData = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        pipeline.Fit(trainData);

        var info = new PreprocessingInfo<double, Matrix<double>, Vector<double>>(pipeline);

        var testData = CreateTestMatrix(new double[,] { { 2 } }); // Mean value

        // Act
        var result = info.TransformFeatures(testData);

        // Assert - Mean value (2) should transform to ~0
        Assert.True(Math.Abs(result[0, 0]) < Tolerance);
    }

    [Fact]
    public void PreprocessingInfo_TransformFeaturesWithoutFit_ThrowsInvalidOperationException()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        // Note: Not fitted

        var info = new PreprocessingInfo<double, Matrix<double>, Vector<double>>(pipeline);
        var data = CreateTestMatrix(new double[,] { { 1 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => info.TransformFeatures(data));
    }

    [Fact]
    public void PreprocessingInfo_TransformFeaturesWithNullPipeline_ThrowsInvalidOperationException()
    {
        // Arrange
        var info = new PreprocessingInfo<double, Matrix<double>, Vector<double>>();
        var data = CreateTestMatrix(new double[,] { { 1 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => info.TransformFeatures(data));
    }

    [Fact]
    public void PreprocessingInfo_InverseTransformPredictions_WithNoTargetPipeline_ReturnsInput()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add(new StandardScaler<double>());
        var trainData = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        pipeline.Fit(trainData);

        var info = new PreprocessingInfo<double, Matrix<double>, Vector<double>>(pipeline);
        var predictions = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        // Act - No target pipeline, should return input unchanged
        var result = info.InverseTransformPredictions(predictions);

        // Assert
        Assert.Equal(predictions.Length, result.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.Equal(predictions[i], result[i]);
        }
    }

    #endregion

    #region Serialization Tests (Clone Functionality)

    [Fact]
    public void Clone_IsShallowClone_SharesTransformerInstances()
    {
        // Arrange
        var original = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        var scaler = new StandardScaler<double>();
        original.Add("scaler", scaler);

        // Act
        var clone = original.Clone();

        // Assert - Clone is shallow, so transformers are the same instance
        // (Note: Clone creates a shallow copy as documented)
        Assert.Equal(original.Count, clone.Count);
        Assert.False(clone.IsFitted); // Clone is not fitted
    }

    [Fact]
    public void Clone_ProducesSameResultsWhenFittedOnSameData()
    {
        // Arrange
        var original = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        original.Add("scaler1", new StandardScaler<double>());
        original.Add("scaler2", new MinMaxScaler<double>());

        var data = CreateTestMatrix(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 } });

        // Act
        original.Fit(data);
        var clone = original.Clone();
        clone.Fit(data); // Fit on same data

        var originalResult = original.Transform(data);
        var cloneResult = clone.Transform(data);

        // Assert - Should produce identical results
        AssertMatrixEqual(originalResult, cloneResult);
    }

    [Fact]
    public void Clone_WithMultipleTransformerTypes_PreservesStructure()
    {
        // Arrange
        var original = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        original.Add("standard", new StandardScaler<double>());
        original.Add("minmax", new MinMaxScaler<double>());

        // Act
        var clone = original.Clone();

        // Assert - Structure is preserved
        Assert.Equal(original.Count, clone.Count);
        Assert.Equal(original.Steps[0].Name, clone.Steps[0].Name);
        Assert.Equal(original.Steps[1].Name, clone.Steps[1].Name);
        // Note: Clone is shallow, so transformer instances are the same
        Assert.Same(original.GetStep("standard"), clone.GetStep("standard"));
        Assert.Same(original.GetStep("minmax"), clone.GetStep("minmax"));
    }

    [Fact]
    public void Clone_PreservesSupportsInverseTransformFlag()
    {
        // Arrange
        var pipeline1 = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline1.Add(new StandardScaler<double>()); // Supports inverse

        var pipeline2 = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline2.Add(new Normalizer<double>()); // Does not support inverse

        // Act
        var clone1 = pipeline1.Clone();
        var clone2 = pipeline2.Clone();

        // Assert
        Assert.True(clone1.SupportsInverseTransform);
        Assert.False(clone2.SupportsInverseTransform);
    }

    [Fact]
    public void Clone_OriginalModificationDoesNotAffectClone()
    {
        // Arrange
        var original = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        original.Add("scaler1", new StandardScaler<double>());

        var clone = original.Clone();

        // Act - Modify original after cloning
        original.Add("scaler2", new MinMaxScaler<double>());

        // Assert - Clone should not be affected
        Assert.Equal(2, original.Count);
        Assert.Equal(1, clone.Count);
    }

    #endregion

    #region TransformerBase Tests (via concrete implementations)

    [Fact]
    public void TransformerBase_Fit_SetsIsFittedToTrue()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        // Act
        scaler.Fit(data);

        // Assert
        Assert.True(scaler.IsFitted);
    }

    [Fact]
    public void TransformerBase_FitTransform_SetsIsFittedToTrue()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        // Act
        scaler.FitTransform(data);

        // Assert
        Assert.True(scaler.IsFitted);
    }

    [Fact]
    public void TransformerBase_TransformBeforeFit_ThrowsInvalidOperationException()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => scaler.Transform(data));
    }

    [Fact]
    public void TransformerBase_InverseTransformBeforeFit_ThrowsInvalidOperationException()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => scaler.InverseTransform(data));
    }

    [Fact]
    public void TransformerBase_ColumnIndices_LimitsTransformation()
    {
        // Arrange - Only transform column 0
        var scaler = new StandardScaler<double>(columnIndices: new[] { 0 });
        var data = CreateTestMatrix(new double[,] { { 1, 100 }, { 2, 200 }, { 3, 300 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Column 0 should be transformed, column 1 should be unchanged
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);

        // Column 0 should be centered
        double col0Mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(col0Mean) < Tolerance);

        // Column 1 should be unchanged
        Assert.Equal(100, result[0, 1]);
        Assert.Equal(200, result[1, 1]);
        Assert.Equal(300, result[2, 1]);
    }

    [Fact]
    public void TransformerBase_GetFeatureNamesOut_ReturnsInputNamesWhenProvided()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });
        scaler.Fit(data);

        var inputNames = new[] { "feature_a", "feature_b" };

        // Act
        var outputNames = scaler.GetFeatureNamesOut(inputNames);

        // Assert
        Assert.Equal(2, outputNames.Length);
        Assert.Equal("feature_a", outputNames[0]);
        Assert.Equal("feature_b", outputNames[1]);
    }

    [Fact]
    public void TransformerBase_NullData_ThrowsArgumentNullException()
    {
        // Arrange
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => scaler.Fit(null!));
    }

    #endregion
}
