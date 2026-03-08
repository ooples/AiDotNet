using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests verifying that the preprocessing pipeline is fitted ONLY on training data
/// (not on the full dataset before splitting), preventing data leakage.
///
/// Data leakage occurs when the preprocessing pipeline (e.g., StandardScaler) is fitted on the
/// entire dataset including test/validation data. This causes the scaler to learn statistics
/// (mean, std dev) from test data, leaking information from the test set into training.
/// The fix ensures: split first, then FitTransform only on training data, then Transform on val/test.
/// </summary>
public class DataLeakagePreventionIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Data Leakage Prevention Tests

    [Fact]
    public async Task BuildAsync_WithPreprocessing_ScalerStatisticsReflectTrainingDataOnly()
    {
        // Arrange: Create a dataset where each sample has a distinct value to guarantee
        // that any 70/30 split produces different scaler statistics than all data.
        // Using widely spaced unique values ensures mean/std differ significantly.
        int samples = 100;
        int features = 2;
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        // Assign each row a unique value spread across [0, 1000] with large gaps.
        // The 30% holdout always removes enough mass to shift the training mean/std.
        for (int i = 0; i < samples; i++)
        {
            double val = i * 10.0; // values: 0, 10, 20, ..., 990
            for (int j = 0; j < features; j++)
            {
                x[i, j] = val + (j * 0.1);
            }
            y[i] = x[i, 0] + x[i, 1];
        }

        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act: Build with preprocessing (scaler will be fitted on training split only)
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigurePreprocessing(pipeline => pipeline
                .Add(new StandardScaler<double>()))
            .BuildAsync();

        // Also manually fit a scaler on ALL data to compare
        var allDataScaler = new StandardScaler<double>();
        allDataScaler.FitTransform(x);

        // The pipeline in result should be fitted on training data only
        var trainFittedPipeline = result.PreprocessingInfo?.Pipeline;
        Assert.NotNull(trainFittedPipeline);
        Assert.True(trainFittedPipeline.IsFitted);

        // Transform a test point through both pipelines
        var testPoint = new Matrix<double>(1, features);
        testPoint[0, 0] = 50.0;
        testPoint[0, 1] = 50.0;

        var transformedByTrainPipeline = trainFittedPipeline.Transform(testPoint);
        var transformedByAllDataScaler = allDataScaler.Transform(testPoint);

        // The two transformations should produce DIFFERENT results because they were
        // fitted on different data (training subset vs full dataset).
        // If the bug existed (fitting on all data), they would be identical.
        bool areDifferent = false;
        for (int j = 0; j < features; j++)
        {
            if (Math.Abs(transformedByTrainPipeline[0, j] - transformedByAllDataScaler[0, j]) > Tolerance)
            {
                areDifferent = true;
                break;
            }
        }

        Assert.True(areDifferent,
            "Scaler fitted on training split should produce different results than scaler fitted on all data, " +
            "confirming preprocessing is NOT fitted on the full dataset (data leakage prevention).");
    }

    [Fact]
    public async Task BuildAsync_WithPreprocessing_TransformProducesCorrectShape()
    {
        // Arrange: Build a model with preprocessing and verify the fitted pipeline
        // can transform new data and produces output with the correct dimensions.
        var (x, y) = CreateLinearDataset(samples: 50, features: 3, seed: 77);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigurePreprocessing(pipeline => pipeline
                .Add(new StandardScaler<double>()))
            .BuildAsync();

        // Assert: The pipeline stored in PreprocessingInfo is fitted and can transform data
        var preprocessingInfo = result.PreprocessingInfo;
        Assert.NotNull(preprocessingInfo);
        Assert.NotNull(preprocessingInfo.Pipeline);
        Assert.True(preprocessingInfo.Pipeline.IsFitted,
            "The pipeline should be fitted and ready to transform new data");

        // Transform new data using the stored pipeline
        var newData = new Matrix<double>(2, 3);
        newData[0, 0] = 5.0; newData[0, 1] = 6.0; newData[0, 2] = 7.0;
        newData[1, 0] = 8.0; newData[1, 1] = 9.0; newData[1, 2] = 10.0;

        var transformed = preprocessingInfo.Pipeline.Transform(newData);
        Assert.NotNull(transformed);

        // Verify output shape matches input shape
        Assert.Equal(2, transformed.Rows);
        Assert.Equal(3, transformed.Columns);

        // Verify values are actually transformed (not passthrough)
        // StandardScaler centers data, so at least some values should differ from input
        bool anyDifferent = false;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (Math.Abs(transformed[i, j] - newData[i, j]) > Tolerance)
                {
                    anyDifferent = true;
                    break;
                }
            }
            if (anyDifferent) break;
        }

        Assert.True(anyDifferent,
            "StandardScaler should transform values (center and scale), not return input unchanged");
    }

    [Fact]
    public async Task BuildAsync_WithPreprocessing_PipelineIsFittedAndStored()
    {
        // Arrange: Verify the preprocessing pipeline is fitted during build and stored in the result.
        // This ensures the pipeline can be used later for transforming new data at inference time.
        var (x, y) = CreateLinearDataset(samples: 60, features: 4, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigurePreprocessing(pipeline => pipeline
                .Add(new StandardScaler<double>()))
            .BuildAsync();

        // Assert: preprocessing pipeline is fitted and stored
        Assert.NotNull(result);
        Assert.NotNull(result.PreprocessingInfo);
        Assert.True(result.PreprocessingInfo.IsFitted,
            "Preprocessing pipeline should be fitted after build");
        Assert.NotNull(result.OptimizationResult);

        // Assert: preprocessing can transform new data without error
        var newData = new Matrix<double>(3, 4);
        var rng = new Random(99);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                newData[i, j] = rng.NextDouble() * 10;

        var transformed = result.PreprocessingInfo.TransformFeatures(newData);
        Assert.NotNull(transformed);
        Assert.Equal(3, transformed.Rows);
        Assert.Equal(4, transformed.Columns);
    }

    [Fact]
    public async Task BuildAsync_WithoutPreprocessing_StillWorksCorrectly()
    {
        // Arrange: Ensure the fix doesn't break the path where no preprocessing is configured
        var (x, y) = CreateLinearDataset(samples: 50, features: 3, seed: 55);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act: Build WITHOUT preprocessing
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        // Assert: Model should build without preprocessing
        Assert.NotNull(result);
        Assert.NotNull(result.OptimizationResult);

        // PreprocessingInfo should be null (no preprocessing configured)
        Assert.Null(result.PreprocessingInfo);

        // Model should have been trained successfully
        Assert.NotNull(result.Model);
    }

    [Fact]
    public async Task BuildAsync_WithPreprocessing_TransformIsNotIdentity()
    {
        // Arrange: Create data with known non-zero mean and non-unit variance.
        // After StandardScaler, the training data should be centered (mean ~0) and scaled (std ~1).
        // If the pipeline wasn't fitted at all, transform would be identity.
        int samples = 80;
        int features = 2;
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        var rng = new Random(123);
        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = 100.0 + rng.NextDouble() * 10.0; // mean ~105
            x[i, 1] = 500.0 + rng.NextDouble() * 50.0; // mean ~525
            y[i] = x[i, 0] * 2.0 + x[i, 1] * 0.5;
        }

        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigurePreprocessing(pipeline => pipeline
                .Add(new StandardScaler<double>()))
            .BuildAsync();

        var pipeline = result.PreprocessingInfo?.Pipeline;
        Assert.NotNull(pipeline);

        // Transform the mean point — after StandardScaler, the mean of training data
        // maps to approximately 0. A point at ~105, ~525 should be near the mean.
        var meanPoint = new Matrix<double>(1, features);
        meanPoint[0, 0] = 105.0;
        meanPoint[0, 1] = 525.0;

        var transformed = pipeline.Transform(meanPoint);

        // The transformed values should be close to 0 (since input is near training mean)
        // and definitely not 105/525 (which would indicate identity/unfitted transform)
        Assert.True(Math.Abs(transformed[0, 0]) < 10.0,
            $"Transformed value {transformed[0, 0]} should be centered near 0, not raw value ~105");
        Assert.True(Math.Abs(transformed[0, 1]) < 10.0,
            $"Transformed value {transformed[0, 1]} should be centered near 0, not raw value ~525");
    }

    #endregion

    #region Helper Methods

    private static (Matrix<double> x, Vector<double> y) CreateLinearDataset(
        int samples, int features, int seed)
    {
        var random = new Random(seed);
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        var trueCoefficients = new double[features];
        for (int j = 0; j < features; j++)
        {
            trueCoefficients[j] = random.NextDouble() * 4 - 2;
        }

        for (int i = 0; i < samples; i++)
        {
            double target = 0;
            for (int j = 0; j < features; j++)
            {
                x[i, j] = random.NextDouble() * 10;
                target += trueCoefficients[j] * x[i, j];
            }
            y[i] = target + random.NextDouble() * 0.1;
        }

        return (x, y);
    }

    #endregion
}
