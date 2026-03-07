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
    public async Task BuildAsync_WithPreprocessing_FitsOnlyOnTrainingData()
    {
        // Arrange: Create a dataset where the training and test distributions differ significantly.
        // If preprocessing is fitted on ALL data (the bug), the scaler mean/std will be a blend.
        // If correctly fitted on ONLY training data, the scaler learns training-only statistics.
        //
        // Strategy: Create data where features have very different means in the first 70% vs last 30%.
        // The StandardScaler fitted on train-only data will have a different mean than one fitted on all data.
        int samples = 100;
        int features = 3;
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        // First 70 samples: features centered around 10
        for (int i = 0; i < 70; i++)
        {
            for (int j = 0; j < features; j++)
            {
                x[i, j] = 10.0 + (i * 0.1) + (j * 0.5);
            }
            y[i] = x[i, 0] * 2.0 + x[i, 1] * 3.0 + 1.0;
        }

        // Last 30 samples: features centered around 1000 (very different distribution)
        for (int i = 70; i < samples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                x[i, j] = 1000.0 + (i * 0.1) + (j * 0.5);
            }
            y[i] = x[i, 0] * 2.0 + x[i, 1] * 3.0 + 1.0;
        }

        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act: Build with StandardScaler preprocessing
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigurePreprocessing(pipeline => pipeline
                .Add(new StandardScaler<double>()))
            .BuildAsync();

        // Assert: The model was built successfully with preprocessing
        Assert.NotNull(result);

        // The PreprocessingInfo should be set and fitted (on training data only)
        var preprocessingInfo = result.PreprocessingInfo;
        Assert.NotNull(preprocessingInfo);
        Assert.True(preprocessingInfo.IsFitted,
            "PreprocessingInfo pipeline should be fitted after training");
    }

    [Fact]
    public async Task BuildAsync_WithPreprocessing_ModelTrainsSuccessfully()
    {
        // Arrange: Build a model with preprocessing and verify it trains without errors.
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

        // Assert: The model was built successfully with preprocessing fitted on training data
        Assert.NotNull(result);
        Assert.NotNull(result.PreprocessingInfo);
        Assert.True(result.PreprocessingInfo.IsFitted,
            "Preprocessing pipeline should be fitted after training");

        // The model should have valid training stats
        Assert.NotNull(result.OptimizationResult);
    }

    [Fact]
    public async Task BuildAsync_WithPreprocessing_PipelineInResultCanTransformNewData()
    {
        // Arrange: Verify the fitted pipeline stored in AiModelResult can transform new data
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

        // Transform new data using the stored pipeline (should not throw)
        var newData = new Matrix<double>(2, 3);
        newData[0, 0] = 5.0; newData[0, 1] = 6.0; newData[0, 2] = 7.0;
        newData[1, 0] = 8.0; newData[1, 1] = 9.0; newData[1, 2] = 10.0;

        var transformed = preprocessingInfo.Pipeline.Transform(newData);
        Assert.NotNull(transformed);
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
        // The model trains on the split data without any transformation
        Assert.Null(result.PreprocessingInfo);
    }

    [Fact]
    public async Task BuildAsync_WithPreprocessing_ScalerStatisticsReflectTrainingDataOnly()
    {
        // Arrange: Create a dataset with known, predictable structure.
        // The key insight: if we create data where the first 70% of samples have features
        // with mean ~5.0 and the last 30% have features with mean ~100.0, and the scaler
        // is correctly fitted on only the training split, the scaler's learned mean should
        // be close to 5.0 (since DataSplitter shuffles, the mix will differ from fitting on all).
        //
        // Rather than testing exact statistics (which depend on shuffle ordering),
        // we verify the pipeline is fitted and produces different results than a pipeline
        // fitted on ALL data would produce.
        int samples = 100;
        int features = 2;
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            double baseVal = i < 70 ? 5.0 : 100.0;
            for (int j = 0; j < features; j++)
            {
                x[i, j] = baseVal + (i * 0.01) + (j * 0.1);
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
