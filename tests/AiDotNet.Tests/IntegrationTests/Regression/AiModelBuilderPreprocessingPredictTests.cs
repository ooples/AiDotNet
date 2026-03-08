using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Tests that verify preprocessing transforms are actually applied during Predict().
/// Before these tests, ZERO tests verified that ConfigurePreprocessing() had any effect
/// on the prediction path. A model could silently skip normalization and produce garbage.
/// </summary>
public class AiModelBuilderPreprocessingPredictTests
{
    [Fact]
    public void BuildWithPreprocessing_PreprocessingInfoIsFitted()
    {
        // Arrange: Create data with very different feature scales.
        // Feature 1: 0-1 range, Feature 2: 0-10000 range
        var random = new Random(42);
        int samples = 80;
        var x = new Matrix<double>(samples, 2);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = random.NextDouble();             // small scale: 0-1
            x[i, 1] = random.NextDouble() * 10000;     // large scale: 0-10000
            y[i] = 5.0 * x[i, 0] + 0.001 * x[i, 1] + random.NextDouble() * 0.1;
        }

        var loader = DataLoaders.FromMatrixVector(x, y);

        // Train WITH preprocessing (StandardScaler)
        var resultWithPreprocessing = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigurePreprocessing(new StandardScaler<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        // Assert: Verify preprocessing was configured and fitted
        Assert.NotNull(resultWithPreprocessing.PreprocessingInfo);
        Assert.True(resultWithPreprocessing.PreprocessingInfo.IsFitted);

        // Create test data with the same scale characteristics
        var testData = new Matrix<double>(5, 2);
        var testRng = new Random(100);
        for (int i = 0; i < 5; i++)
        {
            testData[i, 0] = testRng.NextDouble();
            testData[i, 1] = testRng.NextDouble() * 10000;
        }

        // Act: Predict through the preprocessing model
        var predictions = resultWithPreprocessing.Predict(testData);

        // Assert: All predictions should be finite and in a reasonable range
        Assert.Equal(5, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction {i} is NaN — preprocessing may not have been applied");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction {i} is Infinity — preprocessing may not have been applied");
            // y = 5*x1 + 0.001*x2, x1 in [0,1], x2 in [0,10000] -> y in [0, 15]
            // Predictions should be in a reasonable range given the data distribution
            Assert.InRange(predictions[i], -50.0, 50.0);
        }
    }

    [Fact]
    public void BuildWithPreprocessing_PredictAppliesInverseTransform()
    {
        // Arrange: Train on known linear data y = 2*x + 3 with preprocessing.
        // Predictions should be in the ORIGINAL scale, not the standardized scale.
        var random = new Random(77);
        int samples = 60;
        var x = new Matrix<double>(samples, 1);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = random.NextDouble() * 20;  // x in [0, 20]
            y[i] = 2.0 * x[i, 0] + 3.0 + random.NextDouble() * 0.5;
        }

        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigurePreprocessing(new StandardScaler<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        // Test: predict for x = 10 → expected y ≈ 2*10 + 3 = 23
        var testData = new Matrix<double>(1, 1);
        testData[0, 0] = 10.0;

        var predictions = result.Predict(testData);
        Assert.Equal(1, predictions.Length);

        // The prediction should be in original scale (around 23), NOT in standardized scale (around 0)
        // Allow generous tolerance since regression has noise and may not perfectly recover coefficients
        Assert.InRange(predictions[0], 10.0, 40.0);
        // A prediction near 0.0 would indicate inverse transform was NOT applied
        Assert.True(Math.Abs(predictions[0]) > 5.0,
            $"Prediction {predictions[0]} is too close to 0 — inverse transform may not have been applied. " +
            "Expected ~23 (original scale), got standardized-scale value.");
    }

    [Fact]
    public void BuildWithCustomPipeline_PredictProducesFiniteOutput()
    {
        // Arrange: Use pipeline with SimpleImputer + StandardScaler.
        // Training data has some NaN values that the imputer must handle.
        var random = new Random(55);
        int samples = 60;
        var x = new Matrix<double>(samples, 3);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = random.NextDouble() * 10;
            x[i, 1] = random.NextDouble() * 10;
            x[i, 2] = random.NextDouble() * 10;
            y[i] = x[i, 0] + 2 * x[i, 1] + 3 * x[i, 2] + random.NextDouble() * 0.5;
        }

        // Inject NaN values into ~10% of training data
        for (int i = 0; i < samples; i++)
        {
            if (random.NextDouble() < 0.1)
                x[i, random.Next(3)] = double.NaN;
        }

        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigurePreprocessing(pipeline => pipeline
                .Add(new SimpleImputer<double>(ImputationStrategy.Mean))
                .Add(new StandardScaler<double>()))
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        // Test: predict with clean data (no NaN)
        var testData = new Matrix<double>(3, 3);
        var testRng = new Random(200);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                testData[i, j] = testRng.NextDouble() * 10;

        var predictions = result.Predict(testData);

        // Assert: All predictions must be finite and in a reasonable range
        // y = x1 + 2*x2 + 3*x3, with xi in [0,10] -> y in [0, 60]
        Assert.Equal(3, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction {i} is NaN — imputer or scaler may not have been applied");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction {i} is Infinity — imputer or scaler may not have been applied");
            Assert.InRange(predictions[i], -100.0, 200.0);
        }
    }

    [Fact]
    public void BuildWithoutPreprocessing_PredictSkipsTransform()
    {
        // Arrange: Train WITHOUT ConfigurePreprocessing
        var (x, y) = CreateLinearDataset(samples: 50, features: 3, seed: 88);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            // Note: NO ConfigurePreprocessing call
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        // Assert: PreprocessingInfo should be null or not fitted
        var preprocessingInfo = result.PreprocessingInfo;
        bool hasPreprocessing = preprocessingInfo?.IsFitted ?? false;
        Assert.False(hasPreprocessing,
            "Without ConfigurePreprocessing(), PreprocessingInfo should not be fitted");

        // Predict should still work (no transform step)
        var testData = CreateTestData(rows: 3, cols: 3, seed: 300);
        var predictions = result.Predict(testData);
        Assert.Equal(3, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]));
        }
    }

    #region Helper Methods

    private static (Matrix<double> x, Vector<double> y) CreateLinearDataset(
        int samples, int features, int seed)
    {
        var random = new Random(seed);
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        var trueCoefficients = new double[features];
        for (int j = 0; j < features; j++)
            trueCoefficients[j] = random.NextDouble() * 4 - 2;

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

    private static Matrix<double> CreateTestData(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = random.NextDouble() * 10;
        return matrix;
    }

    #endregion
}
