using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Tests that verify DeepCopy of AiModelResult produces a truly independent copy.
/// Before these tests, ZERO tests verified that DeepCopy preserves prediction behavior
/// or that mutations to the copy don't affect the original.
/// </summary>
public class AiModelResultDeepCopyTests
{
    [Fact]
    public void DeepCopy_PredictionsMatchOriginal()
    {
        // Arrange: Train model, deep copy, verify both produce identical predictions
        var (x, y) = CreateLinearDataset(samples: 60, features: 4, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var testData = CreateTestData(rows: 5, cols: 4, seed: 100);
        var originalPredictions = result.Predict(testData);

        // Act: DeepCopy returns IFullModel, cast to AiModelResult
        var copy = result.DeepCopy();
        Assert.IsType<AiModelResult<double, Matrix<double>, Vector<double>>>(copy);
        var copyResult = (AiModelResult<double, Matrix<double>, Vector<double>>)copy;
        var copyPredictions = copyResult.Predict(testData);

        // Assert: Predictions must be identical
        Assert.Equal(originalPredictions.Length, copyPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], copyPredictions[i], precision: 10);
        }
    }

    [Fact]
    public void DeepCopy_MutatingCopyDoesNotAffectOriginal()
    {
        // Arrange: Train model, deep copy
        var (x, y) = CreateLinearDataset(samples: 60, features: 3, seed: 77);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var testData = CreateTestData(rows: 3, cols: 3, seed: 200);
        var originalPredictions = result.Predict(testData);

        // Act: DeepCopy and mutate the copy's optimization result
        var copy = (AiModelResult<double, Matrix<double>, Vector<double>>)result.DeepCopy();

        // Mutate the copy's SelectedFeatureIndices
        Assert.NotNull(copy.OptimizationResult);
        var copyIndices = copy.OptimizationResult.SelectedFeatureIndices;
        Assert.NotNull(copyIndices);
        Assert.NotEmpty(copyIndices);
        copyIndices.Clear();
        copyIndices.Add(0); // force to only use feature 0

        // Assert: Original's predictions should be unchanged after mutating copy
        var originalPredictionsAfterMutation = result.Predict(testData);
        Assert.Equal(originalPredictions.Length, originalPredictionsAfterMutation.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], originalPredictionsAfterMutation[i], precision: 10);
        }
    }

    [Fact]
    public void DeepCopy_PreservesPreprocessingInfo()
    {
        // Arrange: Train with preprocessing, deep copy, verify preprocessing works on copy
        var (x, y) = CreateLinearDataset(samples: 60, features: 3, seed: 55);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigurePreprocessing(new StandardScaler<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        Assert.NotNull(result.PreprocessingInfo);
        Assert.True(result.PreprocessingInfo.IsFitted);

        var testData = CreateTestData(rows: 3, cols: 3, seed: 300);
        var originalPredictions = result.Predict(testData);

        // Act: DeepCopy
        var copy = (AiModelResult<double, Matrix<double>, Vector<double>>)result.DeepCopy();

        // Assert: Copy preserves preprocessing
        Assert.NotNull(copy.PreprocessingInfo);
        Assert.True(copy.PreprocessingInfo.IsFitted,
            "DeepCopy should preserve PreprocessingInfo.IsFitted");

        // Predictions through copy with preprocessing should match original
        var copyPredictions = copy.Predict(testData);
        Assert.Equal(originalPredictions.Length, copyPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], copyPredictions[i], precision: 8);
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
