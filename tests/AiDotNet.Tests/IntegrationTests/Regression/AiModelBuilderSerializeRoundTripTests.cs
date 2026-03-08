using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Tests that verify Serialize → Deserialize → Predict round-trip correctness.
/// The most critical gap in the test suite: ZERO tests existed for this path.
/// If serialization drops model weights, preprocessing state, or feature selection indices,
/// users save models and silently get wrong predictions when loading them later.
/// </summary>
public class AiModelBuilderSerializeRoundTripTests
{
    [Fact]
    public async Task RidgeRegression_SerializeDeserialize_PredictMatchesOriginal()
    {
        // Arrange: Train a RidgeRegression model through the builder
        var (x, y) = CreateLinearDataset(samples: 60, features: 4, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync();

        var testData = CreateTestData(rows: 5, cols: 4, seed: 100);

        // Get original predictions
        var originalPredictions = result.Predict(testData);
        Assert.NotNull(originalPredictions);
        Assert.Equal(5, originalPredictions.Length);

        // Act: Serialize → Deserialize
        var bytes = result.Serialize();
        Assert.NotNull(bytes);
        Assert.True(bytes.Length > 0, "Serialized bytes should not be empty");

        var restored = new AiModelResult<double, Matrix<double>, Vector<double>>();
        restored.Deserialize(bytes);

        var restoredPredictions = restored.Predict(testData);

        // Assert: Predictions must be identical (same weights + same data = same result)
        Assert.NotNull(restoredPredictions);
        Assert.Equal(originalPredictions.Length, restoredPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], restoredPredictions[i], precision: 10);
        }
    }

    [Fact]
    public async Task MultipleRegression_SerializeDeserialize_PredictMatchesOriginal()
    {
        // Arrange: Different model type to verify serializer generality
        var (x, y) = CreateLinearDataset(samples: 50, features: 3, seed: 77);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new MultipleRegression<double>())
            .BuildAsync();

        var testData = CreateTestData(rows: 4, cols: 3, seed: 200);
        var originalPredictions = result.Predict(testData);

        // Act: Round-trip
        var bytes = result.Serialize();
        var restored = new AiModelResult<double, Matrix<double>, Vector<double>>();
        restored.Deserialize(bytes);
        var restoredPredictions = restored.Predict(testData);

        // Assert: Identical predictions
        Assert.Equal(originalPredictions.Length, restoredPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], restoredPredictions[i], precision: 10);
        }
    }

    [Fact]
    public async Task SerializeDeserialize_PreservesOptimizationResult()
    {
        // Arrange: Train with default optimizer that performs feature selection
        var (x, y) = CreateLinearDataset(samples: 60, features: 6, seed: 55);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync();

        var originalIndices = result.OptimizationResult?.SelectedFeatureIndices;
        Assert.NotNull(originalIndices);
        Assert.NotEmpty(originalIndices);

        // Act: Round-trip
        var bytes = result.Serialize();
        var restored = new AiModelResult<double, Matrix<double>, Vector<double>>();
        restored.Deserialize(bytes);

        // Assert: SelectedFeatureIndices preserved exactly
        var restoredIndices = restored.OptimizationResult?.SelectedFeatureIndices;
        Assert.NotNull(restoredIndices);
        Assert.Equal(originalIndices.Count, restoredIndices.Count);
        for (int i = 0; i < originalIndices.Count; i++)
        {
            Assert.Equal(originalIndices[i], restoredIndices[i]);
        }

        // Assert: BestSolution restored (model can predict)
        Assert.NotNull(restored.OptimizationResult?.BestSolution);
    }

    [Fact]
    public async Task SerializeDeserialize_PreservesPreprocessingInfo()
    {
        // Arrange: Train with explicit preprocessing
        var (x, y) = CreateLinearDataset(samples: 60, features: 4, seed: 88);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigurePreprocessing(new StandardScaler<double>())
            .BuildAsync();

        // Verify preprocessing is fitted before serialization
        Assert.NotNull(result.PreprocessingInfo);
        Assert.True(result.PreprocessingInfo.IsFitted,
            "PreprocessingInfo should be fitted after training");

        var testData = CreateTestData(rows: 3, cols: 4, seed: 300);
        var originalPredictions = result.Predict(testData);

        // Act: Round-trip
        var bytes = result.Serialize();
        var restored = new AiModelResult<double, Matrix<double>, Vector<double>>();
        restored.Deserialize(bytes);

        // Assert: PreprocessingInfo preserved and functional
        Assert.NotNull(restored.PreprocessingInfo);
        Assert.True(restored.PreprocessingInfo.IsFitted,
            "PreprocessingInfo.IsFitted should be true after deserialization");

        // Predictions through deserialized model with preprocessing must match
        var restoredPredictions = restored.Predict(testData);
        Assert.Equal(originalPredictions.Length, restoredPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], restoredPredictions[i], precision: 8);
        }
    }

    [Fact]
    public void SerializeDeserialize_EmptyModel_SerializeDoesNotCrash()
    {
        // Arrange: Create an AiModelResult with no model set
        var emptyResult = new AiModelResult<double, Matrix<double>, Vector<double>>();

        // Act: Serialize should not throw (fixed: SupportsJitCompilation returns false for null model)
        var bytes = emptyResult.Serialize();
        Assert.NotNull(bytes);

        // Deserialize should also not throw
        var restored = new AiModelResult<double, Matrix<double>, Vector<double>>();
        restored.Deserialize(bytes);

        // But Predict SHOULD throw because there's no model
        Assert.Throws<InvalidOperationException>(() =>
        {
            var testData = CreateTestData(rows: 1, cols: 2, seed: 1);
            restored.Predict(testData);
        });
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
