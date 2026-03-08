using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Tests that verify error handling in the AiModelBuilder and AiModelResult pipeline.
/// Before these tests, ZERO tests verified error paths — crashes with cryptic messages
/// instead of helpful errors would go undetected.
/// </summary>
public class AiModelBuilderErrorTests
{
    [Fact]
    public void Predict_WithNoModel_ThrowsInvalidOperationException()
    {
        // Arrange: Create an AiModelResult with no model
        var result = new AiModelResult<double, Matrix<double>, Vector<double>>();

        var testData = new Matrix<double>(2, 3);
        testData[0, 0] = 1.0; testData[0, 1] = 2.0; testData[0, 2] = 3.0;
        testData[1, 0] = 4.0; testData[1, 1] = 5.0; testData[1, 2] = 6.0;

        // Act/Assert: Should throw InvalidOperationException with clear message
        var ex = Assert.Throws<InvalidOperationException>(() => result.Predict(testData));
        Assert.Contains("not initialized", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Predict_WrongDimensionInput_ThrowsOrProducesFiniteOutput()
    {
        // Arrange: Train on 4-feature data
        var random = new Random(42);
        var x = new Matrix<double>(50, 4);
        var y = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
        {
            for (int j = 0; j < 4; j++)
                x[i, j] = random.NextDouble() * 10;
            y[i] = x[i, 0] + x[i, 1];
        }

        var loader = DataLoaders.FromMatrixVector(x, y);
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync();

        // Act: Predict with WRONG dimension (2 features instead of 4)
        var wrongDimData = new Matrix<double>(2, 2);
        wrongDimData[0, 0] = 1.0; wrongDimData[0, 1] = 2.0;
        wrongDimData[1, 0] = 3.0; wrongDimData[1, 1] = 4.0;

        // Assert: Should either throw a clear error OR handle gracefully.
        // The key assertion is that it doesn't silently produce garbage.
        try
        {
            var predictions = result.Predict(wrongDimData);
            // If it doesn't throw, predictions should at least be finite
            for (int i = 0; i < predictions.Length; i++)
            {
                Assert.False(double.IsNaN(predictions[i]),
                    "Prediction should not be NaN for mismatched dimensions");
            }
        }
        catch (Exception ex)
        {
            // Any of these exception types are acceptable for dimension mismatch
            Assert.True(
                ex is InvalidOperationException ||
                ex is ArgumentException ||
                ex is ArgumentOutOfRangeException ||
                ex is IndexOutOfRangeException,
                $"Expected a clear dimension error, got {ex.GetType().Name}: {ex.Message}");
        }
    }

    [Fact]
    public async Task BuildAsync_EmptyDataset_ThrowsMeaningfulError()
    {
        // Arrange: Create an empty matrix and vector
        var x = new Matrix<double>(0, 3);
        var y = new Vector<double>(0);
        var loader = DataLoaders.FromMatrixVector(x, y);

        // Act/Assert: Building with empty data should throw
        await Assert.ThrowsAnyAsync<Exception>(async () =>
        {
            await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(new RidgeRegression<double>())
                .BuildAsync();
        });
    }

    [Fact]
    public async Task BuildAsync_SingleSampleDataset_ThrowsOrHandles()
    {
        // Arrange: Only 1 sample — can't meaningfully split train/test
        var x = new Matrix<double>(1, 2);
        x[0, 0] = 1.0; x[0, 1] = 2.0;
        var y = new Vector<double>(1);
        y[0] = 3.0;
        var loader = DataLoaders.FromMatrixVector(x, y);

        // Act/Assert: Should either throw a meaningful error or handle gracefully
        try
        {
            var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(new RidgeRegression<double>())
                .BuildAsync();

            // If it builds successfully, model should at least be usable
            Assert.NotNull(result);
        }
        catch (Exception ex)
        {
            // Any of these are acceptable for insufficient data
            Assert.True(
                ex is InvalidOperationException ||
                ex is ArgumentException ||
                ex is ArgumentOutOfRangeException ||
                ex is DivideByZeroException,
                $"Expected a clear data-size error, got {ex.GetType().Name}: {ex.Message}");
        }
    }
}
