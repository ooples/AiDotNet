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
    public void Predict_WrongDimensionInput_ThrowsDimensionError()
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
        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        // Act: Predict with WRONG dimension (2 features instead of 4)
        var wrongDimData = new Matrix<double>(2, 2);
        wrongDimData[0, 0] = 1.0; wrongDimData[0, 1] = 2.0;
        wrongDimData[1, 0] = 3.0; wrongDimData[1, 1] = 4.0;

        // Assert: Should throw a dimension-related error.
        // The model was trained on 4 features; predicting with 2 must fail.
        var ex = Assert.ThrowsAny<Exception>(() => result.Predict(wrongDimData));
        Assert.True(
            ex is InvalidOperationException or ArgumentException or ArgumentOutOfRangeException,
            $"Expected a dimension-related error, got {ex.GetType().Name}: {ex.Message}");
    }

    [Fact]
    public void BuildAsync_EmptyDataset_ThrowsArgumentException()
    {
        // Arrange: Create an empty matrix and vector
        var x = new Matrix<double>(0, 3);
        var y = new Vector<double>(0);
        var loader = DataLoaders.FromMatrixVector(x, y);

        // Act/Assert: Building with empty data should throw an argument-related exception
        var ex = Assert.ThrowsAny<ArgumentException>(() =>
        {
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(new RidgeRegression<double>())
                .BuildAsync()
                .GetAwaiter()
                .GetResult();
        });
        Assert.NotEmpty(ex.Message);
    }

    [Fact]
    public void BuildAsync_SingleSampleDataset_ThrowsInsufficientDataError()
    {
        // Arrange: Only 1 sample — can't meaningfully split train/test
        var x = new Matrix<double>(1, 2);
        x[0, 0] = 1.0; x[0, 1] = 2.0;
        var y = new Vector<double>(1);
        y[0] = 3.0;
        var loader = DataLoaders.FromMatrixVector(x, y);

        // Act/Assert: Should throw a meaningful error about insufficient data
        // Must be an ArgumentException or InvalidOperationException (not NullReferenceException etc.)
        var ex = Assert.ThrowsAny<Exception>(() =>
        {
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(new RidgeRegression<double>())
                .BuildAsync()
                .GetAwaiter()
                .GetResult();
        });

        Assert.True(
            ex is InvalidOperationException or ArgumentException or ArgumentOutOfRangeException,
            $"Expected a data-size error (InvalidOperationException or ArgumentException), " +
            $"got {ex.GetType().Name}: {ex.Message}");
        Assert.NotEmpty(ex.Message);
        // The error message should mention the data issue, not be a generic crash
        Assert.True(
            ex.Message.Contains("sample", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("data", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("row", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("size", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("split", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("insufficient", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("empty", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("length", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("count", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("must", StringComparison.OrdinalIgnoreCase),
            $"Error message should mention the data issue, got: {ex.Message}");
    }
}
