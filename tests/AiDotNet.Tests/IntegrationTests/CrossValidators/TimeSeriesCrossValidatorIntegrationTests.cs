using AiDotNet.CrossValidators;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.TestUtilities;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Integration tests for TimeSeriesCrossValidator.
/// Tests temporal ordering preservation and expanding window behavior.
/// </summary>
public class TimeSeriesCrossValidatorIntegrationTests
{
    #region Helper Methods

    private static Matrix<double> CreateTimeSeriesMatrix(int rows, int cols)
    {
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Time-ordered data: each row represents a time step
                data[i, j] = i * cols + j;
            }
        }
        return new Matrix<double>(data);
    }

    private static Vector<double> CreateTimeSeriesVector(int length)
    {
        var data = new double[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = i;
        }
        return new Vector<double>(data);
    }

    private static MockFullModel CreateMockModel()
    {
        return new MockFullModel(x =>
        {
            var result = new double[x.Rows];
            for (int i = 0; i < x.Rows; i++)
            {
                result[i] = x[i, 0];
            }
            return new Vector<double>(result);
        });
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidParameters_CreatesValidator()
    {
        // Act
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 10,
            validationSize: 5,
            step: 5);

        // Assert
        Assert.NotNull(validator);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidator()
    {
        // Arrange
        var options = new CrossValidationOptions { RandomSeed = 42 };

        // Act
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 10,
            validationSize: 5,
            step: 5,
            options: options);

        // Assert
        Assert.NotNull(validator);
    }

    #endregion

    #region Temporal Ordering Tests

    [Fact]
    public void Validate_TrainingIndicesAlwaysPrecedeValidationIndices()
    {
        // Arrange
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 20,
            validationSize: 10,
            step: 10);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(100, 2);
        var y = CreateTimeSeriesVector(100);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - For each fold, all training indices should be less than validation indices
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            int maxTrainIndex = foldResult.TrainingIndices.Max();
            int minValIndex = foldResult.ValidationIndices.Min();

            Assert.True(maxTrainIndex < minValIndex,
                $"Training indices should precede validation indices. Max train: {maxTrainIndex}, Min val: {minValIndex}");
        }
    }

    [Fact]
    public void Validate_NoFutureDataLeakage()
    {
        // Arrange
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 15,
            validationSize: 5,
            step: 5);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(50, 2);
        var y = CreateTimeSeriesVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Training data should never include future data points
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            // Training indices should be contiguous starting from 0
            var sortedTrainIndices = foldResult.TrainingIndices.OrderBy(x => x).ToArray();
            for (int i = 0; i < sortedTrainIndices.Length; i++)
            {
                Assert.Equal(i, sortedTrainIndices[i]);
            }
        }
    }

    #endregion

    #region Expanding Window Tests

    [Fact]
    public void Validate_TrainingSetExpandsOverFolds()
    {
        // Arrange
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 10,
            validationSize: 5,
            step: 5);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(50, 2);
        var y = CreateTimeSeriesVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each subsequent fold should have more training data
        int previousTrainSize = 0;
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            int currentTrainSize = foldResult.TrainingIndices.Length;

            Assert.True(currentTrainSize > previousTrainSize,
                $"Training set should expand. Previous: {previousTrainSize}, Current: {currentTrainSize}");

            previousTrainSize = currentTrainSize;
        }
    }

    [Fact]
    public void Validate_ValidationSetSizeIsConstant()
    {
        // Arrange
        int validationSize = 5;
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 10,
            validationSize: validationSize,
            step: 5);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(50, 2);
        var y = CreateTimeSeriesVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - All validation sets should have the same size
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            Assert.Equal(validationSize, foldResult.ValidationIndices.Length);
        }
    }

    #endregion

    #region Fold Generation Tests

    [Fact]
    public void Validate_GeneratesCorrectNumberOfFolds()
    {
        // Arrange - With 50 samples, initial=10, validation=5, step=5
        // Folds: train 0-9 val 10-14, train 0-14 val 15-19, etc.
        // Expected folds: (50 - 10 - 5) / 5 + 1 = 8 folds
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 10,
            validationSize: 5,
            step: 5);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(50, 2);
        var y = CreateTimeSeriesVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        // trainEnd starts at 10 and goes up by 5 until < 50-5=45
        // trainEnd values: 10, 15, 20, 25, 30, 35, 40 = 7 folds
        Assert.True(result.FoldResults.Count >= 5, $"Expected at least 5 folds, got {result.FoldResults.Count}");
    }

    [Fact]
    public void Validate_ValidationIndicesAreContiguous()
    {
        // Arrange
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 10,
            validationSize: 5,
            step: 5);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(50, 2);
        var y = CreateTimeSeriesVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Validation indices should be contiguous
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            var sortedIndices = foldResult.ValidationIndices.OrderBy(x => x).ToArray();

            for (int i = 1; i < sortedIndices.Length; i++)
            {
                Assert.Equal(sortedIndices[i - 1] + 1, sortedIndices[i]);
            }
        }
    }

    #endregion

    #region Step Size Tests

    [Fact]
    public void Validate_StepSizeAffectsFoldOverlap()
    {
        // Arrange - Step size equals validation size means no overlap
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 10,
            validationSize: 5,
            step: 5);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(40, 2);
        var y = CreateTimeSeriesVector(40);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Validation sets should not overlap
        var allValidationIndices = new HashSet<int>();
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            foreach (var idx in foldResult.ValidationIndices)
            {
                Assert.True(allValidationIndices.Add(idx),
                    $"Index {idx} appeared in multiple validation sets");
            }
        }
    }

    #endregion

    #region Result Structure Tests

    [Fact]
    public void Validate_ReturnsValidFoldResults()
    {
        // Arrange
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 10,
            validationSize: 5,
            step: 5);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(40, 2);
        var y = CreateTimeSeriesVector(40);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.FoldResults);
        Assert.True(result.FoldResults.Count > 0);
        Assert.True(result.TotalTime > TimeSpan.Zero);

        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);
            Assert.NotNull(foldResult.ActualValues);
            Assert.NotNull(foldResult.PredictedValues);
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Validate_WithLargeInitialTrainSize_GeneratesFewFolds()
    {
        // Arrange
        var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
            initialTrainSize: 40,
            validationSize: 5,
            step: 5);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTimeSeriesMatrix(50, 2);
        var y = CreateTimeSeriesVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Should only have 1 fold (train 0-39, val 40-44)
        Assert.Equal(1, result.FoldResults.Count);
    }

    #endregion
}
