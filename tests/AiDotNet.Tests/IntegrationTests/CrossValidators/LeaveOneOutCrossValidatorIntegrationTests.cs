#nullable disable
using AiDotNet.CrossValidators;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.TestUtilities;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Integration tests for LeaveOneOutCrossValidator.
/// Tests that each sample is used exactly once as validation.
/// </summary>
public class LeaveOneOutCrossValidatorIntegrationTests
{
    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(int rows, int cols)
    {
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                data[i, j] = i * cols + j;
            }
        }
        return new Matrix<double>(data);
    }

    private static Vector<double> CreateTestVector(int length)
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
    public void Constructor_WithDefaultOptions_CreatesValidator()
    {
        // Act
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(validator);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidator()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            ShuffleData = false,
            RandomSeed = 42
        };

        // Act
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(validator);
    }

    #endregion

    #region Number of Folds Tests

    [Theory]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(15)]
    public void Validate_NumberOfFoldsEqualsNumberOfSamples(int numSamples)
    {
        // Arrange
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = false });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - LOO should create exactly n folds for n samples
        Assert.Equal(numSamples, result.FoldResults.Count);
    }

    #endregion

    #region Single Sample Validation Tests

    [Fact]
    public void Validate_EachValidationSetHasExactlyOneSample()
    {
        // Arrange
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = false });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(10, 2);
        var y = CreateTestVector(10);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each validation set should have exactly 1 sample
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            Assert.Single(foldResult.ValidationIndices);
        }
    }

    [Fact]
    public void Validate_EachSampleUsedExactlyOnceForValidation()
    {
        // Arrange
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = false });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        int numSamples = 8;
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Count validation occurrences for each index
        var validationCounts = new int[numSamples];
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            foreach (var idx in foldResult.ValidationIndices)
            {
                validationCounts[idx]++;
            }
        }

        // Each sample should be used exactly once as validation
        foreach (var count in validationCounts)
        {
            Assert.Equal(1, count);
        }
    }

    #endregion

    #region Training Set Size Tests

    [Fact]
    public void Validate_TrainingSetHasNMinusOneSamples()
    {
        // Arrange
        int numSamples = 10;
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = false });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each training set should have n-1 samples
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.Equal(numSamples - 1, foldResult.TrainingIndices.Length);
        }
    }

    #endregion

    #region No Data Leakage Tests

    [Fact]
    public void Validate_NoOverlapBetweenTrainAndValidation()
    {
        // Arrange
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = false });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(10, 2);
        var y = CreateTestVector(10);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - For each fold, train and validation should not overlap
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var trainSet = new HashSet<int>(foldResult.TrainingIndices);
            var valSet = new HashSet<int>(foldResult.ValidationIndices);

            Assert.Empty(trainSet.Intersect(valSet));
        }
    }

    [Fact]
    public void Validate_TrainAndValidationCoverAllIndices()
    {
        // Arrange
        int numSamples = 10;
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = false });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Train + Validation should cover all indices
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var allIndices = foldResult.TrainingIndices.Concat(foldResult.ValidationIndices).ToHashSet();
            Assert.Equal(numSamples, allIndices.Count);
        }
    }

    #endregion

    #region Result Structure Tests

    [Fact]
    public void Validate_ReturnsValidFoldResults()
    {
        // Arrange
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = false });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(5, 2);
        var y = CreateTestVector(5);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.FoldResults);
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

    #region Shuffle Tests

    [Fact]
    public void Validate_WithShuffle_StillUsesEachSampleOnce()
    {
        // Arrange
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = true, RandomSeed = 42 });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        int numSamples = 10;
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Even with shuffle, each sample should be used exactly once for validation
        var validationCounts = new int[numSamples];
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            foreach (var idx in foldResult.ValidationIndices)
            {
                validationCounts[idx]++;
            }
        }

        foreach (var count in validationCounts)
        {
            Assert.Equal(1, count);
        }
    }

    [Fact]
    public void Validate_WithSameSeed_ProducesReproducibleResults()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            ShuffleData = true,
            RandomSeed = 42
        };

        var model1 = CreateMockModel();
        var model2 = CreateMockModel();
        var optimizer1 = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model1);
        var optimizer2 = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model2);
        var X = CreateTestMatrix(8, 2);
        var y = CreateTestVector(8);

        // Act
        var validator1 = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var result1 = validator1.Validate(model1, X, y, optimizer1);

        var validator2 = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var result2 = validator2.Validate(model2, X, y, optimizer2);

        // Assert - Same seed should produce same fold order
        for (int i = 0; i < result1.FoldResults.Count; i++)
        {
            Assert.NotNull(result1.FoldResults[i].ValidationIndices);
            Assert.NotNull(result2.FoldResults[i].ValidationIndices);
            Assert.Equal(
                result1.FoldResults[i].ValidationIndices,
                result2.FoldResults[i].ValidationIndices);
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Validate_WithTwoSamples_CreatesTwoFolds()
    {
        // Arrange - Minimum meaningful LOO
        var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { ShuffleData = false });
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(2, 2);
        var y = CreateTestVector(2);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(2, result.FoldResults.Count);

        // First fold: train on [1], validate on [0]
        // Second fold: train on [0], validate on [1]
        Assert.NotNull(result.FoldResults[0].ValidationIndices);
        Assert.NotNull(result.FoldResults[1].ValidationIndices);
        Assert.Single(result.FoldResults[0].ValidationIndices);
        Assert.Single(result.FoldResults[1].ValidationIndices);
    }

    #endregion
}
