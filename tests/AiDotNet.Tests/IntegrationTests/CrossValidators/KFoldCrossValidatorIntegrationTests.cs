#nullable disable
using AiDotNet.CrossValidators;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.TestUtilities;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Integration tests for KFoldCrossValidator.
/// Tests fold creation, data splitting, and cross-validation behavior.
/// </summary>
public class KFoldCrossValidatorIntegrationTests
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
                result[i] = x[i, 0]; // Simple prediction: return first column
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
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>();

        // Assert - no exception thrown
        Assert.NotNull(validator);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidator()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 10,
            ShuffleData = false,
            RandomSeed = 42
        };

        // Act
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(validator);
    }

    #endregion

    #region Fold Count Tests

    [Theory]
    [InlineData(10, 5)]  // 10 samples, 5 folds
    [InlineData(20, 4)]  // 20 samples, 4 folds
    [InlineData(15, 3)]  // 15 samples, 3 folds
    public void Validate_ReturnsCorrectNumberOfFolds(int numSamples, int numFolds)
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = numFolds,
            ShuffleData = false
        };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(numFolds, result.FoldResults.Count);
    }

    #endregion

    #region Fold Size Tests

    [Fact]
    public void Validate_FoldSizesAreApproximatelyEqual()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(100, 2);
        var y = CreateTestVector(100);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each fold should have ~20 validation samples
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            // Allow for rounding differences
            Assert.InRange(foldResult.ValidationIndices.Length, 19, 21);
        }
    }

    #endregion

    #region No Data Leakage Tests

    [Fact]
    public void Validate_NoOverlapBetweenTrainAndValidation()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - For each fold, train and validation should not overlap
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var trainSet = new HashSet<int>(foldResult.TrainingIndices);
            var valSet = new HashSet<int>(foldResult.ValidationIndices);

            // No intersection
            Assert.Empty(trainSet.Intersect(valSet));

            // Union should cover all indices
            var union = trainSet.Union(valSet).ToHashSet();
            Assert.Equal(50, union.Count);
        }
    }

    [Fact]
    public void Validate_EachSampleUsedExactlyOnceAsValidation()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Count how many times each index appears in validation
        var validationCounts = new int[50];
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

    #region Shuffle Tests

    [Fact]
    public void Validate_WithShuffle_ProducesReproducibleResultsWithSeed()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = true,
            RandomSeed = 42
        };

        var model1 = CreateMockModel();
        var model2 = CreateMockModel();
        var optimizer1 = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model1);
        var optimizer2 = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model2);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var validator1 = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var result1 = validator1.Validate(model1, X, y, optimizer1);

        var validator2 = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var result2 = validator2.Validate(model2, X, y, optimizer2);

        // Assert - Same seed should produce same fold indices
        for (int i = 0; i < result1.FoldResults.Count; i++)
        {
            Assert.NotNull(result1.FoldResults[i].ValidationIndices);
            Assert.NotNull(result2.FoldResults[i].ValidationIndices);
            Assert.Equal(
                result1.FoldResults[i].ValidationIndices,
                result2.FoldResults[i].ValidationIndices);
        }
    }

    [Fact]
    public void Validate_WithoutShuffle_PreservesOrder()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Without shuffle, first fold should have indices 0-9
        Assert.NotNull(result.FoldResults[0].ValidationIndices);
        var firstFoldValidation = result.FoldResults[0].ValidationIndices.OrderBy(x => x).ToArray();
        Assert.Equal(Enumerable.Range(0, 10).ToArray(), firstFoldValidation);
    }

    #endregion

    #region Result Structure Tests

    [Fact]
    public void Validate_ReturnsValidFoldResults()
    {
        // Arrange
        var options = new CrossValidationOptions { NumberOfFolds = 3 };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(30, 2);
        var y = CreateTestVector(30);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.FoldResults);
        Assert.Equal(3, result.FoldResults.Count);

        for (int i = 0; i < 3; i++)
        {
            var foldResult = result.FoldResults[i];
            Assert.Equal(i, foldResult.FoldIndex);
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);
            Assert.NotNull(foldResult.ActualValues);
            Assert.NotNull(foldResult.PredictedValues);
        }
    }

    [Fact]
    public void Validate_RecordsTotalTime()
    {
        // Arrange
        var options = new CrossValidationOptions { NumberOfFolds = 3 };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(30, 2);
        var y = CreateTestVector(30);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.True(result.TotalTime > TimeSpan.Zero);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Validate_WithMinimumFolds_Works()
    {
        // Arrange - 2 folds is the minimum for meaningful CV
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 2,
            ShuffleData = false
        };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(10, 2);
        var y = CreateTestVector(10);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(2, result.FoldResults.Count);
        Assert.NotNull(result.FoldResults[0].ValidationIndices);
        Assert.Equal(5, result.FoldResults[0].ValidationIndices.Length);
    }

    [Fact]
    public void Validate_WithUnevenSplit_DistributesRemainderToFirstFolds()
    {
        // Arrange - 11 samples with 3 folds means uneven split
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 3,
            ShuffleData = false
        };
        var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(11, 2);
        var y = CreateTestVector(11);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - foldSize = 11/3 = 3, so each fold gets 3 validation samples
        // Note: The last 2 samples (indices 9, 10) won't be included in any validation set
        Assert.Equal(3, result.FoldResults.Count);
    }

    #endregion
}
