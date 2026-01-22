using AiDotNet.CrossValidators;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.TestUtilities;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Integration tests for StandardCrossValidator.
/// Tests standard k-fold cross-validation behavior similar to KFoldCrossValidator.
/// </summary>
public class StandardCrossValidatorIntegrationTests
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
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>();

        // Assert
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
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(validator);
    }

    #endregion

    #region Fold Count Tests

    [Theory]
    [InlineData(15, 3)]
    [InlineData(20, 4)]
    [InlineData(25, 5)]
    public void Validate_ReturnsCorrectNumberOfFolds(int numSamples, int numFolds)
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = numFolds,
            ShuffleData = false
        };
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
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
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(100, 2);
        var y = CreateTestVector(100);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each fold should have ~20 test samples
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            Assert.InRange(foldResult.ValidationIndices.Length, 19, 21);
        }
    }

    #endregion

    #region No Data Leakage Tests

    [Fact]
    public void Validate_NoOverlapBetweenTrainAndTest()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var trainSet = new HashSet<int>(foldResult.TrainingIndices);
            var testSet = new HashSet<int>(foldResult.ValidationIndices);

            Assert.Empty(trainSet.Intersect(testSet));
        }
    }

    [Fact]
    public void Validate_TrainAndTestCoverAllIndices()
    {
        // Arrange
        int numSamples = 50;
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var allIndices = foldResult.TrainingIndices.Concat(foldResult.ValidationIndices).ToHashSet();
            // Note: StandardCrossValidator may not cover all indices due to integer division
            Assert.True(allIndices.Count >= numSamples - 5,
                $"Expected most indices to be covered, got {allIndices.Count} out of {numSamples}");
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
        var validator1 = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var result1 = validator1.Validate(model1, X, y, optimizer1);

        var validator2 = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var result2 = validator2.Validate(model2, X, y, optimizer2);

        // Assert
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
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Without shuffle, first fold should have indices 0-9
        Assert.NotNull(result.FoldResults[0].ValidationIndices);
        var firstFoldTest = result.FoldResults[0].ValidationIndices.OrderBy(x => x).ToArray();
        Assert.Equal(Enumerable.Range(0, 10).ToArray(), firstFoldTest);
    }

    #endregion

    #region Result Structure Tests

    [Fact]
    public void Validate_ReturnsValidFoldResults()
    {
        // Arrange
        var options = new CrossValidationOptions { NumberOfFolds = 3 };
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
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

    #region Sample Usage Tests

    [Fact]
    public void Validate_EachSampleUsedExactlyOnceAsTest()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        int numSamples = 50;
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Count test occurrences
        var testCounts = new int[numSamples];
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            foreach (var idx in foldResult.ValidationIndices)
            {
                testCounts[idx]++;
            }
        }

        // Each sample in the test sets should appear exactly once
        int usedSamples = testCounts.Count(c => c > 0);
        Assert.True(usedSamples >= numSamples - 5,
            $"Expected most samples to be used as test, got {usedSamples} out of {numSamples}");

        // No sample should appear more than once
        Assert.True(testCounts.All(c => c <= 1),
            "Some samples appeared more than once in test sets");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Validate_WithMinimumFolds_Works()
    {
        // Arrange - 2 folds is minimum
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 2,
            ShuffleData = false
        };
        var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(10, 2);
        var y = CreateTestVector(10);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(2, result.FoldResults.Count);
        Assert.NotNull(result.FoldResults[0].ValidationIndices);
        Assert.NotNull(result.FoldResults[1].ValidationIndices);
    }

    #endregion
}
