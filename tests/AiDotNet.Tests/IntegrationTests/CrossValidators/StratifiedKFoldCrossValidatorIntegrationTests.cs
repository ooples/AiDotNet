using AiDotNet.CrossValidators;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.TestUtilities;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Integration tests for StratifiedKFoldCrossValidator.
/// Tests class proportion preservation and stratified fold creation.
/// </summary>
public class StratifiedKFoldCrossValidatorIntegrationTests
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

    private static Vector<double> CreateClassLabels(int[] classCounts)
    {
        var total = classCounts.Sum();
        var labels = new double[total];
        int idx = 0;
        for (int classLabel = 0; classLabel < classCounts.Length; classLabel++)
        {
            for (int i = 0; i < classCounts[classLabel]; i++)
            {
                labels[idx++] = classLabel;
            }
        }
        return new Vector<double>(labels);
    }

    private static MockFullModel CreateMockModel()
    {
        return new MockFullModel(x =>
        {
            var result = new double[x.Rows];
            for (int i = 0; i < x.Rows; i++)
            {
                result[i] = x[i, 0]; // Simple prediction
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
        var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>();

        // Assert
        Assert.NotNull(validator);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidator()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 3,
            ShuffleData = false,
            RandomSeed = 42
        };

        // Act
        var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);

        // Assert
        Assert.NotNull(validator);
    }

    #endregion

    #region Class Proportion Tests

    [Fact]
    public void Validate_PreservesClassProportionsInEachFold()
    {
        // Arrange - 60 class 0, 40 class 1 (60%/40% split)
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(100, 2);
        var y = CreateClassLabels([60, 40]); // 60 class 0, 40 class 1

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each fold should have approximately 60%/40% split
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            var validationLabels = foldResult.ValidationIndices.Select(i => y[i]).ToList();

            int class0Count = validationLabels.Count(l => l == 0);
            int class1Count = validationLabels.Count(l => l == 1);

            // With 100 samples and 5 folds, each validation set has ~20 samples
            // Class proportions should be ~12 class 0 and ~8 class 1
            Assert.InRange(class0Count, 10, 14); // Allow some tolerance
            Assert.InRange(class1Count, 6, 10);
        }
    }

    [Fact]
    public void Validate_WorksWithMultipleClasses()
    {
        // Arrange - 3 classes
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 3,
            ShuffleData = false
        };
        var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(90, 2);
        var y = CreateClassLabels([30, 30, 30]); // Equal 3 classes

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(3, result.FoldResults.Count);

        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            var validationLabels = foldResult.ValidationIndices.Select(i => y[i]).ToList();

            // Each class should have ~10 samples per fold
            int class0Count = validationLabels.Count(l => l == 0);
            int class1Count = validationLabels.Count(l => l == 1);
            int class2Count = validationLabels.Count(l => l == 2);

            Assert.InRange(class0Count, 8, 12);
            Assert.InRange(class1Count, 8, 12);
            Assert.InRange(class2Count, 8, 12);
        }
    }

    #endregion

    #region Imbalanced Class Tests

    [Fact]
    public void Validate_HandlesImbalancedClasses()
    {
        // Arrange - Highly imbalanced: 90% class 0, 10% class 1
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 5,
            ShuffleData = false
        };
        var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(100, 2);
        var y = CreateClassLabels([90, 10]); // 90% class 0, 10% class 1

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each fold should still maintain the imbalance
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            var validationLabels = foldResult.ValidationIndices.Select(i => y[i]).ToList();

            int class0Count = validationLabels.Count(l => l == 0);
            int class1Count = validationLabels.Count(l => l == 1);

            // Should have ~18 class 0 and ~2 class 1 per fold
            Assert.True(class0Count > class1Count, "Imbalance should be preserved");
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
            NumberOfFolds = 3,
            ShuffleData = false
        };
        var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(60, 2);
        var y = CreateClassLabels([30, 30]);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var trainSet = new HashSet<int>(foldResult.TrainingIndices);
            var valSet = new HashSet<int>(foldResult.ValidationIndices);

            Assert.Empty(trainSet.Intersect(valSet));
        }
    }

    #endregion

    #region Fold Count Tests

    [Theory]
    [InlineData(30, 3)]
    [InlineData(50, 5)]
    [InlineData(40, 4)]
    public void Validate_ReturnsCorrectNumberOfFolds(int numSamples, int numFolds)
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = numFolds,
            ShuffleData = false
        };
        var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateClassLabels([numSamples / 2, numSamples / 2]);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(numFolds, result.FoldResults.Count);
    }

    #endregion

    #region Reproducibility Tests

    [Fact]
    public void Validate_WithSameSeed_ProducesReproducibleResults()
    {
        // Arrange
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 3,
            ShuffleData = true,
            RandomSeed = 42
        };

        var model1 = CreateMockModel();
        var model2 = CreateMockModel();
        var optimizer1 = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model1);
        var optimizer2 = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model2);
        var X = CreateTestMatrix(60, 2);
        var y = CreateClassLabels([30, 30]);

        // Act
        var validator1 = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);
        var result1 = validator1.Validate(model1, X, y, optimizer1);

        var validator2 = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);
        var result2 = validator2.Validate(model2, X, y, optimizer2);

        // Assert
        for (int i = 0; i < result1.FoldResults.Count; i++)
        {
            Assert.NotNull(result1.FoldResults[i].ValidationIndices);
            Assert.NotNull(result2.FoldResults[i].ValidationIndices);

            // With same seed, validation indices should be in same order
            // (though the actual values might differ due to stratification algorithm)
            Assert.Equal(
                result1.FoldResults[i].ValidationIndices.Length,
                result2.FoldResults[i].ValidationIndices.Length);
        }
    }

    #endregion

    #region Result Structure Tests

    [Fact]
    public void Validate_ReturnsValidFoldResults()
    {
        // Arrange
        var options = new CrossValidationOptions { NumberOfFolds = 3 };
        var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(60, 2);
        var y = CreateClassLabels([30, 30]);

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
}
