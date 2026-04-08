using AiDotNet.CrossValidators;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.TestUtilities;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Integration tests for NestedCrossValidator.
/// Tests outer/inner loop cross-validation for hyperparameter tuning.
/// </summary>
public class NestedCrossValidatorIntegrationTests
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

    /// <summary>
    /// Simple model selector that returns the model from the result.
    /// </summary>
    private static IFullModel<double, Matrix<double>, Vector<double>> SelectBestModel(
        CrossValidationResult<double, Matrix<double>, Vector<double>> result)
    {
        // Return the model from the first fold
        return result.FoldResults.First().Model ?? CreateMockModel();
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidators_CreatesNestedValidator()
    {
        // Arrange
        var outerOptions = new CrossValidationOptions { NumberOfFolds = 3 };
        var innerOptions = new CrossValidationOptions { NumberOfFolds = 2 };
        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(outerOptions);
        var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(innerOptions);

        // Act
        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, SelectBestModel);

        // Assert
        Assert.NotNull(nestedValidator);
    }

    [Fact]
    public void Constructor_WithOptions_CreatesNestedValidator()
    {
        // Arrange
        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 3 });
        var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 2 });
        var nestedOptions = new CrossValidationOptions { RandomSeed = 42 };

        // Act
        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, SelectBestModel, nestedOptions);

        // Assert
        Assert.NotNull(nestedValidator);
    }

    #endregion

    #region Outer Loop Tests

    [Fact]
    public void Validate_ReturnsCorrectNumberOfOuterFolds()
    {
        // Arrange
        int outerFolds = 3;
        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = outerFolds, ShuffleData = false });
        var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 2, ShuffleData = false });

        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, SelectBestModel);

        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(60, 2);
        var y = CreateTestVector(60);

        // Act
        var result = nestedValidator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(outerFolds, result.FoldResults.Count);
    }

    #endregion

    #region No Data Leakage Tests

    [Fact]
    public void Validate_OuterFoldsHaveNoOverlap()
    {
        // Arrange
        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 3, ShuffleData = false });
        var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 2, ShuffleData = false });

        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, SelectBestModel);

        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(60, 2);
        var y = CreateTestVector(60);

        // Act
        var result = nestedValidator.Validate(model, X, y, optimizer);

        // Assert - No overlap between train and validation in outer folds
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

    #region Result Structure Tests

    [Fact]
    public void Validate_ReturnsValidFoldResults()
    {
        // Arrange
        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 3, ShuffleData = false });
        var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 2, ShuffleData = false });

        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, SelectBestModel);

        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(60, 2);
        var y = CreateTestVector(60);

        // Act
        var result = nestedValidator.Validate(model, X, y, optimizer);

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

    [Fact]
    public void Validate_RecordsTotalTimeIncludingInnerLoops()
    {
        // Arrange
        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 3, ShuffleData = false });
        var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 2, ShuffleData = false });

        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, SelectBestModel);

        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(60, 2);
        var y = CreateTestVector(60);

        // Act
        var result = nestedValidator.Validate(model, X, y, optimizer);

        // Assert
        Assert.True(result.TotalTime > TimeSpan.Zero);

        // Each fold should also have training time (includes inner CV time)
        foreach (var foldResult in result.FoldResults)
        {
            Assert.True(foldResult.TrainingTime >= TimeSpan.Zero);
        }
    }

    #endregion

    #region Model Selection Tests

    [Fact]
    public void Validate_UsesModelSelectorForEachOuterFold()
    {
        // Arrange
        int modelSelectionCallCount = 0;

        IFullModel<double, Matrix<double>, Vector<double>> TrackingModelSelector(
            CrossValidationResult<double, Matrix<double>, Vector<double>> result)
        {
            modelSelectionCallCount++;
            return result.FoldResults.First().Model ?? CreateMockModel();
        }

        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 3, ShuffleData = false });
        var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 2, ShuffleData = false });

        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, TrackingModelSelector);

        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(60, 2);
        var y = CreateTestVector(60);

        // Act
        var result = nestedValidator.Validate(model, X, y, optimizer);

        // Assert - Model selector should be called once per outer fold
        Assert.Equal(3, modelSelectionCallCount);
    }

    #endregion

    #region Different Validator Combinations Tests

    [Fact]
    public void Validate_WorksWithDifferentValidatorTypes()
    {
        // Arrange - Use KFold for outer and Standard for inner
        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 3, ShuffleData = false });
        var innerValidator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 2, ShuffleData = false });

        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, SelectBestModel);

        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(60, 2);
        var y = CreateTestVector(60);

        // Act
        var result = nestedValidator.Validate(model, X, y, optimizer);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(3, result.FoldResults.Count);
    }

    #endregion

    #region Training Index Tests

    [Fact]
    public void Validate_TrainingIndicesAreFromOuterFold()
    {
        // Arrange
        int numSamples = 60;
        var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 3, ShuffleData = false });
        var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
            new CrossValidationOptions { NumberOfFolds = 2, ShuffleData = false });

        var nestedValidator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
            outerValidator, innerValidator, SelectBestModel);

        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = nestedValidator.Validate(model, X, y, optimizer);

        // Assert - Training indices should be valid
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            // All indices should be within valid range
            Assert.All(foldResult.TrainingIndices, i => Assert.InRange(i, 0, numSamples - 1));
            Assert.All(foldResult.ValidationIndices, i => Assert.InRange(i, 0, numSamples - 1));
        }
    }

    #endregion
}
