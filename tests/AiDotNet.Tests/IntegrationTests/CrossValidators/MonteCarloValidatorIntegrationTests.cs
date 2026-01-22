using AiDotNet.CrossValidators;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.TestUtilities;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Integration tests for MonteCarloValidator.
/// Tests random repeated splits and validation size configuration.
/// </summary>
public class MonteCarloValidatorIntegrationTests
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
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(validator);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesValidator()
    {
        // Arrange
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 10,
            ValidationSize = 0.3,
            RandomSeed = 42
        };

        // Act
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(validator);
    }

    #endregion

    #region Number of Iterations Tests

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(10)]
    public void Validate_ReturnsCorrectNumberOfIterations(int numIterations)
    {
        // Arrange
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = numIterations,
            ValidationSize = 0.2,
            RandomSeed = 42
        };
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(numIterations, result.FoldResults.Count);
    }

    #endregion

    #region Validation Size Tests

    [Theory]
    [InlineData(0.1, 100, 10)]  // 10% validation
    [InlineData(0.2, 100, 20)]  // 20% validation
    [InlineData(0.3, 100, 30)]  // 30% validation
    public void Validate_ValidationSetSizeMatchesConfiguration(double validationRatio, int numSamples, int expectedValSize)
    {
        // Arrange
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 3,
            ValidationSize = validationRatio,
            RandomSeed = 42
        };
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each iteration should have the expected validation size
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            Assert.Equal(expectedValSize, foldResult.ValidationIndices.Length);
        }
    }

    [Fact]
    public void Validate_TrainingSizeIsComplementOfValidation()
    {
        // Arrange
        int numSamples = 100;
        double validationRatio = 0.2;
        int expectedValSize = (int)(numSamples * validationRatio);
        int expectedTrainSize = numSamples - expectedValSize;

        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 3,
            ValidationSize = validationRatio,
            RandomSeed = 42
        };
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
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
            Assert.Equal(expectedTrainSize, foldResult.TrainingIndices.Length);
        }
    }

    #endregion

    #region No Data Leakage Tests

    [Fact]
    public void Validate_NoOverlapBetweenTrainAndValidation()
    {
        // Arrange
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 5,
            ValidationSize = 0.2,
            RandomSeed = 42
        };
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - For each iteration, train and validation should not overlap
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
        int numSamples = 50;
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 3,
            ValidationSize = 0.2,
            RandomSeed = 42
        };
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Train + Validation should cover all indices in each iteration
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var allIndices = foldResult.TrainingIndices.Concat(foldResult.ValidationIndices).ToHashSet();
            Assert.Equal(numSamples, allIndices.Count);
        }
    }

    #endregion

    #region Randomness Tests

    [Fact]
    public void Validate_DifferentIterationsHaveDifferentSplits()
    {
        // Arrange
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 5,
            ValidationSize = 0.2,
            RandomSeed = 42
        };
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - With random splits, not all iterations should be identical
        var validationSets = result.FoldResults
            .Select(f => new HashSet<int>(f.ValidationIndices ?? Array.Empty<int>()))
            .ToList();

        // At least some validation sets should be different
        bool foundDifferent = false;
        for (int i = 1; i < validationSets.Count; i++)
        {
            if (!validationSets[0].SetEquals(validationSets[i]))
            {
                foundDifferent = true;
                break;
            }
        }

        Assert.True(foundDifferent, "Monte Carlo should produce different random splits across iterations");
    }

    [Fact]
    public void Validate_WithSameSeed_ProducesReproducibleResults()
    {
        // Arrange
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 3,
            ValidationSize = 0.2,
            RandomSeed = 42
        };

        var model1 = CreateMockModel();
        var model2 = CreateMockModel();
        var optimizer1 = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model1);
        var optimizer2 = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model2);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

        // Act
        var validator1 = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var result1 = validator1.Validate(model1, X, y, optimizer1);

        var validator2 = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var result2 = validator2.Validate(model2, X, y, optimizer2);

        // Assert - Same seed should produce same splits
        for (int i = 0; i < result1.FoldResults.Count; i++)
        {
            Assert.NotNull(result1.FoldResults[i].ValidationIndices);
            Assert.NotNull(result2.FoldResults[i].ValidationIndices);

            var set1 = new HashSet<int>(result1.FoldResults[i].ValidationIndices);
            var set2 = new HashSet<int>(result2.FoldResults[i].ValidationIndices);

            Assert.True(set1.SetEquals(set2), $"Iteration {i} should have same validation set with same seed");
        }
    }

    #endregion

    #region Result Structure Tests

    [Fact]
    public void Validate_ReturnsValidFoldResults()
    {
        // Arrange
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 3,
            ValidationSize = 0.2,
            RandomSeed = 42
        };
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(50, 2);
        var y = CreateTestVector(50);

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

    #region Sample Usage Tests

    [Fact]
    public void Validate_SamplesMayAppearInMultipleValidationSets()
    {
        // Arrange - Monte Carlo allows samples to appear in multiple validation sets
        var options = new MonteCarloValidationOptions
        {
            NumberOfFolds = 10,  // Many iterations
            ValidationSize = 0.2,
            RandomSeed = 42
        };
        var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        int numSamples = 50;
        var X = CreateTestMatrix(numSamples, 2);
        var y = CreateTestVector(numSamples);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Count validation occurrences
        var validationCounts = new int[numSamples];
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            foreach (var idx in foldResult.ValidationIndices)
            {
                validationCounts[idx]++;
            }
        }

        // Some samples should appear more than once (unlike k-fold where each appears exactly once)
        bool someAppearsMultipleTimes = validationCounts.Any(c => c > 1);
        Assert.True(someAppearsMultipleTimes,
            "With 10 iterations, some samples should appear in multiple validation sets");
    }

    #endregion
}
