using AiDotNet.CrossValidators;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.TestUtilities;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Integration tests for GroupKFoldCrossValidator.
/// Tests that samples from the same group are kept together in folds.
/// </summary>
public class GroupKFoldCrossValidatorIntegrationTests
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
    /// Creates group assignments for samples.
    /// Example: [0,0,0,1,1,1,2,2,2] means first 3 samples belong to group 0, etc.
    /// </summary>
    private static int[] CreateGroupAssignments(int[] samplesPerGroup)
    {
        var groups = new List<int>();
        for (int groupId = 0; groupId < samplesPerGroup.Length; groupId++)
        {
            for (int i = 0; i < samplesPerGroup[groupId]; i++)
            {
                groups.Add(groupId);
            }
        }
        return groups.ToArray();
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_WithGroups_CreatesValidator()
    {
        // Arrange
        int[] groups = [0, 0, 1, 1, 2, 2];

        // Act
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups);

        // Assert
        Assert.NotNull(validator);
    }

    [Fact]
    public void Constructor_WithGroupsAndOptions_CreatesValidator()
    {
        // Arrange
        int[] groups = [0, 0, 1, 1, 2, 2];
        var options = new CrossValidationOptions
        {
            NumberOfFolds = 3,
            RandomSeed = 42
        };

        // Act
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);

        // Assert
        Assert.NotNull(validator);
    }

    #endregion

    #region Group Preservation Tests

    [Fact]
    public void Validate_SamplesFromSameGroupStayTogether()
    {
        // Arrange - 3 groups with 4 samples each
        int[] groups = CreateGroupAssignments([4, 4, 4]);  // 12 samples total
        var options = new CrossValidationOptions { NumberOfFolds = 3 };
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(12, 2);
        var y = CreateTestVector(12);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - For each fold, check that indices from the same group are either all in train or all in validation
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var trainSet = new HashSet<int>(foldResult.TrainingIndices);
            var valSet = new HashSet<int>(foldResult.ValidationIndices);

            // Check each group
            for (int groupId = 0; groupId < 3; groupId++)
            {
                var groupIndices = Enumerable.Range(0, 12)
                    .Where(i => groups[i] == groupId)
                    .ToList();

                // All indices of this group should be either all in train or all in validation
                bool allInTrain = groupIndices.All(i => trainSet.Contains(i));
                bool allInVal = groupIndices.All(i => valSet.Contains(i));

                Assert.True(allInTrain || allInVal,
                    $"Group {groupId} should be entirely in training or entirely in validation");
            }
        }
    }

    [Fact]
    public void Validate_GroupNeverSplitAcrossTrainAndValidation()
    {
        // Arrange - 5 groups with varying sizes
        int[] groups = CreateGroupAssignments([3, 5, 2, 4, 6]);  // 20 samples total
        var options = new CrossValidationOptions { NumberOfFolds = 5 };
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(20, 2);
        var y = CreateTestVector(20);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            var trainSet = new HashSet<int>(foldResult.TrainingIndices);
            var valSet = new HashSet<int>(foldResult.ValidationIndices);

            // For each unique group, verify no split
            var uniqueGroups = groups.Distinct();
            foreach (var groupId in uniqueGroups)
            {
                var groupIndices = Enumerable.Range(0, groups.Length)
                    .Where(i => groups[i] == groupId)
                    .ToList();

                int inTrain = groupIndices.Count(i => trainSet.Contains(i));
                int inVal = groupIndices.Count(i => valSet.Contains(i));

                // Either all in train (inVal == 0) or all in val (inTrain == 0)
                Assert.True(inTrain == 0 || inVal == 0,
                    $"Group {groupId} has {inTrain} samples in train and {inVal} in validation - should not be split");
            }
        }
    }

    #endregion

    #region No Data Leakage Tests

    [Fact]
    public void Validate_NoOverlapBetweenTrainAndValidation()
    {
        // Arrange
        int[] groups = CreateGroupAssignments([5, 5, 5, 5]);  // 4 groups, 5 samples each
        var options = new CrossValidationOptions { NumberOfFolds = 4 };
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(20, 2);
        var y = CreateTestVector(20);

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

    [Fact]
    public void Validate_ReturnsCorrectNumberOfFolds()
    {
        // Arrange
        int[] groups = CreateGroupAssignments([3, 3, 3, 3, 3]);  // 5 groups
        var options = new CrossValidationOptions { NumberOfFolds = 5 };
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(15, 2);
        var y = CreateTestVector(15);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert
        Assert.Equal(5, result.FoldResults.Count);
    }

    #endregion

    #region Result Structure Tests

    [Fact]
    public void Validate_ReturnsValidFoldResults()
    {
        // Arrange
        int[] groups = CreateGroupAssignments([4, 4, 4]);
        var options = new CrossValidationOptions { NumberOfFolds = 3 };
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(12, 2);
        var y = CreateTestVector(12);

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

    #region Group Distribution Tests

    [Fact]
    public void Validate_EachGroupUsedExactlyOnceForValidation()
    {
        // Arrange - Equal number of groups and folds
        int[] groups = CreateGroupAssignments([2, 2, 2, 2, 2]);  // 5 groups
        var options = new CrossValidationOptions { NumberOfFolds = 5 };
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(10, 2);
        var y = CreateTestVector(10);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Each group should appear in exactly one validation set
        var groupValidationCounts = new int[5];
        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.ValidationIndices);
            var groupsInValidation = foldResult.ValidationIndices
                .Select(i => groups[i])
                .Distinct();

            foreach (var g in groupsInValidation)
            {
                groupValidationCounts[g]++;
            }
        }

        foreach (var count in groupValidationCounts)
        {
            Assert.Equal(1, count);
        }
    }

    #endregion

    #region Uneven Group Sizes Tests

    [Fact]
    public void Validate_HandlesUnevenGroupSizes()
    {
        // Arrange - Groups with different sizes
        int[] groups = CreateGroupAssignments([10, 2, 5, 3]);  // 4 groups, very different sizes
        var options = new CrossValidationOptions { NumberOfFolds = 4 };
        var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);
        var model = CreateMockModel();
        var optimizer = new PassthroughOptimizer<double, Matrix<double>, Vector<double>>(model);
        var X = CreateTestMatrix(20, 2);
        var y = CreateTestVector(20);

        // Act
        var result = validator.Validate(model, X, y, optimizer);

        // Assert - Should still work, groups should not be split
        Assert.Equal(4, result.FoldResults.Count);

        foreach (var foldResult in result.FoldResults)
        {
            Assert.NotNull(foldResult.TrainingIndices);
            Assert.NotNull(foldResult.ValidationIndices);

            // Verify no group split
            var trainSet = new HashSet<int>(foldResult.TrainingIndices);
            var valSet = new HashSet<int>(foldResult.ValidationIndices);

            for (int groupId = 0; groupId < 4; groupId++)
            {
                var groupIndices = Enumerable.Range(0, 20)
                    .Where(i => groups[i] == groupId)
                    .ToList();

                bool allInTrain = groupIndices.All(i => trainSet.Contains(i));
                bool allInVal = groupIndices.All(i => valSet.Contains(i));

                Assert.True(allInTrain || allInVal,
                    $"Group {groupId} should not be split across train and validation");
            }
        }
    }

    #endregion
}
