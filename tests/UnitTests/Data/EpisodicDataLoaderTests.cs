using AiDotNet.Data.Loaders;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Data;

/// <summary>
/// Unit tests for the EpisodicDataLoader class.
/// </summary>
public class EpisodicDataLoaderTests
{
    #region Test Helper Methods

    /// <summary>
    /// Creates a synthetic dataset for testing with the specified number of classes and examples per class.
    /// </summary>
    private (Matrix<double> X, Vector<double> Y) CreateTestDataset(int numClasses, int examplesPerClass, int numFeatures)
    {
        int totalExamples = numClasses * examplesPerClass;
        var X = new Matrix<double>(totalExamples, numFeatures);
        var Y = new Vector<double>(totalExamples);

        for (int classIdx = 0; classIdx < numClasses; classIdx++)
        {
            for (int exampleIdx = 0; exampleIdx < examplesPerClass; exampleIdx++)
            {
                int rowIdx = classIdx * examplesPerClass + exampleIdx;

                // Populate features with unique values based on class and example
                for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++)
                {
                    X[rowIdx, featureIdx] = classIdx * 1000 + exampleIdx * 10 + featureIdx;
                }

                // Set label
                Y[rowIdx] = classIdx;
            }
        }

        return (X, Y);
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_ValidInputs_InitializesSuccessfully()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);

        // Act
        var loader = new EpisodicDataLoader<double>(
            datasetX: X,
            datasetY: Y,
            nWay: 5,
            kShot: 3,
            queryShots: 10,
            seed: 42);

        // Assert
        Assert.NotNull(loader);
    }

    [Fact]
    public void Constructor_NullDatasetX_ThrowsArgumentNullException()
    {
        // Arrange
        var (_, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new EpisodicDataLoader<double>(
                datasetX: null!,
                datasetY: Y,
                nWay: 5,
                kShot: 3,
                queryShots: 10));

        Assert.Contains("datasetX", exception.Message);
    }

    [Fact]
    public void Constructor_NullDatasetY_ThrowsArgumentNullException()
    {
        // Arrange
        var (X, _) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new EpisodicDataLoader<double>(
                datasetX: X,
                datasetY: null!,
                nWay: 5,
                kShot: 3,
                queryShots: 10));

        Assert.Contains("datasetY", exception.Message);
    }

    [Fact]
    public void Constructor_MismatchedDimensions_ThrowsArgumentException()
    {
        // Arrange
        var X = new Matrix<double>(100, 5);
        var Y = new Vector<double>(50); // Mismatched: X has 100 rows, Y has 50 elements

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new EpisodicDataLoader<double>(
                datasetX: X,
                datasetY: Y,
                nWay: 5,
                kShot: 3,
                queryShots: 10));

        Assert.Contains("must match", exception.Message);
    }

    [Fact]
    public void Constructor_NWayLessThan2_ThrowsArgumentException()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new EpisodicDataLoader<double>(
                datasetX: X,
                datasetY: Y,
                nWay: 1,
                kShot: 3,
                queryShots: 10));

        Assert.Contains("nWay", exception.ParamName);
        Assert.Contains("at least 2", exception.Message);
    }

    [Fact]
    public void Constructor_KShotLessThan1_ThrowsArgumentException()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new EpisodicDataLoader<double>(
                datasetX: X,
                datasetY: Y,
                nWay: 5,
                kShot: 0,
                queryShots: 10));

        Assert.Contains("kShot", exception.ParamName);
        Assert.Contains("at least 1", exception.Message);
    }

    [Fact]
    public void Constructor_QueryShotsLessThan1_ThrowsArgumentException()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new EpisodicDataLoader<double>(
                datasetX: X,
                datasetY: Y,
                nWay: 5,
                kShot: 3,
                queryShots: 0));

        Assert.Contains("queryShots", exception.ParamName);
        Assert.Contains("at least 1", exception.Message);
    }

    [Fact]
    public void Constructor_InsufficientClasses_ThrowsArgumentException()
    {
        // Arrange - Only 3 classes, but requesting 5-way
        var (X, Y) = CreateTestDataset(numClasses: 3, examplesPerClass: 20, numFeatures: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new EpisodicDataLoader<double>(
                datasetX: X,
                datasetY: Y,
                nWay: 5,
                kShot: 3,
                queryShots: 10));

        Assert.Contains("only 3 classes", exception.Message);
        Assert.Contains("nWay=5", exception.Message);
    }

    [Fact]
    public void Constructor_InsufficientExamplesPerClass_ThrowsArgumentException()
    {
        // Arrange - Only 5 examples per class, but need 3+10=13
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 5, numFeatures: 5);

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new EpisodicDataLoader<double>(
                datasetX: X,
                datasetY: Y,
                nWay: 5,
                kShot: 3,
                queryShots: 10));

        Assert.Contains("insufficient examples", exception.Message);
    }

    #endregion

    #region GetNextTask Tests

    [Fact]
    public void GetNextTask_VerifyTaskDimensions_MatchesExpectedShape()
    {
        // Arrange
        int nWay = 5;
        int kShot = 3;
        int queryShots = 10;
        int numFeatures = 784;

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures);
        var loader = new EpisodicDataLoader<double>(X, Y, nWay, kShot, queryShots, seed: 42);

        // Act
        var task = loader.GetNextTask();

        // Assert - Support set dimensions
        Assert.Equal(2, task.SupportSetX.Rank); // Should be 2D tensor
        Assert.Equal(nWay * kShot, task.SupportSetX.Shape[0]); // 5 * 3 = 15 examples
        Assert.Equal(numFeatures, task.SupportSetX.Shape[1]); // 784 features

        Assert.Equal(1, task.SupportSetY.Rank); // Should be 1D tensor
        Assert.Equal(nWay * kShot, task.SupportSetY.Shape[0]); // 15 labels

        // Assert - Query set dimensions
        Assert.Equal(2, task.QuerySetX.Rank); // Should be 2D tensor
        Assert.Equal(nWay * queryShots, task.QuerySetX.Shape[0]); // 5 * 10 = 50 examples
        Assert.Equal(numFeatures, task.QuerySetX.Shape[1]); // 784 features

        Assert.Equal(1, task.QuerySetY.Rank); // Should be 1D tensor
        Assert.Equal(nWay * queryShots, task.QuerySetY.Shape[0]); // 50 labels
    }

    [Fact]
    public void GetNextTask_VerifyUniqueClasses_ExactlyNWayClasses()
    {
        // Arrange
        int nWay = 5;
        int kShot = 3;
        int queryShots = 10;

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);
        var loader = new EpisodicDataLoader<double>(X, Y, nWay, kShot, queryShots, seed: 42);

        // Act
        var task = loader.GetNextTask();

        // Assert - Support set has exactly nWay unique classes
        var supportClasses = new HashSet<double>();
        for (int i = 0; i < task.SupportSetY.Shape[0]; i++)
        {
            supportClasses.Add(task.SupportSetY[new[] { i }]);
        }
        Assert.Equal(nWay, supportClasses.Count);

        // Assert - Query set has exactly nWay unique classes
        var queryClasses = new HashSet<double>();
        for (int i = 0; i < task.QuerySetY.Shape[0]; i++)
        {
            queryClasses.Add(task.QuerySetY[new[] { i }]);
        }
        Assert.Equal(nWay, queryClasses.Count);

        // Assert - Same classes appear in both sets
        Assert.Equal(supportClasses, queryClasses);

        // Assert - Classes are remapped to 0..nWay-1
        var expectedClasses = new HashSet<double> { 0, 1, 2, 3, 4 };
        Assert.Equal(expectedClasses, supportClasses);
    }

    [Fact]
    public void GetNextTask_VerifyNoOverlap_SupportAndQuerySetsAreDisjoint()
    {
        // Arrange
        int nWay = 5;
        int kShot = 3;
        int queryShots = 10;
        int numFeatures = 10;

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures);
        var loader = new EpisodicDataLoader<double>(X, Y, nWay, kShot, queryShots, seed: 42);

        // Act
        var task = loader.GetNextTask();

        // Assert - Build a set of "fingerprints" for support examples
        // We'll use the first 3 features as a simple fingerprint
        var supportFingerprints = new HashSet<string>();
        for (int i = 0; i < task.SupportSetX.Shape[0]; i++)
        {
            var fingerprint = $"{task.SupportSetX[new[] { i, 0 }]}_{task.SupportSetX[new[] { i, 1 }]}_{task.SupportSetX[new[] { i, 2 }]}";
            supportFingerprints.Add(fingerprint);
        }

        // Assert - Check that no query examples have the same fingerprint
        var overlaps = new List<string>();
        for (int i = 0; i < task.QuerySetX.Shape[0]; i++)
        {
            var fingerprint = $"{task.QuerySetX[new[] { i, 0 }]}_{task.QuerySetX[new[] { i, 1 }]}_{task.QuerySetX[new[] { i, 2 }]}";
            if (supportFingerprints.Contains(fingerprint))
            {
                overlaps.Add(fingerprint);
            }
        }

        Assert.Empty(overlaps); // No overlaps should exist
    }

    [Fact]
    public void GetNextTask_VerifyClassDistribution_EachClassHasCorrectCounts()
    {
        // Arrange
        int nWay = 5;
        int kShot = 3;
        int queryShots = 10;

        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);
        var loader = new EpisodicDataLoader<double>(X, Y, nWay, kShot, queryShots, seed: 42);

        // Act
        var task = loader.GetNextTask();

        // Assert - Support set: each class should have exactly kShot examples
        var supportClassCounts = new Dictionary<double, int>();
        for (int i = 0; i < task.SupportSetY.Shape[0]; i++)
        {
            double label = task.SupportSetY[new[] { i }];
            if (!supportClassCounts.ContainsKey(label))
            {
                supportClassCounts[label] = 0;
            }
            supportClassCounts[label]++;
        }

        Assert.Equal(nWay, supportClassCounts.Count);
        foreach (var count in supportClassCounts.Values)
        {
            Assert.Equal(kShot, count);
        }

        // Assert - Query set: each class should have exactly queryShots examples
        var queryClassCounts = new Dictionary<double, int>();
        for (int i = 0; i < task.QuerySetY.Shape[0]; i++)
        {
            double label = task.QuerySetY[new[] { i }];
            if (!queryClassCounts.ContainsKey(label))
            {
                queryClassCounts[label] = 0;
            }
            queryClassCounts[label]++;
        }

        Assert.Equal(nWay, queryClassCounts.Count);
        foreach (var count in queryClassCounts.Values)
        {
            Assert.Equal(queryShots, count);
        }
    }

    [Fact]
    public void GetNextTask_MultipleCalls_ProducesDifferentTasks()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);
        var loader = new EpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 3, queryShots: 10);

        // Act - Generate multiple tasks
        var task1 = loader.GetNextTask();
        var task2 = loader.GetNextTask();
        var task3 = loader.GetNextTask();

        // Assert - Tasks should have different data (check first example of support set)
        bool task1And2Different = task1.SupportSetX[new[] { 0, 0 }] != task2.SupportSetX[new[] { 0, 0 }] ||
                                   task1.SupportSetX[new[] { 0, 1 }] != task2.SupportSetX[new[] { 0, 1 }];

        bool task2And3Different = task2.SupportSetX[new[] { 0, 0 }] != task3.SupportSetX[new[] { 0, 0 }] ||
                                   task2.SupportSetX[new[] { 0, 1 }] != task3.SupportSetX[new[] { 0, 1 }];

        // At least one pair should be different (very high probability with randomness)
        Assert.True(task1And2Different || task2And3Different);
    }

    [Fact]
    public void GetNextTask_WithSeed_ProducesReproducibleTasks()
    {
        // Arrange
        int seed = 42;
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 20, numFeatures: 5);

        var loader1 = new EpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: seed);
        var loader2 = new EpisodicDataLoader<double>(X, Y, nWay: 5, kShot: 3, queryShots: 10, seed: seed);

        // Act
        var task1 = loader1.GetNextTask();
        var task2 = loader2.GetNextTask();

        // Assert - Both tasks should be identical
        Assert.Equal(task1.SupportSetX.Shape[0], task2.SupportSetX.Shape[0]);
        Assert.Equal(task1.SupportSetX.Shape[1], task2.SupportSetX.Shape[1]);

        // Check first few values to verify they match
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(task1.SupportSetX[new[] { i, 0 }], task2.SupportSetX[new[] { i, 0 }]);
            Assert.Equal(task1.SupportSetY[new[] { i }], task2.SupportSetY[new[] { i }]);
        }
    }

    [Theory]
    [InlineData(2, 1, 1)]  // Minimal: 2-way 1-shot
    [InlineData(5, 1, 5)]  // Standard: 5-way 1-shot
    [InlineData(5, 5, 15)] // Standard: 5-way 5-shot
    [InlineData(20, 1, 1)] // Many-way: 20-way 1-shot
    public void GetNextTask_VariousConfigurations_ProducesCorrectDimensions(int nWay, int kShot, int queryShots)
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 25, examplesPerClass: 30, numFeatures: 10);
        var loader = new EpisodicDataLoader<double>(X, Y, nWay, kShot, queryShots, seed: 42);

        // Act
        var task = loader.GetNextTask();

        // Assert
        Assert.Equal(nWay * kShot, task.SupportSetX.Shape[0]);
        Assert.Equal(nWay * kShot, task.SupportSetY.Shape[0]);
        Assert.Equal(nWay * queryShots, task.QuerySetX.Shape[0]);
        Assert.Equal(nWay * queryShots, task.QuerySetY.Shape[0]);
    }

    [Fact]
    public void GetNextTask_VerifyDataIntegrity_FeaturesMatchOriginalDataset()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 5, examplesPerClass: 20, numFeatures: 10);
        var loader = new EpisodicDataLoader<double>(X, Y, nWay: 3, kShot: 2, queryShots: 5, seed: 42);

        // Act
        var task = loader.GetNextTask();

        // Assert - All values in the task should come from the original dataset
        // Check that all support examples have valid feature values
        for (int i = 0; i < task.SupportSetX.Shape[0]; i++)
        {
            for (int j = 0; j < task.SupportSetX.Shape[1]; j++)
            {
                double value = task.SupportSetX[new[] { i, j }];
                // Values should be in the range we created (0 to 5000)
                Assert.InRange(value, 0, 5000);
            }
        }

        // Check that all query examples have valid feature values
        for (int i = 0; i < task.QuerySetX.Shape[0]; i++)
        {
            for (int j = 0; j < task.QuerySetX.Shape[1]; j++)
            {
                double value = task.QuerySetX[new[] { i, j }];
                // Values should be in the range we created (0 to 5000)
                Assert.InRange(value, 0, 5000);
            }
        }
    }

    #endregion
}
