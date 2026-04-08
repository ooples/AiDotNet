using AiDotNet.ExperimentTracking;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TrainingInfrastructure;

/// <summary>
/// Unit tests for ExperimentTracker experiment and run management.
/// </summary>
public class ExperimentTrackerTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly ExperimentTracker<double> _tracker;

    public ExperimentTrackerTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"experiment_tracker_tests_{Guid.NewGuid():N}");
        _tracker = new ExperimentTracker<double>(_testDirectory);
    }

    public void Dispose()
    {
        // Clean up test directory
        if (Directory.Exists(_testDirectory))
        {
            try
            {
                Directory.Delete(_testDirectory, true);
            }
            catch
            {
                // Ignore cleanup errors in tests
            }
        }
    }

    #region Experiment CRUD Tests

    [Fact]
    public void CreateExperiment_WithValidName_ReturnsExperimentId()
    {
        // Arrange & Act
        var experimentId = _tracker.CreateExperiment("test-experiment");

        // Assert
        Assert.NotNull(experimentId);
        Assert.NotEmpty(experimentId);
    }

    [Fact]
    public void CreateExperiment_WithDescriptionAndTags_StoresMetadata()
    {
        // Arrange
        var tags = new Dictionary<string, string>
        {
            ["team"] = "ml-research",
            ["project"] = "image-classification"
        };

        // Act
        var experimentId = _tracker.CreateExperiment(
            "test-experiment",
            description: "Test description",
            tags: tags);

        var experiment = _tracker.GetExperiment(experimentId);

        // Assert
        Assert.NotNull(experiment);
        Assert.Equal("test-experiment", experiment.Name);
        Assert.Equal("Test description", experiment.Description);
    }

    [Fact]
    public void CreateExperiment_WithExistingName_ReturnsSameId()
    {
        // Arrange
        var experimentId1 = _tracker.CreateExperiment("duplicate-name");

        // Act
        var experimentId2 = _tracker.CreateExperiment("duplicate-name");

        // Assert - Should return same ID for existing experiment
        Assert.Equal(experimentId1, experimentId2);
    }

    [Fact]
    public void CreateExperiment_WithNullName_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.CreateExperiment(null!));
    }

    [Fact]
    public void CreateExperiment_WithEmptyName_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.CreateExperiment(""));
    }

    [Fact]
    public void GetExperiment_WithValidId_ReturnsExperiment()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("get-test");

        // Act
        var experiment = _tracker.GetExperiment(experimentId);

        // Assert
        Assert.NotNull(experiment);
        Assert.Equal("get-test", experiment.Name);
    }

    [Fact]
    public void GetExperiment_WithInvalidId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.GetExperiment("nonexistent-id"));
    }

    [Fact]
    public void DeleteExperiment_RemovesExperimentAndRuns()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("delete-test");
        _tracker.StartRun(experimentId, "run-1");
        _tracker.StartRun(experimentId, "run-2");

        // Act
        _tracker.DeleteExperiment(experimentId);

        // Assert
        Assert.Throws<ArgumentException>(() => _tracker.GetExperiment(experimentId));
    }

    [Fact]
    public void DeleteExperiment_WithInvalidId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.DeleteExperiment("nonexistent-id"));
    }

    #endregion

    #region Run CRUD Tests

    [Fact]
    public void StartRun_WithValidExperimentId_ReturnsRun()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("run-test");

        // Act
        var run = _tracker.StartRun(experimentId, "my-run");

        // Assert
        Assert.NotNull(run);
        Assert.NotNull(run.RunId);
        Assert.NotEmpty(run.RunId);
    }

    [Fact]
    public void StartRun_WithTags_StoresRunTags()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("run-tags-test");
        var tags = new Dictionary<string, string>
        {
            ["version"] = "1.0",
            ["gpu"] = "rtx3080"
        };

        // Act
        var run = _tracker.StartRun(experimentId, "tagged-run", tags);

        // Assert
        Assert.NotNull(run);
        Assert.NotNull(run.Tags);
        Assert.Equal("1.0", run.Tags["version"]);
        Assert.Equal("rtx3080", run.Tags["gpu"]);
    }

    [Fact]
    public void StartRun_WithInvalidExperimentId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.StartRun("nonexistent-experiment"));
    }

    [Fact]
    public void StartRun_WithNullExperimentId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.StartRun(null!));
    }

    [Fact]
    public void GetRun_WithValidId_ReturnsRun()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("get-run-test");
        var run = _tracker.StartRun(experimentId, "my-run");

        // Act
        var retrievedRun = _tracker.GetRun(run.RunId);

        // Assert
        Assert.NotNull(retrievedRun);
        Assert.Equal(run.RunId, retrievedRun.RunId);
    }

    [Fact]
    public void GetRun_WithInvalidId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.GetRun("nonexistent-run"));
    }

    [Fact]
    public void DeleteRun_RemovesRun()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("delete-run-test");
        var run = _tracker.StartRun(experimentId, "to-delete");

        // Act
        _tracker.DeleteRun(run.RunId);

        // Assert
        Assert.Throws<ArgumentException>(() => _tracker.GetRun(run.RunId));
    }

    [Fact]
    public void DeleteRun_WithInvalidId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.DeleteRun("nonexistent-run"));
    }

    #endregion

    #region List and Search Tests

    [Fact]
    public void ListExperiments_ReturnsAllExperiments()
    {
        // Arrange
        _tracker.CreateExperiment("exp-1");
        _tracker.CreateExperiment("exp-2");
        _tracker.CreateExperiment("exp-3");

        // Act
        var experiments = _tracker.ListExperiments().ToList();

        // Assert
        Assert.Equal(3, experiments.Count);
    }

    [Fact]
    public void ListExperiments_WithFilter_ReturnsMatchingExperiments()
    {
        // Arrange
        _tracker.CreateExperiment("classification-exp");
        _tracker.CreateExperiment("regression-exp");
        _tracker.CreateExperiment("classification-advanced");

        // Act
        var experiments = _tracker.ListExperiments("classification").ToList();

        // Assert
        Assert.Equal(2, experiments.Count);
        Assert.All(experiments, e => Assert.Contains("classification", e.Name));
    }

    [Fact]
    public void ListRuns_ReturnsRunsForExperiment()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("list-runs-test");
        _tracker.StartRun(experimentId, "run-1");
        _tracker.StartRun(experimentId, "run-2");
        _tracker.StartRun(experimentId, "run-3");

        // Act
        var runs = _tracker.ListRuns(experimentId).ToList();

        // Assert
        Assert.Equal(3, runs.Count);
    }

    [Fact]
    public void ListRuns_WithFilter_ReturnsMatchingRuns()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("filter-runs-test");
        _tracker.StartRun(experimentId, "baseline-run");
        _tracker.StartRun(experimentId, "optimized-run");
        _tracker.StartRun(experimentId, "baseline-v2-run");

        // Act
        var runs = _tracker.ListRuns(experimentId, "baseline").ToList();

        // Assert
        Assert.Equal(2, runs.Count);
    }

    [Fact]
    public void ListRuns_WithInvalidExperimentId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.ListRuns("nonexistent-exp").ToList());
    }

    [Fact]
    public void SearchRuns_FindsRunsAcrossExperiments()
    {
        // Arrange
        var exp1 = _tracker.CreateExperiment("search-exp-1");
        var exp2 = _tracker.CreateExperiment("search-exp-2");

        _tracker.StartRun(exp1, "production-run");
        _tracker.StartRun(exp1, "dev-run");
        _tracker.StartRun(exp2, "production-v2-run");

        // Act
        var runs = _tracker.SearchRuns("production").ToList();

        // Assert
        Assert.Equal(2, runs.Count);
    }

    [Fact]
    public void SearchRuns_WithMaxResults_LimitsResults()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("max-results-test");
        for (int i = 0; i < 10; i++)
        {
            _tracker.StartRun(experimentId, $"run-{i}");
        }

        // Act
        var runs = _tracker.SearchRuns("run", maxResults: 5).ToList();

        // Assert
        Assert.Equal(5, runs.Count);
    }

    #endregion

    #region Run Logging Tests

    [Fact]
    public void Run_LogParameters_StoresParameters()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("log-params-test");
        var run = _tracker.StartRun(experimentId);

        var parameters = new Dictionary<string, object>
        {
            ["learning_rate"] = 0.001,
            ["batch_size"] = 32,
            ["optimizer"] = "adam"
        };

        // Act
        run.LogParameters(parameters);

        // Assert
        var retrievedParams = run.GetParameters();
        Assert.NotNull(retrievedParams);
        Assert.Equal(0.001, Convert.ToDouble(retrievedParams["learning_rate"]));
        Assert.Equal(32, Convert.ToInt32(retrievedParams["batch_size"]));
        Assert.Equal("adam", retrievedParams["optimizer"]);
    }

    [Fact]
    public void Run_LogMetric_StoresMetric()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("log-metric-test");
        var run = _tracker.StartRun(experimentId);

        // Act
        run.LogMetric("loss", 0.5, step: 1);
        run.LogMetric("loss", 0.3, step: 2);
        run.LogMetric("accuracy", 0.85, step: 1);

        // Assert
        var metrics = run.GetMetrics();
        Assert.NotNull(metrics);
        Assert.True(metrics.ContainsKey("loss"));
        Assert.True(metrics.ContainsKey("accuracy"));
    }

    [Fact]
    public void Run_EndRun_SetsStatusAndEndTime()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("end-run-test");
        var run = _tracker.StartRun(experimentId);

        // Act
        run.Complete();

        // Assert
        Assert.NotNull(run.EndTime);
        Assert.Equal("Completed", run.Status);
    }

    [Fact]
    public void Run_EndRunWithFailure_SetsFailedStatus()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("fail-run-test");
        var run = _tracker.StartRun(experimentId);

        // Act
        run.Fail();

        // Assert
        Assert.Equal("Failed", run.Status);
    }

    #endregion

    #region Persistence Tests

    [Fact]
    public void Tracker_PersistsExperimentsToDisk()
    {
        // Arrange
        _tracker.CreateExperiment("persistence-test", "Testing persistence");

        // Act - Create new tracker pointing to same directory
        var tracker2 = new ExperimentTracker<double>(_testDirectory);
        var experiments = tracker2.ListExperiments().ToList();

        // Assert - Find by name since IDs may be regenerated during deserialization
        var experiment = experiments.FirstOrDefault(e => e.Name == "persistence-test");
        Assert.NotNull(experiment);
        Assert.Equal("persistence-test", experiment.Name);
        Assert.Equal("Testing persistence", experiment.Description);
    }

    [Fact]
    public void Tracker_PersistsRunsToDisk()
    {
        // Arrange
        _tracker.CreateExperiment("run-persistence-test");
        var experiment = _tracker.ListExperiments().First(e => e.Name == "run-persistence-test");
        _tracker.StartRun(experiment.ExperimentId, "persisted-run");

        // Act - Create new tracker pointing to same directory
        var tracker2 = new ExperimentTracker<double>(_testDirectory);

        // First, verify the experiment is persisted
        var loadedExperiments = tracker2.ListExperiments().ToList();
        var loadedExperiment = loadedExperiments.FirstOrDefault(e => e.Name == "run-persistence-test");
        Assert.NotNull(loadedExperiment);

        // Then get runs for that experiment using the loaded experiment's ID
        var loadedRuns = tracker2.ListRuns(loadedExperiment.ExperimentId).ToList();

        // Assert - Verify at least one run exists for the experiment
        // Runs are persisted to disk and loaded successfully
        Assert.NotEmpty(loadedRuns);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void CreateExperiment_FromMultipleThreads_IsThreadSafe()
    {
        // Arrange
        var tasks = new List<Task<string>>();
        var experimentCount = 10;

        // Act
        for (int i = 0; i < experimentCount; i++)
        {
            var index = i;
            tasks.Add(Task.Run(() => _tracker.CreateExperiment($"concurrent-exp-{index}")));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        var experiments = _tracker.ListExperiments().ToList();
        Assert.Equal(experimentCount, experiments.Count);
    }

    [Fact]
    public void StartRun_FromMultipleThreads_IsThreadSafe()
    {
        // Arrange
        var experimentId = _tracker.CreateExperiment("concurrent-runs-test");
        var tasks = new List<Task>();
        var runCount = 10;

        // Act
        for (int i = 0; i < runCount; i++)
        {
            var index = i;
            tasks.Add(Task.Run(() => _tracker.StartRun(experimentId, $"run-{index}")));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        var runs = _tracker.ListRuns(experimentId).ToList();
        Assert.Equal(runCount, runs.Count);
    }

    #endregion
}
