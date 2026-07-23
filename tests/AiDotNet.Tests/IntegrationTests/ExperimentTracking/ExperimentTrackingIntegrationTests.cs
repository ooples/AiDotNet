using Xunit;
using AiDotNet.ExperimentTracking;
using AiDotNet.Models;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ExperimentTracking;

/// <summary>
/// Comprehensive integration tests for the ExperimentTracking module.
/// Tests Experiment, ExperimentRun, and ExperimentTracker classes.
/// </summary>
public class ExperimentTrackingIntegrationTests : IDisposable
{
    private readonly string _testStorageDir;

    public ExperimentTrackingIntegrationTests()
    {
        // Create a unique test directory for each test run
        _testStorageDir = Path.Combine(Path.GetTempPath(), $"mlruns_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_testStorageDir);
    }

    public void Dispose()
    {
        // Cleanup test directory
        if (Directory.Exists(_testStorageDir))
        {
            try
            {
                Directory.Delete(_testStorageDir, true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    #region Experiment Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task Experiment_ValidConstruction()
    {
        // Arrange & Act
        var experiment = new Experiment("TestExperiment");

        // Assert
        Assert.NotNull(experiment);
        Assert.Equal("TestExperiment", experiment.Name);
        Assert.NotEmpty(experiment.ExperimentId);
        Assert.Equal("Active", experiment.Status);
        Assert.NotNull(experiment.Tags);
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_WithDescription()
    {
        // Arrange & Act
        var experiment = new Experiment("TestExperiment", "Test description");

        // Assert
        Assert.Equal("Test description", experiment.Description);
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_WithTags()
    {
        // Arrange
        var tags = new Dictionary<string, string>
        {
            { "environment", "test" },
            { "version", "1.0" }
        };

        // Act
        var experiment = new Experiment("TestExperiment", null, tags);

        // Assert
        Assert.Equal(2, experiment.Tags.Count);
        Assert.Equal("test", experiment.Tags["environment"]);
        Assert.Equal("1.0", experiment.Tags["version"]);
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_NullName_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new Experiment(null!));
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_EmptyName_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new Experiment(""));
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_WhitespaceName_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new Experiment("   "));
    }

    #endregion

    #region Experiment Properties Tests

    [Fact(Timeout = 120000)]
    public async Task Experiment_ExperimentId_IsGuid()
    {
        // Arrange & Act
        var experiment = new Experiment("TestExperiment");

        // Assert
        Assert.True(Guid.TryParse(experiment.ExperimentId, out _));
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_CreatedAt_IsSet()
    {
        // Arrange
        var before = DateTime.UtcNow.AddSeconds(-1);

        // Act
        var experiment = new Experiment("TestExperiment");

        // Assert
        Assert.True(experiment.CreatedAt >= before);
        Assert.True(experiment.CreatedAt <= DateTime.UtcNow.AddSeconds(1));
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_LastUpdatedAt_IsInitialized()
    {
        // Arrange & Act
        var experiment = new Experiment("TestExperiment");

        // Assert
        Assert.True(experiment.LastUpdatedAt >= experiment.CreatedAt.AddMilliseconds(-1));
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_Name_CanBeSet()
    {
        // Arrange
        var experiment = new Experiment("Original");

        // Act
        experiment.Name = "Updated";

        // Assert
        Assert.Equal("Updated", experiment.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_Name_SetNull_ThrowsException()
    {
        // Arrange
        var experiment = new Experiment("Original");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => experiment.Name = null!);
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_Description_CanBeSet()
    {
        // Arrange
        var experiment = new Experiment("Test");

        // Act
        experiment.Description = "New description";

        // Assert
        Assert.Equal("New description", experiment.Description);
    }

    #endregion

    #region Experiment Methods Tests

    [Fact(Timeout = 120000)]
    public async Task Experiment_Archive_ChangesStatusToArchived()
    {
        // Arrange
        var experiment = new Experiment("TestExperiment");

        // Act
        experiment.Archive();

        // Assert
        Assert.Equal("Archived", experiment.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_Archive_UpdatesLastUpdatedAt()
    {
        // Arrange
        var experiment = new Experiment("TestExperiment");
        var originalUpdated = experiment.LastUpdatedAt;
        Thread.Sleep(10); // Ensure time difference

        // Act
        experiment.Archive();

        // Assert
        Assert.True(experiment.LastUpdatedAt > originalUpdated);
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_Restore_ChangesStatusToActive()
    {
        // Arrange
        var experiment = new Experiment("TestExperiment");
        experiment.Archive();

        // Act
        experiment.Restore();

        // Assert
        Assert.Equal("Active", experiment.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task Experiment_Restore_UpdatesLastUpdatedAt()
    {
        // Arrange
        var experiment = new Experiment("TestExperiment");
        experiment.Archive();
        var archivedTime = experiment.LastUpdatedAt;
        Thread.Sleep(10); // Ensure time difference

        // Act
        experiment.Restore();

        // Assert
        Assert.True(experiment.LastUpdatedAt > archivedTime);
    }

    #endregion

    #region ExperimentRun Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_ValidConstruction()
    {
        // Arrange & Act
        var run = new ExperimentRun<double>("exp-123");

        // Assert
        Assert.NotNull(run);
        Assert.NotEmpty(run.RunId);
        Assert.Equal("exp-123", run.ExperimentId);
        Assert.Equal("Running", run.Status);
        Assert.Null(run.EndTime);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_WithRunName()
    {
        // Arrange & Act
        var run = new ExperimentRun<double>("exp-123", "my-run");

        // Assert
        Assert.Equal("my-run", run.RunName);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_WithTags()
    {
        // Arrange
        var tags = new Dictionary<string, string>
        {
            { "model", "resnet" },
            { "epochs", "100" }
        };

        // Act
        var run = new ExperimentRun<double>("exp-123", null, tags);

        // Assert
        Assert.Equal(2, run.Tags.Count);
        Assert.Equal("resnet", run.Tags["model"]);
        Assert.Equal("100", run.Tags["epochs"]);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_NullExperimentId_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ExperimentRun<double>(null!));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_RunId_IsGuid()
    {
        // Arrange & Act
        var run = new ExperimentRun<double>("exp-123");

        // Assert
        Assert.True(Guid.TryParse(run.RunId, out _));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_StartTime_IsSet()
    {
        // Arrange
        var before = DateTime.UtcNow.AddSeconds(-1);

        // Act
        var run = new ExperimentRun<double>("exp-123");

        // Assert
        Assert.True(run.StartTime >= before);
    }

    #endregion

    #region ExperimentRun Parameter Logging Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogParameter_StoresParameter()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.LogParameter("learning_rate", 0.001);

        // Assert
        var parameters = run.GetParameters();
        Assert.Single(parameters);
        Assert.Equal(0.001, parameters["learning_rate"]);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogParameter_EmptyKey_ThrowsException()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => run.LogParameter("", 0.001));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogParameters_StoresMultiple()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");
        var parameters = new Dictionary<string, object>
        {
            { "learning_rate", 0.001 },
            { "batch_size", 32 },
            { "optimizer", "adam" }
        };

        // Act
        run.LogParameters(parameters);

        // Assert
        var logged = run.GetParameters();
        Assert.Equal(3, logged.Count);
        Assert.Equal(0.001, logged["learning_rate"]);
        Assert.Equal(32, logged["batch_size"]);
        Assert.Equal("adam", logged["optimizer"]);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogParameters_NullDictionary_ThrowsException()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => run.LogParameters(null!));
    }

    #endregion

    #region ExperimentRun Metric Logging Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogMetric_StoresMetric()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.LogMetric("accuracy", 0.95, step: 10);

        // Assert
        var metrics = run.GetMetrics();
        Assert.Single(metrics);
        Assert.Single(metrics["accuracy"]);
        Assert.Equal(0.95, metrics["accuracy"][0].Value);
        Assert.Equal(10, metrics["accuracy"][0].Step);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogMetric_EmptyKey_ThrowsException()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => run.LogMetric("", 0.5));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogMetric_MultipleSteps()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.LogMetric("loss", 1.0, step: 0);
        run.LogMetric("loss", 0.5, step: 1);
        run.LogMetric("loss", 0.25, step: 2);

        // Assert
        var metrics = run.GetMetrics();
        Assert.Equal(3, metrics["loss"].Count);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogMetrics_StoresMultiple()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");
        var metrics = new Dictionary<string, double>
        {
            { "accuracy", 0.95 },
            { "loss", 0.1 },
            { "f1_score", 0.92 }
        };

        // Act
        run.LogMetrics(metrics, step: 5);

        // Assert
        var logged = run.GetMetrics();
        Assert.Equal(3, logged.Count);
        Assert.Equal(0.95, logged["accuracy"][0].Value);
        Assert.Equal(5, logged["accuracy"][0].Step);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogMetrics_NullDictionary_ThrowsException()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => run.LogMetrics(null!));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_GetLatestMetric_ReturnsLatestByStep()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");
        run.LogMetric("loss", 1.0, step: 0);
        run.LogMetric("loss", 0.5, step: 1);
        run.LogMetric("loss", 0.25, step: 2);

        // Act
        var latest = run.GetLatestMetric("loss");

        // Assert
        Assert.Equal(0.25, latest);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_GetLatestMetric_NonExistentMetric_ReturnsDefault()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        var latest = run.GetLatestMetric("nonexistent");

        // Assert
        Assert.Equal(default(double), latest);
    }

    #endregion

    #region ExperimentRun Artifact Logging Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogArtifact_StoresPath()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.LogArtifact("/path/to/file.txt", "file.txt");

        // Assert
        var artifacts = run.GetArtifacts();
        Assert.Single(artifacts);
        Assert.Equal("file.txt", artifacts[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogArtifact_EmptyPath_ThrowsException()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => run.LogArtifact(""));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_LogArtifact_UsesFileName_WhenNoArtifactPath()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.LogArtifact("/some/path/model.pkl");

        // Assert
        var artifacts = run.GetArtifacts();
        Assert.Single(artifacts);
        Assert.Equal("model.pkl", artifacts[0]);
    }

    #endregion

    #region ExperimentRun Status Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_Complete_ChangesStatusToCompleted()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.Complete();

        // Assert
        Assert.Equal("Completed", run.Status);
        Assert.NotNull(run.EndTime);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_Fail_ChangesStatusToFailed()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.Fail("Out of memory");

        // Assert
        Assert.Equal("Failed", run.Status);
        Assert.NotNull(run.EndTime);
        Assert.Equal("Out of memory", run.GetErrorMessage());
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_Fail_WithoutMessage()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.Fail();

        // Assert
        Assert.Equal("Failed", run.Status);
        Assert.Null(run.GetErrorMessage());
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_GetDuration_Running_ReturnsElapsedTime()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");
        Thread.Sleep(50);

        // Act
        var duration = run.GetDuration();

        // Assert
        Assert.NotNull(duration);
        Assert.True(duration.Value.TotalMilliseconds >= 50);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_GetDuration_Completed_ReturnsTotalDuration()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");
        Thread.Sleep(50);
        run.Complete();

        // Act
        var duration = run.GetDuration();

        // Assert
        Assert.NotNull(duration);
        Assert.True(duration.Value.TotalMilliseconds >= 50);
    }

    #endregion

    #region ExperimentRun Notes Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_AddNote_StoresNote()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act
        run.AddNote("Training started successfully");

        // Assert
        var notes = run.GetNotes();
        Assert.Single(notes);
        Assert.Equal("Training started successfully", notes[0].Note);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_AddNote_EmptyNote_ThrowsException()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => run.AddNote(""));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentRun_GetNotes_OrderedByTimestamp()
    {
        // Arrange
        var run = new ExperimentRun<double>("exp-123");
        run.AddNote("First note");
        Thread.Sleep(10);
        run.AddNote("Second note");

        // Act
        var notes = run.GetNotes();

        // Assert
        Assert.Equal(2, notes.Count);
        Assert.True(notes[0].Timestamp <= notes[1].Timestamp);
    }

    #endregion

    #region ExperimentTracker Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_ValidConstruction()
    {
        // Act
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Assert - no exception means success
        Assert.NotNull(tracker);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_CreatesStorageDirectory()
    {
        // Arrange
        var subDir = Path.Combine(_testStorageDir, "subdir");

        // Act
        var tracker = new ExperimentTracker<double>(subDir);

        // Assert
        Assert.True(Directory.Exists(subDir));
    }

    #endregion

    #region ExperimentTracker CreateExperiment Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_CreateExperiment_ReturnsExperimentId()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act
        var experimentId = tracker.CreateExperiment("MyExperiment");

        // Assert
        Assert.NotEmpty(experimentId);
        Assert.True(Guid.TryParse(experimentId, out _));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_CreateExperiment_WithDescription()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act
        var experimentId = tracker.CreateExperiment("MyExperiment", "Test description");
        var experiment = tracker.GetExperiment(experimentId);

        // Assert
        Assert.Equal("Test description", experiment.Description);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_CreateExperiment_WithTags()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var tags = new Dictionary<string, string> { { "key", "value" } };

        // Act
        var experimentId = tracker.CreateExperiment("MyExperiment", null, tags);
        var experiment = tracker.GetExperiment(experimentId);

        // Assert
        Assert.Equal("value", experiment.Tags["key"]);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_CreateExperiment_EmptyName_ThrowsException()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tracker.CreateExperiment(""));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_CreateExperiment_DuplicateName_ReturnsExistingId()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act
        var id1 = tracker.CreateExperiment("MyExperiment");
        var id2 = tracker.CreateExperiment("MyExperiment");

        // Assert - should return same ID for duplicate name
        Assert.Equal(id1, id2);
    }

    #endregion

    #region ExperimentTracker StartRun Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_StartRun_ReturnsRun()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");

        // Act
        var run = tracker.StartRun(experimentId);

        // Assert
        Assert.NotNull(run);
        Assert.Equal(experimentId, run.ExperimentId);
        Assert.Equal("Running", run.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_StartRun_WithRunName()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");

        // Act
        var run = tracker.StartRun(experimentId, "my-run");

        // Assert
        Assert.Equal("my-run", run.RunName);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_StartRun_NonexistentExperiment_ThrowsException()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tracker.StartRun("nonexistent"));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_StartRun_EmptyExperimentId_ThrowsException()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tracker.StartRun(""));
    }

    #endregion

    #region ExperimentTracker GetExperiment Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_GetExperiment_ReturnsExperiment()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment", "Description");

        // Act
        var experiment = tracker.GetExperiment(experimentId);

        // Assert
        Assert.Equal("MyExperiment", experiment.Name);
        Assert.Equal("Description", experiment.Description);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_GetExperiment_NonexistentId_ThrowsException()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tracker.GetExperiment("nonexistent"));
    }

    #endregion

    #region ExperimentTracker GetRun Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_GetRun_ReturnsRun()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");
        var run = tracker.StartRun(experimentId, "my-run");

        // Act
        var retrievedRun = tracker.GetRun(run.RunId);

        // Assert
        Assert.Equal("my-run", retrievedRun.RunName);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_GetRun_NonexistentId_ThrowsException()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tracker.GetRun("nonexistent"));
    }

    #endregion

    #region ExperimentTracker ListExperiments Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_ListExperiments_ReturnsAll()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        tracker.CreateExperiment("Experiment1");
        tracker.CreateExperiment("Experiment2");
        tracker.CreateExperiment("Experiment3");

        // Act
        var experiments = tracker.ListExperiments().ToList();

        // Assert
        Assert.Equal(3, experiments.Count);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_ListExperiments_WithFilter()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        tracker.CreateExperiment("TestA");
        tracker.CreateExperiment("TestB");
        tracker.CreateExperiment("Other");

        // Act
        var experiments = tracker.ListExperiments("Test").ToList();

        // Assert
        Assert.Equal(2, experiments.Count);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_ListExperiments_OrderedByLastUpdated()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var id1 = tracker.CreateExperiment("Experiment1");
        Thread.Sleep(10);
        var id2 = tracker.CreateExperiment("Experiment2");
        Thread.Sleep(10);
        var id3 = tracker.CreateExperiment("Experiment3");

        // Act
        var experiments = tracker.ListExperiments().ToList();

        // Assert - most recent should be first
        Assert.Equal(id3, experiments[0].ExperimentId);
    }

    #endregion

    #region ExperimentTracker ListRuns Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_ListRuns_ReturnsRunsForExperiment()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");
        tracker.StartRun(experimentId, "run1");
        tracker.StartRun(experimentId, "run2");

        // Act
        var runs = tracker.ListRuns(experimentId).ToList();

        // Assert
        Assert.Equal(2, runs.Count);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_ListRuns_NonexistentExperiment_ThrowsException()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tracker.ListRuns("nonexistent"));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_ListRuns_WithFilter()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");
        tracker.StartRun(experimentId, "test_run");
        tracker.StartRun(experimentId, "other_run");

        // Act
        var runs = tracker.ListRuns(experimentId, "test").ToList();

        // Assert
        Assert.Single(runs);
    }

    #endregion

    #region ExperimentTracker SearchRuns Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_SearchRuns_FindsByRunName()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");
        tracker.StartRun(experimentId, "unique_run_name");
        tracker.StartRun(experimentId, "other_run");

        // Act
        var runs = tracker.SearchRuns("unique").ToList();

        // Assert
        Assert.Single(runs);
        Assert.Equal("unique_run_name", runs[0].RunName);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_SearchRuns_FindsByStatus()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");
        var run1 = tracker.StartRun(experimentId, "run1");
        var run2 = tracker.StartRun(experimentId, "run2");
        ((ExperimentRun<double>)run1).Complete();

        // Act
        var runs = tracker.SearchRuns("Completed").ToList();

        // Assert
        Assert.Single(runs);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_SearchRuns_RespectsMaxResults()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");
        for (int i = 0; i < 10; i++)
        {
            tracker.StartRun(experimentId, $"run{i}");
        }

        // Act
        var runs = tracker.SearchRuns("run", maxResults: 5).ToList();

        // Assert
        Assert.Equal(5, runs.Count);
    }

    #endregion

    #region ExperimentTracker DeleteExperiment Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_DeleteExperiment_RemovesExperiment()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");

        // Act
        tracker.DeleteExperiment(experimentId);

        // Assert
        Assert.Throws<ArgumentException>(() => tracker.GetExperiment(experimentId));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_DeleteExperiment_DeletesAssociatedRuns()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");
        var run = tracker.StartRun(experimentId, "my-run");
        var runId = run.RunId;

        // Act
        tracker.DeleteExperiment(experimentId);

        // Assert
        Assert.Throws<ArgumentException>(() => tracker.GetRun(runId));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_DeleteExperiment_NonexistentId_ThrowsException()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tracker.DeleteExperiment("nonexistent"));
    }

    #endregion

    #region ExperimentTracker DeleteRun Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_DeleteRun_RemovesRun()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("MyExperiment");
        var run = tracker.StartRun(experimentId, "my-run");

        // Act
        tracker.DeleteRun(run.RunId);

        // Assert
        Assert.Throws<ArgumentException>(() => tracker.GetRun(run.RunId));
    }

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_DeleteRun_NonexistentId_ThrowsException()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tracker.DeleteRun("nonexistent"));
    }

    #endregion

    #region ExperimentTracker Float Type Tests

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_Float_Works()
    {
        // Arrange
        var tracker = new ExperimentTracker<float>(_testStorageDir);
        var experimentId = tracker.CreateExperiment("FloatExperiment");

        // Act
        var run = tracker.StartRun(experimentId, "float-run");
        ((ExperimentRun<float>)run).LogMetric("accuracy", 0.95f, step: 1);

        // Assert
        Assert.Equal(0.95f, run.GetLatestMetric("accuracy"));
    }

    #endregion

    #region Path Sanitization Tests (Via ExperimentTracker)

    [Fact(Timeout = 120000)]
    public async Task ExperimentTracker_CreateExperiment_SanitizesName()
    {
        // Arrange
        var tracker = new ExperimentTracker<double>(_testStorageDir);

        // Act - experiment name with potentially dangerous characters
        var experimentId = tracker.CreateExperiment("test..experiment");
        var experiment = tracker.GetExperiment(experimentId);

        // Assert - should create successfully
        Assert.NotNull(experiment);
    }

    #endregion
}
