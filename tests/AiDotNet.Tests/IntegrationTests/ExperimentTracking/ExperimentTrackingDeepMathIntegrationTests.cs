using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ExperimentTracking;

/// <summary>
/// Deep integration tests for Experiment (creation, archive/restore, validation,
/// unique IDs, timestamps) and ExperimentRun (parameter logging, metric logging,
/// artifact tracking, status transitions, thread-safe operations, duration).
/// </summary>
public class ExperimentTrackingDeepMathIntegrationTests
{
    // ============================
    // Experiment: Creation Tests
    // ============================

    [Fact]
    public void Experiment_Creation_GeneratesUniqueId()
    {
        var exp = new Experiment("Test Experiment");

        Assert.False(string.IsNullOrWhiteSpace(exp.ExperimentId));
        Assert.True(Guid.TryParse(exp.ExperimentId, out _));
    }

    [Fact]
    public void Experiment_Creation_SetsName()
    {
        var exp = new Experiment("My ML Experiment");
        Assert.Equal("My ML Experiment", exp.Name);
    }

    [Fact]
    public void Experiment_Creation_DefaultStatus_Active()
    {
        var exp = new Experiment("Test");
        Assert.Equal("Active", exp.Status);
    }

    [Fact]
    public void Experiment_Creation_SetsTimestamps()
    {
        var before = DateTime.UtcNow;
        var exp = new Experiment("Test");
        var after = DateTime.UtcNow;

        Assert.InRange(exp.CreatedAt, before, after);
        Assert.InRange(exp.LastUpdatedAt, before, after);
    }

    [Fact]
    public void Experiment_Creation_WithDescription()
    {
        var exp = new Experiment("Test", "A detailed description");
        Assert.Equal("A detailed description", exp.Description);
    }

    [Fact]
    public void Experiment_Creation_WithTags()
    {
        var tags = new Dictionary<string, string> { { "env", "production" }, { "team", "ml" } };
        var exp = new Experiment("Test", tags: tags);

        Assert.Equal(2, exp.Tags.Count);
        Assert.Equal("production", exp.Tags["env"]);
    }

    [Fact]
    public void Experiment_Creation_NullTags_DefaultsToEmpty()
    {
        var exp = new Experiment("Test");
        Assert.NotNull(exp.Tags);
        Assert.Empty(exp.Tags);
    }

    [Fact]
    public void Experiment_MultipleInstances_UniqueIds()
    {
        var ids = new HashSet<string>();
        for (int i = 0; i < 100; i++)
        {
            var exp = new Experiment($"Experiment {i}");
            ids.Add(exp.ExperimentId);
        }

        Assert.Equal(100, ids.Count);
    }

    // ============================
    // Experiment: Name Validation Tests
    // ============================

    [Fact]
    public void Experiment_NullName_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new Experiment(null!));
    }

    [Fact]
    public void Experiment_SetName_Null_Throws()
    {
        var exp = new Experiment("initial");
        Assert.Throws<ArgumentNullException>(() => exp.Name = null!);
    }

    [Fact]
    public void Experiment_SetName_WhitespaceOnly_Throws()
    {
        var exp = new Experiment("initial");
        Assert.Throws<ArgumentException>(() => exp.Name = "   ");
    }

    // ============================
    // Experiment: Archive/Restore Tests
    // ============================

    [Fact]
    public void Experiment_Archive_ChangesStatus()
    {
        var exp = new Experiment("Test");
        Assert.Equal("Active", exp.Status);

        exp.Archive();
        Assert.Equal("Archived", exp.Status);
    }

    [Fact]
    public void Experiment_Restore_ChangesStatusBack()
    {
        var exp = new Experiment("Test");
        exp.Archive();
        Assert.Equal("Archived", exp.Status);

        exp.Restore();
        Assert.Equal("Active", exp.Status);
    }

    [Fact]
    public void Experiment_Archive_UpdatesTimestamp()
    {
        var exp = new Experiment("Test");
        var initialUpdate = exp.LastUpdatedAt;

        Thread.Sleep(10);
        exp.Archive();

        Assert.True(exp.LastUpdatedAt >= initialUpdate);
    }

    [Fact]
    public void Experiment_Touch_UpdatesTimestamp()
    {
        var exp = new Experiment("Test");
        var initialUpdate = exp.LastUpdatedAt;

        Thread.Sleep(10);
        exp.Touch();

        Assert.True(exp.LastUpdatedAt >= initialUpdate);
    }

    // ============================
    // ExperimentRun: Creation Tests
    // ============================

    [Fact]
    public void ExperimentRun_Creation_GeneratesUniqueId()
    {
        var run = new ExperimentRun<double>("exp-1");

        Assert.False(string.IsNullOrWhiteSpace(run.RunId));
        Assert.True(Guid.TryParse(run.RunId, out _));
    }

    [Fact]
    public void ExperimentRun_Creation_DefaultStatus_Running()
    {
        var run = new ExperimentRun<double>("exp-1");
        Assert.Equal("Running", run.Status);
    }

    [Fact]
    public void ExperimentRun_Creation_SetsExperimentId()
    {
        var run = new ExperimentRun<double>("my-experiment");
        Assert.Equal("my-experiment", run.ExperimentId);
    }

    [Fact]
    public void ExperimentRun_Creation_NullExperimentId_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ExperimentRun<double>(null!));
    }

    [Fact]
    public void ExperimentRun_Creation_WithNameAndTags()
    {
        var tags = new Dictionary<string, string> { { "model", "random_forest" } };
        var run = new ExperimentRun<double>("exp-1", "Run #1", tags);

        Assert.Equal("Run #1", run.RunName);
        Assert.True(run.Tags.ContainsKey("model"));
    }

    [Fact]
    public void ExperimentRun_MultipleInstances_UniqueIds()
    {
        var ids = new HashSet<string>();
        for (int i = 0; i < 100; i++)
        {
            var run = new ExperimentRun<double>("exp-1");
            ids.Add(run.RunId);
        }

        Assert.Equal(100, ids.Count);
    }

    // ============================
    // ExperimentRun: Parameter Logging Tests
    // ============================

    [Fact]
    public void LogParameter_SingleParam_Retrievable()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogParameter("learning_rate", 0.01);

        var params2 = run.GetParameters();
        Assert.Equal(0.01, params2["learning_rate"]);
    }

    [Fact]
    public void LogParameter_OverwritesSameKey()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogParameter("lr", 0.01);
        run.LogParameter("lr", 0.001);

        Assert.Equal(0.001, run.GetParameters()["lr"]);
    }

    [Fact]
    public void LogParameter_EmptyKey_Throws()
    {
        var run = new ExperimentRun<double>("exp-1");
        Assert.Throws<ArgumentException>(() => run.LogParameter("", 1.0));
    }

    [Fact]
    public void LogParameters_MultiplePairs_AllRetrievable()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogParameters(new Dictionary<string, object>
        {
            { "lr", 0.01 },
            { "epochs", 100 },
            { "batch_size", 32 }
        });

        var p = run.GetParameters();
        Assert.Equal(3, p.Count);
        Assert.Equal(0.01, p["lr"]);
        Assert.Equal(100, p["epochs"]);
        Assert.Equal(32, p["batch_size"]);
    }

    [Fact]
    public void LogParameters_NullDict_Throws()
    {
        var run = new ExperimentRun<double>("exp-1");
        Assert.Throws<ArgumentNullException>(() => run.LogParameters(null!));
    }

    // ============================
    // ExperimentRun: Metric Logging Tests
    // ============================

    [Fact]
    public void LogMetric_SingleMetric_Retrievable()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogMetric("loss", 0.5, step: 1);

        var metrics = run.GetMetrics();
        Assert.True(metrics.ContainsKey("loss"));
        Assert.Single(metrics["loss"]);
        Assert.Equal(0.5, metrics["loss"][0].Value);
        Assert.Equal(1, metrics["loss"][0].Step);
    }

    [Fact]
    public void LogMetric_MultipleSteps_AllRecorded()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogMetric("loss", 1.0, step: 0);
        run.LogMetric("loss", 0.5, step: 1);
        run.LogMetric("loss", 0.3, step: 2);
        run.LogMetric("loss", 0.1, step: 3);

        var metrics = run.GetMetrics();
        Assert.Equal(4, metrics["loss"].Count);

        // Values should decrease (simulating training convergence)
        for (int i = 1; i < metrics["loss"].Count; i++)
        {
            Assert.True(metrics["loss"][i].Value < metrics["loss"][i - 1].Value);
        }
    }

    [Fact]
    public void LogMetric_EmptyKey_Throws()
    {
        var run = new ExperimentRun<double>("exp-1");
        Assert.Throws<ArgumentException>(() => run.LogMetric("", 1.0));
    }

    [Fact]
    public void GetLatestMetric_ReturnsHighestStep()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogMetric("accuracy", 0.7, step: 1);
        run.LogMetric("accuracy", 0.8, step: 2);
        run.LogMetric("accuracy", 0.85, step: 3);

        var latest = run.GetLatestMetric("accuracy");
        Assert.Equal(0.85, latest);
    }

    [Fact]
    public void GetLatestMetric_UnknownMetric_ReturnsDefault()
    {
        var run = new ExperimentRun<double>("exp-1");
        var result = run.GetLatestMetric("nonexistent");
        Assert.Equal(0.0, result);
    }

    [Fact]
    public void LogMetrics_MultiplePairs_AllRecorded()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogMetrics(new Dictionary<string, double>
        {
            { "loss", 0.5 },
            { "accuracy", 0.85 },
            { "f1_score", 0.82 }
        }, step: 1);

        var metrics = run.GetMetrics();
        Assert.Equal(3, metrics.Count);
        Assert.Equal(0.5, metrics["loss"][0].Value);
        Assert.Equal(0.85, metrics["accuracy"][0].Value);
    }

    // ============================
    // ExperimentRun: Status Transitions
    // ============================

    [Fact]
    public void Complete_ChangesStatusToCompleted()
    {
        var run = new ExperimentRun<double>("exp-1");
        Assert.Equal("Running", run.Status);

        run.Complete();

        Assert.Equal("Completed", run.Status);
        Assert.NotNull(run.EndTime);
    }

    [Fact]
    public void Fail_ChangesStatusToFailed()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.Fail("Out of memory");

        Assert.Equal("Failed", run.Status);
        Assert.NotNull(run.EndTime);
        Assert.Equal("Out of memory", run.GetErrorMessage());
    }

    [Fact]
    public void Fail_WithoutMessage_NoError()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.Fail();

        Assert.Equal("Failed", run.Status);
        Assert.Null(run.GetErrorMessage());
    }

    [Fact]
    public void Fail_WithMessage_AddsNote()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.Fail("Training diverged");

        var notes = run.GetNotes();
        Assert.Single(notes);
        Assert.Contains("Training diverged", notes[0].Note);
    }

    // ============================
    // ExperimentRun: Notes Tests
    // ============================

    [Fact]
    public void AddNote_Retrievable()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.AddNote("Learning rate adjusted");

        var notes = run.GetNotes();
        Assert.Single(notes);
        Assert.Equal("Learning rate adjusted", notes[0].Note);
    }

    [Fact]
    public void AddNote_EmptyNote_Throws()
    {
        var run = new ExperimentRun<double>("exp-1");
        Assert.Throws<ArgumentException>(() => run.AddNote(""));
    }

    [Fact]
    public void AddNote_MultipleNotes_OrderedByTimestamp()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.AddNote("First note");
        Thread.Sleep(10);
        run.AddNote("Second note");

        var notes = run.GetNotes();
        Assert.Equal(2, notes.Count);
        Assert.True(notes[0].Timestamp <= notes[1].Timestamp);
    }

    // ============================
    // ExperimentRun: Artifact Tests
    // ============================

    [Fact]
    public void LogArtifact_SingleFile_Recorded()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogArtifact("/tmp/model.pkl");

        var artifacts = run.GetArtifacts();
        Assert.Single(artifacts);
        Assert.Equal("model.pkl", artifacts[0]);
    }

    [Fact]
    public void LogArtifact_WithCustomPath_UsesCustomPath()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogArtifact("/tmp/model.pkl", "models/v1/model.pkl");

        var artifacts = run.GetArtifacts();
        Assert.Contains("models/v1/model.pkl", artifacts);
    }

    [Fact]
    public void LogArtifact_EmptyPath_Throws()
    {
        var run = new ExperimentRun<double>("exp-1");
        Assert.Throws<ArgumentException>(() => run.LogArtifact(""));
    }

    // ============================
    // ExperimentRun: Duration Tests
    // ============================

    [Fact]
    public void GetDuration_Running_ReturnsElapsedTime()
    {
        var run = new ExperimentRun<double>("exp-1");

        // While running, duration should be non-null and non-negative
        var duration = run.GetDuration();
        Assert.NotNull(duration);
        Assert.True(duration.Value >= TimeSpan.Zero);
    }

    [Fact]
    public void GetDuration_Completed_ReturnsEndMinusStart()
    {
        var run = new ExperimentRun<double>("exp-1");
        Thread.Sleep(50);
        run.Complete();

        var duration = run.GetDuration();
        Assert.NotNull(duration);
        Assert.True(duration.Value.TotalMilliseconds >= 40); // At least ~40ms elapsed
    }

    // ============================
    // ExperimentRun: GetParameters Returns Copy
    // ============================

    [Fact]
    public void GetParameters_ReturnsCopy_ModificationDoesNotAffectOriginal()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogParameter("lr", 0.01);

        var params1 = run.GetParameters();
        params1["lr"] = 999.0;

        var params2 = run.GetParameters();
        Assert.Equal(0.01, params2["lr"]); // Original unchanged
    }

    // ============================
    // ExperimentRun: GetMetrics Returns Copy
    // ============================

    [Fact]
    public void GetMetrics_ReturnsCopy_ModificationDoesNotAffectOriginal()
    {
        var run = new ExperimentRun<double>("exp-1");
        run.LogMetric("loss", 0.5, step: 1);

        var metrics1 = run.GetMetrics();
        metrics1["loss"].Clear();

        var metrics2 = run.GetMetrics();
        Assert.Single(metrics2["loss"]); // Original unchanged
    }

    // ============================
    // Cross-Component Tests
    // ============================

    [Fact]
    public void Experiment_And_Run_FullWorkflow()
    {
        // Create experiment
        var exp = new Experiment("Image Classification", "ResNet fine-tuning");

        // Create runs
        var run1 = new ExperimentRun<double>(exp.ExperimentId, "Baseline");
        var run2 = new ExperimentRun<double>(exp.ExperimentId, "With augmentation");

        // Log parameters
        run1.LogParameter("lr", 0.001);
        run2.LogParameter("lr", 0.0005);

        // Log metrics for convergence
        for (int epoch = 0; epoch < 5; epoch++)
        {
            double loss1 = 1.0 / (epoch + 1);
            double loss2 = 0.8 / (epoch + 1);
            run1.LogMetric("loss", loss1, step: epoch);
            run2.LogMetric("loss", loss2, step: epoch);
        }

        // Complete runs
        run1.Complete();
        run2.Complete();

        // Verify both completed
        Assert.Equal("Completed", run1.Status);
        Assert.Equal("Completed", run2.Status);

        // Verify run2 had lower loss (augmentation helped)
        var loss1Final = run1.GetLatestMetric("loss");
        var loss2Final = run2.GetLatestMetric("loss");

        Assert.NotNull(loss1Final);
        Assert.NotNull(loss2Final);
        Assert.True(loss2Final < loss1Final);
    }
}
