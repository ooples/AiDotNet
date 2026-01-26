using AiDotNet.CheckpointManagement;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CheckpointManagement;

/// <summary>
/// Integration tests for the CheckpointManagement module.
/// Tests checkpoint lifecycle, auto-checkpointing configuration, and cleanup strategies.
/// </summary>
public class CheckpointManagementIntegrationTests : IDisposable
{
    private const double Tolerance = 1e-10;
    private readonly string _testDirectory;

    public CheckpointManagementIntegrationTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"AiDotNet_CheckpointTests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);
    }

    public void Dispose()
    {
        // Clean up test directory
        if (Directory.Exists(_testDirectory))
        {
            try
            {
                Directory.Delete(_testDirectory, recursive: true);
            }
            catch (IOException)
            {
                // Ignore cleanup errors
            }
        }
    }

    #region CheckpointManagerBase Tests

    [Fact]
    public void CheckpointManager_Constructor_CreatesDirectory()
    {
        var checkpointDir = Path.Combine(_testDirectory, "checkpoints");
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(checkpointDir);

        Assert.True(Directory.Exists(checkpointDir));
    }

    [Fact]
    public void CheckpointManager_GetCheckpointDirectory_ReturnsCorrectPath()
    {
        var checkpointDir = Path.Combine(_testDirectory, "test_checkpoints");
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(checkpointDir);

        var returnedDir = manager.GetCheckpointDirectory();

        Assert.Equal(Path.GetFullPath(checkpointDir), returnedDir);
    }

    [Fact]
    public void CheckpointManager_DefaultDirectory_IsRelativeToCurrentDirectory()
    {
        // When no directory specified, should use ./checkpoints
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>();

        var checkpointDir = manager.GetCheckpointDirectory();

        Assert.Contains("checkpoints", checkpointDir);
    }

    #endregion

    #region Auto-Checkpointing Configuration Tests

    [Fact]
    public void ConfigureAutoCheckpointing_SetsCorrectState()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "auto_checkpoints"));

        manager.ConfigureAutoCheckpointing(
            saveFrequency: 100,
            keepLast: 5,
            saveOnImprovement: true,
            metricName: "loss");

        var state = manager.GetAutoCheckpointState();

        Assert.True(state.IsEnabled);
        Assert.Equal(100, state.SaveFrequency);
        Assert.Equal(5, state.KeepLast);
        Assert.True(state.SaveOnImprovement);
        Assert.Equal("loss", state.MetricName);
        Assert.Equal(0, state.LastSaveStep);
        Assert.Null(state.BestMetricValue);
    }

    [Fact]
    public void ConfigureAutoCheckpointing_ResetsPreviousState()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "auto_checkpoints2"));

        // Configure once
        manager.ConfigureAutoCheckpointing(saveFrequency: 50, keepLast: 3);

        // Update state to simulate usage
        manager.UpdateAutoSaveState(step: 100, metricValue: 0.5);

        // Configure again - should reset state
        manager.ConfigureAutoCheckpointing(saveFrequency: 200, keepLast: 10);

        var state = manager.GetAutoCheckpointState();

        Assert.Equal(200, state.SaveFrequency);
        Assert.Equal(10, state.KeepLast);
        Assert.Equal(0, state.LastSaveStep);
        Assert.Null(state.BestMetricValue);
    }

    [Fact]
    public void GetAutoCheckpointState_WhenNotConfigured_ReturnsDisabled()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "unconfigured_checkpoints"));

        var state = manager.GetAutoCheckpointState();

        Assert.False(state.IsEnabled);
    }

    #endregion

    #region ShouldAutoSaveCheckpoint Tests

    [Fact]
    public void ShouldAutoSaveCheckpoint_WhenNotConfigured_ReturnsFalse()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "should_save1"));

        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 100);

        Assert.False(shouldSave);
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_AtFrequencyInterval_ReturnsTrue()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "should_save2"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5);

        // At step 100, should save (since last save was step 0)
        var shouldSaveAt100 = manager.ShouldAutoSaveCheckpoint(currentStep: 100);

        Assert.True(shouldSaveAt100);
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_BeforeFrequencyInterval_ReturnsFalse()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "should_save3"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5, saveOnImprovement: false);

        // At step 50, should not save (frequency is 100)
        var shouldSaveAt50 = manager.ShouldAutoSaveCheckpoint(currentStep: 50);

        Assert.False(shouldSaveAt50);
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_OnFirstImprovement_ReturnsTrue()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "should_save4"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // First metric value should trigger save
        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 1, metricValue: 0.5);

        Assert.True(shouldSave);
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_OnMetricImprovement_WhenMinimizing_ReturnsTrue()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "should_save5"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Set initial best value
        manager.UpdateAutoSaveState(step: 1, metricValue: 0.5, shouldMinimize: true);

        // Better (lower) value when minimizing
        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 2, metricValue: 0.3, shouldMinimize: true);

        Assert.True(shouldSave);
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_OnMetricImprovement_WhenMaximizing_ReturnsTrue()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "should_save6"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Set initial best value
        manager.UpdateAutoSaveState(step: 1, metricValue: 0.5, shouldMinimize: false);

        // Better (higher) value when maximizing
        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 2, metricValue: 0.8, shouldMinimize: false);

        Assert.True(shouldSave);
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_NoImprovement_WhenMinimizing_ReturnsFalse()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "should_save7"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Set initial best value
        manager.UpdateAutoSaveState(step: 1, metricValue: 0.5, shouldMinimize: true);

        // Worse (higher) value when minimizing
        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 2, metricValue: 0.7, shouldMinimize: true);

        Assert.False(shouldSave);
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_NoImprovement_WhenMaximizing_ReturnsFalse()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "should_save8"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Set initial best value
        manager.UpdateAutoSaveState(step: 1, metricValue: 0.8, shouldMinimize: false);

        // Worse (lower) value when maximizing
        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 2, metricValue: 0.6, shouldMinimize: false);

        Assert.False(shouldSave);
    }

    #endregion

    #region UpdateAutoSaveState Tests

    [Fact]
    public void UpdateAutoSaveState_UpdatesLastSaveStep()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "update_state1"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5);

        manager.UpdateAutoSaveState(step: 150);

        var state = manager.GetAutoCheckpointState();
        Assert.Equal(150, state.LastSaveStep);
    }

    [Fact]
    public void UpdateAutoSaveState_UpdatesBestMetricValue()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "update_state2"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5, saveOnImprovement: true);

        manager.UpdateAutoSaveState(step: 1, metricValue: 0.5);

        var state = manager.GetAutoCheckpointState();
        Assert.Equal(0.5, state.BestMetricValue!.Value, Tolerance);
    }

    [Fact]
    public void UpdateAutoSaveState_UpdatesBestMetric_WhenImprovement_Minimizing()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "update_state3"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Initial value
        manager.UpdateAutoSaveState(step: 1, metricValue: 0.5, shouldMinimize: true);
        // Better value (lower when minimizing)
        manager.UpdateAutoSaveState(step: 2, metricValue: 0.3, shouldMinimize: true);

        var state = manager.GetAutoCheckpointState();
        Assert.Equal(0.3, state.BestMetricValue!.Value, Tolerance);
    }

    [Fact]
    public void UpdateAutoSaveState_DoesNotUpdateBestMetric_WhenNoImprovement_Minimizing()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "update_state4"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Initial value
        manager.UpdateAutoSaveState(step: 1, metricValue: 0.5, shouldMinimize: true);
        // Worse value (higher when minimizing)
        manager.UpdateAutoSaveState(step: 2, metricValue: 0.7, shouldMinimize: true);

        var state = manager.GetAutoCheckpointState();
        Assert.Equal(0.5, state.BestMetricValue!.Value, Tolerance); // Should still be 0.5
    }

    [Fact]
    public void UpdateAutoSaveState_UpdatesBestMetric_WhenImprovement_Maximizing()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "update_state5"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Initial value
        manager.UpdateAutoSaveState(step: 1, metricValue: 0.5, shouldMinimize: false);
        // Better value (higher when maximizing)
        manager.UpdateAutoSaveState(step: 2, metricValue: 0.8, shouldMinimize: false);

        var state = manager.GetAutoCheckpointState();
        Assert.Equal(0.8, state.BestMetricValue!.Value, Tolerance);
    }

    #endregion

    #region AutoCheckpointState Tests

    [Fact]
    public void AutoCheckpointState_ToString_WhenDisabled_ReturnsDisabledMessage()
    {
        var state = new AutoCheckpointState(
            isEnabled: false,
            saveFrequency: 0,
            saveOnImprovement: false,
            keepLast: 0,
            metricName: null,
            lastSaveStep: 0,
            bestMetricValue: null);

        var str = state.ToString();

        Assert.Equal("Auto-checkpointing disabled", str);
    }

    [Fact]
    public void AutoCheckpointState_ToString_WhenEnabled_ReturnsFormattedString()
    {
        var state = new AutoCheckpointState(
            isEnabled: true,
            saveFrequency: 100,
            saveOnImprovement: true,
            keepLast: 5,
            metricName: "loss",
            lastSaveStep: 50,
            bestMetricValue: 0.1234);

        var str = state.ToString();

        Assert.Contains("freq=100", str);
        Assert.Contains("improvement=True", str);
        Assert.Contains("keep=5", str);
        Assert.Contains("last=50", str);
        Assert.Contains("0.1234", str);
    }

    [Fact]
    public void AutoCheckpointState_Properties_AreSetCorrectly()
    {
        var state = new AutoCheckpointState(
            isEnabled: true,
            saveFrequency: 200,
            saveOnImprovement: false,
            keepLast: 10,
            metricName: "accuracy",
            lastSaveStep: 300,
            bestMetricValue: 0.95);

        Assert.True(state.IsEnabled);
        Assert.Equal(200, state.SaveFrequency);
        Assert.False(state.SaveOnImprovement);
        Assert.Equal(10, state.KeepLast);
        Assert.Equal("accuracy", state.MetricName);
        Assert.Equal(300, state.LastSaveStep);
        Assert.Equal(0.95, state.BestMetricValue!.Value, Tolerance);
    }

    #endregion

    #region Path Validation Tests

    [Fact]
    public void CheckpointManager_ValidatesDirectoryPath()
    {
        // Test that manager accepts valid paths within test directory
        var validPath = Path.Combine(_testDirectory, "valid_checkpoints");
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(validPath);

        // Should create the directory and return the full path
        Assert.True(Directory.Exists(validPath));
        Assert.Equal(Path.GetFullPath(validPath), manager.GetCheckpointDirectory());
    }

    [Fact]
    public void CheckpointManager_AllowsNestedDirectory()
    {
        var nestedDir = Path.Combine(_testDirectory, "level1", "level2", "checkpoints");

        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(nestedDir);

        Assert.True(Directory.Exists(nestedDir));
        Assert.Equal(Path.GetFullPath(nestedDir), manager.GetCheckpointDirectory());
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void CheckpointManager_ConcurrentConfigurationUpdates_AreThreadSafe()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "thread_safe1"));

        // Configure initial state
        manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5, saveOnImprovement: true);

        // Concurrent updates
        var tasks = new List<Task>();
        for (int i = 0; i < 100; i++)
        {
            int step = i;
            tasks.Add(Task.Run(() =>
            {
                manager.UpdateAutoSaveState(step: step, metricValue: step * 0.01, shouldMinimize: true);
                _ = manager.ShouldAutoSaveCheckpoint(currentStep: step, metricValue: step * 0.01);
            }));
        }

        // Should complete without exceptions
        Task.WaitAll(tasks.ToArray());

        var state = manager.GetAutoCheckpointState();
        Assert.True(state.IsEnabled);
        Assert.NotNull(state.BestMetricValue);
    }

    [Fact]
    public void CheckpointManager_ConcurrentStateReads_AreThreadSafe()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "thread_safe2"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 50, keepLast: 3);
        manager.UpdateAutoSaveState(step: 100, metricValue: 0.5);

        // Concurrent reads
        var tasks = new List<Task<AutoCheckpointState>>();
        for (int i = 0; i < 50; i++)
        {
            tasks.Add(Task.Run(() => manager.GetAutoCheckpointState()));
        }

        Task.WaitAll(tasks.ToArray());

        // All should return consistent state
        foreach (var task in tasks)
        {
            Assert.True(task.Result.IsEnabled);
            Assert.Equal(50, task.Result.SaveFrequency);
            Assert.Equal(100, task.Result.LastSaveStep);
        }
    }

    #endregion

    #region ListCheckpoints Tests

    [Fact]
    public void ListCheckpoints_WhenEmpty_ReturnsEmptyList()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "empty_list"));

        var checkpoints = manager.ListCheckpoints();

        Assert.Empty(checkpoints);
    }

    #endregion

    #region LoadLatestCheckpoint Tests

    [Fact]
    public void LoadLatestCheckpoint_WhenEmpty_ReturnsNull()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "empty_latest"));

        var latest = manager.LoadLatestCheckpoint();

        Assert.Null(latest);
    }

    #endregion

    #region LoadBestCheckpoint Tests

    [Fact]
    public void LoadBestCheckpoint_WhenEmpty_ReturnsNull()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "empty_best"));

        var best = manager.LoadBestCheckpoint("loss", MetricOptimizationDirection.Minimize);

        Assert.Null(best);
    }

    [Fact]
    public void LoadBestCheckpoint_WhenNoMetric_ReturnsNull()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "no_metric_best"));

        // Even without checkpoints, should not throw
        var best = manager.LoadBestCheckpoint("nonexistent_metric", MetricOptimizationDirection.Maximize);

        Assert.Null(best);
    }

    #endregion

    #region CleanupOldCheckpoints Tests

    [Fact]
    public void CleanupOldCheckpoints_WhenEmpty_ReturnsZero()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "empty_cleanup"));

        var deleted = manager.CleanupOldCheckpoints(keepLast: 5);

        Assert.Equal(0, deleted);
    }

    #endregion

    #region CleanupKeepBest Tests

    [Fact]
    public void CleanupKeepBest_WhenEmpty_ReturnsZero()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "empty_cleanup_best"));

        var deleted = manager.CleanupKeepBest("loss", keepBest: 3, direction: MetricOptimizationDirection.Minimize);

        Assert.Equal(0, deleted);
    }

    #endregion

    #region MetricOptimizationDirection Tests

    [Fact]
    public void MetricOptimizationDirection_Minimize_IsZero()
    {
        Assert.Equal(0, (int)MetricOptimizationDirection.Minimize);
    }

    [Fact]
    public void MetricOptimizationDirection_Maximize_IsOne()
    {
        Assert.Equal(1, (int)MetricOptimizationDirection.Maximize);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void ShouldAutoSaveCheckpoint_WithZeroFrequency_OnlyTriggersOnImprovement()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "zero_freq"));
        manager.ConfigureAutoCheckpointing(
            saveFrequency: 0, // Disabled frequency-based
            keepLast: 5,
            saveOnImprovement: true);

        // At step 1000, should not save (frequency is 0, no metric)
        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 1000);

        Assert.False(shouldSave);

        // With metric, should save (first improvement)
        shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 1001, metricValue: 0.5);

        Assert.True(shouldSave);
    }

    [Fact]
    public void UpdateAutoSaveState_WithNullMetric_OnlyUpdatesStep()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "null_metric"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5, saveOnImprovement: true);

        // Set initial metric
        manager.UpdateAutoSaveState(step: 1, metricValue: 0.5);

        // Update with null metric
        manager.UpdateAutoSaveState(step: 50, metricValue: null);

        var state = manager.GetAutoCheckpointState();
        Assert.Equal(50, state.LastSaveStep);
        Assert.Equal(0.5, state.BestMetricValue!.Value, Tolerance); // Should remain unchanged
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_ExactlyAtFrequencyBoundary_ReturnsTrue()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "exact_boundary"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5, saveOnImprovement: false);

        // At exactly step 100
        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 100);

        Assert.True(shouldSave);
    }

    [Fact]
    public void ShouldAutoSaveCheckpoint_AfterUpdate_UsesPreviousStep()
    {
        var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(
            Path.Combine(_testDirectory, "after_update"));
        manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5, saveOnImprovement: false);

        // Save at step 100
        manager.UpdateAutoSaveState(step: 100);

        // At step 150, should not save (only 50 steps since last save)
        var shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 150);

        Assert.False(shouldSave);

        // At step 200, should save (100 steps since last save)
        shouldSave = manager.ShouldAutoSaveCheckpoint(currentStep: 200);

        Assert.True(shouldSave);
    }

    #endregion
}
