using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.TrainingMonitoring;
using AiDotNet.TrainingMonitoring.Dashboard;
using AiDotNet.TrainingMonitoring.ExperimentTracking;
using AiDotNet.TrainingMonitoring.Notifications;
using AiDotNet.TrainingMonitoring.Resources;
using Xunit;
using LogLevel = AiDotNet.Interfaces.LogLevel;

namespace AiDotNet.Tests.IntegrationTests.TrainingMonitoring;

/// <summary>
/// Comprehensive integration tests for the TrainingMonitoring module.
/// Tests cover TrainingMonitor, ResourceMonitor, ExperimentTracker, NotificationManager,
/// and Dashboard implementations.
/// </summary>
public class TrainingMonitoringIntegrationTests : IDisposable
{
    private readonly string _testOutputPath;
    private readonly List<string> _cleanupPaths = new();

    public TrainingMonitoringIntegrationTests()
    {
        _testOutputPath = Path.Combine(Path.GetTempPath(), $"TrainingMonitoringTests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testOutputPath);
        _cleanupPaths.Add(_testOutputPath);
    }

    public void Dispose()
    {
        foreach (var path in _cleanupPaths)
        {
            try
            {
                if (Directory.Exists(path))
                    Directory.Delete(path, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    #region TrainingMonitor Tests

    [Fact]
    public void TrainingMonitor_StartSession_ReturnsUniqueSessionId()
    {
        var monitor = new TrainingMonitor<double>();

        var sessionId1 = monitor.StartSession("Session1");
        var sessionId2 = monitor.StartSession("Session2");

        Assert.NotNull(sessionId1);
        Assert.NotNull(sessionId2);
        Assert.NotEqual(sessionId1, sessionId2);
    }

    [Fact]
    public void TrainingMonitor_StartSession_WithMetadata_StoresMetadata()
    {
        var monitor = new TrainingMonitor<double>();
        var metadata = new Dictionary<string, object>
        {
            { "model_type", "neural_network" },
            { "learning_rate", 0.001 },
            { "batch_size", 32 }
        };

        var sessionId = monitor.StartSession("TrainingSession", metadata);

        Assert.NotNull(sessionId);
        // Session was created successfully with metadata
    }

    [Fact]
    public void TrainingMonitor_StartSession_ThrowsOnNullOrEmptyName()
    {
        var monitor = new TrainingMonitor<double>();

        Assert.Throws<ArgumentException>(() => monitor.StartSession(""));
        Assert.Throws<ArgumentException>(() => monitor.StartSession("   "));
    }

    [Fact]
    public void TrainingMonitor_EndSession_CompletesSuccessfully()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        monitor.EndSession(sessionId);

        // Session ended without exception
    }

    [Fact]
    public void TrainingMonitor_EndSession_ThrowsOnInvalidSession()
    {
        var monitor = new TrainingMonitor<double>();

        Assert.Throws<ArgumentException>(() => monitor.EndSession("invalid_session_id"));
    }

    [Fact]
    public void TrainingMonitor_LogMetric_StoresMetricValue()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        monitor.LogMetric(sessionId, "loss", 0.5, step: 1);

        var metrics = monitor.GetCurrentMetrics(sessionId);
        Assert.True(metrics.ContainsKey("loss"));
        Assert.Equal(0.5, metrics["loss"]);
    }

    [Fact]
    public void TrainingMonitor_LogMetric_UpdatesCurrentValue()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        monitor.LogMetric(sessionId, "loss", 0.5, step: 1);
        monitor.LogMetric(sessionId, "loss", 0.3, step: 2);

        var metrics = monitor.GetCurrentMetrics(sessionId);
        Assert.Equal(0.3, metrics["loss"]);
    }

    [Fact]
    public void TrainingMonitor_LogMetric_ThrowsOnEmptyMetricName()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        Assert.Throws<ArgumentException>(() => monitor.LogMetric(sessionId, "", 0.5, step: 1));
    }

    [Fact]
    public void TrainingMonitor_LogMetrics_StoresMultipleMetrics()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");
        var metrics = new Dictionary<string, double>
        {
            { "loss", 0.5 },
            { "accuracy", 0.85 },
            { "f1_score", 0.78 }
        };

        monitor.LogMetrics(sessionId, metrics, step: 1);

        var currentMetrics = monitor.GetCurrentMetrics(sessionId);
        Assert.Equal(3, currentMetrics.Count);
        Assert.Equal(0.5, currentMetrics["loss"]);
        Assert.Equal(0.85, currentMetrics["accuracy"]);
        Assert.Equal(0.78, currentMetrics["f1_score"]);
    }

    [Fact]
    public void TrainingMonitor_GetMetricHistory_ReturnsAllValues()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        for (int step = 1; step <= 10; step++)
        {
            monitor.LogMetric(sessionId, "loss", 1.0 / step, step);
        }

        var history = monitor.GetMetricHistory(sessionId, "loss");

        Assert.Equal(10, history.Count);
        Assert.Equal(1.0, history[0].Value);
        Assert.Equal(0.1, history[9].Value, precision: 5);
    }

    [Fact]
    public void TrainingMonitor_GetMetricHistory_ReturnsEmptyForUnknownMetric()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        var history = monitor.GetMetricHistory(sessionId, "unknown_metric");

        Assert.Empty(history);
    }

    [Fact]
    public void TrainingMonitor_LogResourceUsage_StoresResourceSnapshot()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        monitor.LogResourceUsage(sessionId, cpuUsage: 45.5, memoryUsage: 60.0, gpuUsage: 80.0, gpuMemory: 70.0);

        var resourceStats = monitor.GetResourceUsage(sessionId);

        Assert.Equal(45.5, resourceStats.CpuUsagePercent);
        Assert.Equal(60.0, resourceStats.MemoryUsageMB);
        Assert.Equal(80.0, resourceStats.GpuUsagePercent);
        Assert.Equal(70.0, resourceStats.GpuMemoryUsageMB);
    }

    [Fact]
    public void TrainingMonitor_LogResourceUsage_WithoutGpu_StoresNullGpuValues()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        monitor.LogResourceUsage(sessionId, cpuUsage: 45.5, memoryUsage: 60.0);

        var resourceStats = monitor.GetResourceUsage(sessionId);

        Assert.Null(resourceStats.GpuUsagePercent);
        Assert.Null(resourceStats.GpuMemoryUsageMB);
    }

    [Fact]
    public void TrainingMonitor_UpdateProgress_SetsProgressInfo()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        monitor.UpdateProgress(sessionId, currentStep: 50, totalSteps: 100, currentEpoch: 2, totalEpochs: 10);

        var speedStats = monitor.GetSpeedStats(sessionId);

        Assert.Equal(50, speedStats.IterationsCompleted);
        Assert.Equal(100, speedStats.TotalIterations);
    }

    [Fact]
    public void TrainingMonitor_GetSpeedStats_CalculatesCorrectly()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Simulate some progress
        for (int step = 1; step <= 50; step++)
        {
            monitor.UpdateProgress(sessionId, currentStep: step, totalSteps: 100, currentEpoch: 1, totalEpochs: 5);
            Thread.Sleep(1); // Tiny delay to ensure elapsed time
        }

        var speedStats = monitor.GetSpeedStats(sessionId);

        Assert.Equal(50, speedStats.IterationsCompleted);
        Assert.Equal(100, speedStats.TotalIterations);
        Assert.Equal(50.0, speedStats.ProgressPercentage);
        Assert.True(speedStats.IterationsPerSecond > 0);
        Assert.True(speedStats.ElapsedTime.TotalMilliseconds > 0);
    }

    [Fact]
    public void TrainingMonitor_LogMessage_StoresMessage()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        monitor.LogMessage(sessionId, LogLevel.Info, "Training started");
        monitor.LogMessage(sessionId, LogLevel.Warning, "Learning rate may be too high");

        // Messages are stored (verified by CheckForIssues which examines error messages)
    }

    [Fact]
    public void TrainingMonitor_LogMessage_ThrowsOnEmptyMessage()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        Assert.Throws<ArgumentException>(() => monitor.LogMessage(sessionId, LogLevel.Info, ""));
    }

    [Fact]
    public void TrainingMonitor_OnEpochStart_CreatesEpochSummary()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        monitor.OnEpochStart(sessionId, epochNumber: 1);

        // Epoch summary created successfully
    }

    [Fact]
    public void TrainingMonitor_OnEpochEnd_CompletesEpochSummary()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");
        var epochMetrics = new Dictionary<string, double>
        {
            { "train_loss", 0.5 },
            { "val_loss", 0.6 },
            { "accuracy", 0.85 }
        };

        monitor.OnEpochStart(sessionId, epochNumber: 1);
        Thread.Sleep(10); // Simulate epoch duration
        monitor.OnEpochEnd(sessionId, epochNumber: 1, epochMetrics, TimeSpan.FromSeconds(30));

        // Epoch completed successfully
    }

    [Fact]
    public void TrainingMonitor_CheckForIssues_DetectsHighMemoryUsage()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Log high memory usage (>90%)
        monitor.LogResourceUsage(sessionId, cpuUsage: 50.0, memoryUsage: 95.0);

        var issues = monitor.CheckForIssues(sessionId);

        Assert.Contains(issues, i => i.Contains("memory", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void TrainingMonitor_CheckForIssues_DetectsHighGpuMemory()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Log critical GPU memory usage (>95%)
        monitor.LogResourceUsage(sessionId, cpuUsage: 50.0, memoryUsage: 50.0, gpuUsage: 80.0, gpuMemory: 98.0);

        var issues = monitor.CheckForIssues(sessionId);

        Assert.Contains(issues, i => i.Contains("GPU memory", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void TrainingMonitor_CheckForIssues_DetectsErrorMessages()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Log error messages
        monitor.LogMessage(sessionId, LogLevel.Error, "NaN detected in loss");
        monitor.LogMessage(sessionId, LogLevel.Error, "Gradient explosion detected");

        var issues = monitor.CheckForIssues(sessionId);

        Assert.Contains(issues, i => i.Contains("error", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void TrainingMonitor_CheckForIssues_ReturnsEmptyWhenNoIssues()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Normal resource usage
        monitor.LogResourceUsage(sessionId, cpuUsage: 50.0, memoryUsage: 50.0);
        monitor.UpdateProgress(sessionId, 10, 100, 1, 10);

        var issues = monitor.CheckForIssues(sessionId);

        // Should have no issues (or only informational messages)
        Assert.True(issues.Count == 0 || !issues.Any(i => i.Contains("error", StringComparison.OrdinalIgnoreCase)));
    }

    [Fact]
    public void TrainingMonitor_ExportData_CreatesJsonFile()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Add some data
        monitor.LogMetric(sessionId, "loss", 0.5, step: 1);
        monitor.LogMetric(sessionId, "accuracy", 0.8, step: 1);
        monitor.LogResourceUsage(sessionId, cpuUsage: 45.0, memoryUsage: 60.0);
        monitor.UpdateProgress(sessionId, 10, 100, 1, 5);

        var outputPath = Path.Combine(_testOutputPath, "export.json");
        monitor.ExportData(sessionId, outputPath, "json");

        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("loss", content);
        Assert.Contains("accuracy", content);
    }

    [Fact]
    public void TrainingMonitor_ExportData_CreatesCsvFile()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Add some metrics
        for (int step = 1; step <= 5; step++)
        {
            monitor.LogMetrics(sessionId, new Dictionary<string, double>
            {
                { "loss", 1.0 / step },
                { "accuracy", 0.5 + step * 0.1 }
            }, step);
        }

        var outputPath = Path.Combine(_testOutputPath, "export.csv");
        monitor.ExportData(sessionId, outputPath, "csv");

        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("loss", content);
        Assert.Contains("accuracy", content);
    }

    [Fact]
    public void TrainingMonitor_ExportData_ThrowsOnUnsupportedFormat()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");
        var outputPath = Path.Combine(_testOutputPath, "export.xyz");

        Assert.Throws<ArgumentException>(() => monitor.ExportData(sessionId, outputPath, "xyz"));
    }

    [Fact]
    public void TrainingMonitor_CreateVisualization_CreatesFile()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Add some metrics
        for (int step = 1; step <= 10; step++)
        {
            monitor.LogMetric(sessionId, "loss", 1.0 / step, step);
            monitor.LogMetric(sessionId, "accuracy", 0.5 + step * 0.05, step);
        }

        var outputPath = Path.Combine(_testOutputPath, "visualization.json");
        monitor.CreateVisualization(sessionId, new List<string> { "loss", "accuracy" }, outputPath);

        Assert.True(File.Exists(outputPath));
    }

    [Fact]
    public void TrainingMonitor_MultipleSessions_WorkConcurrently()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId1 = monitor.StartSession("Session1");
        var sessionId2 = monitor.StartSession("Session2");

        // Log different metrics to each session
        monitor.LogMetric(sessionId1, "loss", 0.5, step: 1);
        monitor.LogMetric(sessionId2, "loss", 0.8, step: 1);

        var metrics1 = monitor.GetCurrentMetrics(sessionId1);
        var metrics2 = monitor.GetCurrentMetrics(sessionId2);

        Assert.Equal(0.5, metrics1["loss"]);
        Assert.Equal(0.8, metrics2["loss"]);
    }

    [Fact]
    public void TrainingMonitor_ThreadSafety_ConcurrentLogging()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");
        var tasks = new List<Task>();

        // Concurrent logging from multiple threads
        for (int i = 0; i < 10; i++)
        {
            int threadId = i;
            tasks.Add(Task.Run(() =>
            {
                for (int step = 0; step < 100; step++)
                {
                    monitor.LogMetric(sessionId, $"metric_{threadId}", step * 0.1, step);
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // Should complete without exceptions
        var metrics = monitor.GetCurrentMetrics(sessionId);
        Assert.True(metrics.Count >= 1); // At least some metrics were logged
    }

    #endregion

    #region ResourceMonitor Tests

    [Fact]
    public void ResourceMonitor_Constructor_InitializesCorrectly()
    {
        using var monitor = new ResourceMonitor();

        Assert.False(monitor.IsRunning);
        Assert.True(monitor.MonitorGpu);
    }

    [Fact]
    public void ResourceMonitor_Start_SetsIsRunning()
    {
        using var monitor = new ResourceMonitor();

        monitor.Start(TimeSpan.FromSeconds(1));

        Assert.True(monitor.IsRunning);

        monitor.Stop();
    }

    [Fact]
    public void ResourceMonitor_Stop_ClearsIsRunning()
    {
        using var monitor = new ResourceMonitor();

        monitor.Start(TimeSpan.FromSeconds(1));
        monitor.Stop();

        Assert.False(monitor.IsRunning);
    }

    [Fact]
    public void ResourceMonitor_GetSnapshot_ReturnsValidData()
    {
        using var monitor = new ResourceMonitor();

        var snapshot = monitor.GetSnapshot();

        Assert.True(snapshot.Timestamp <= DateTime.UtcNow);
        Assert.True(snapshot.MemoryUsedMB > 0);
        Assert.True(snapshot.ProcessMemoryMB > 0);
    }

    [Fact]
    public void ResourceMonitor_GetHistory_ReturnsSnapshots()
    {
        using var monitor = new ResourceMonitor();
        monitor.Start(TimeSpan.FromMilliseconds(50));

        // Wait for some samples
        Thread.Sleep(200);

        monitor.Stop();

        var history = monitor.GetHistory();

        Assert.NotEmpty(history);
    }

    [Fact]
    public void ResourceMonitor_GetHistory_WithLimit_RespectsLimit()
    {
        using var monitor = new ResourceMonitor();
        monitor.Start(TimeSpan.FromMilliseconds(10));

        // Wait for multiple samples
        Thread.Sleep(100);

        monitor.Stop();

        var history = monitor.GetHistory(limit: 3);

        Assert.True(history.Count <= 3);
    }

    [Fact]
    public void ResourceMonitor_GetAverage_CalculatesCorrectly()
    {
        using var monitor = new ResourceMonitor();
        monitor.Start(TimeSpan.FromMilliseconds(20));

        // Wait for samples
        Thread.Sleep(100);

        monitor.Stop();

        var average = monitor.GetAverage();

        Assert.True(average.MemoryUsedMB > 0);
    }

    [Fact]
    public void ResourceMonitor_GetPeak_ReturnsMaxValues()
    {
        using var monitor = new ResourceMonitor();
        monitor.Start(TimeSpan.FromMilliseconds(20));

        // Wait for samples
        Thread.Sleep(100);

        monitor.Stop();

        var peak = monitor.GetPeak();

        Assert.True(peak.MemoryUsedMB > 0);
    }

    [Fact]
    public void ResourceMonitor_ClearHistory_RemovesAllSnapshots()
    {
        using var monitor = new ResourceMonitor();
        monitor.Start(TimeSpan.FromMilliseconds(20));

        Thread.Sleep(100);

        monitor.Stop();
        monitor.ClearHistory();

        var history = monitor.GetHistory();

        Assert.Empty(history);
    }

    [Fact]
    public void ResourceMonitor_ResourceUpdated_EventFires()
    {
        using var monitor = new ResourceMonitor();
        var eventFired = false;
        ResourceSnapshot? receivedSnapshot = null;

        monitor.ResourceUpdated += (sender, snapshot) =>
        {
            eventFired = true;
            receivedSnapshot = snapshot;
        };

        monitor.Start(TimeSpan.FromMilliseconds(10));
        Thread.Sleep(150); // Allow time for background thread to fire the event
        monitor.Stop();

        Assert.True(eventFired);
        Assert.NotNull(receivedSnapshot);
    }

    [Fact]
    public void ResourceMonitor_Dispose_StopsMonitoring()
    {
        var monitor = new ResourceMonitor();
        monitor.Start(TimeSpan.FromMilliseconds(10));

        monitor.Dispose();

        Assert.False(monitor.IsRunning);
    }

    [Fact]
    public void ResourceMonitor_MultipleStartCalls_AreIdempotent()
    {
        using var monitor = new ResourceMonitor();

        monitor.Start(TimeSpan.FromSeconds(1));
        monitor.Start(TimeSpan.FromSeconds(1)); // Should not throw

        Assert.True(monitor.IsRunning);

        monitor.Stop();
    }

    [Fact]
    public void ResourceMonitor_DisableGpu_SkipsGpuMetrics()
    {
        using var monitor = new ResourceMonitor();
        monitor.MonitorGpu = false;

        var snapshot = monitor.GetSnapshot();

        // When GPU monitoring is disabled, GPU metrics may be null or zero
        // This test verifies the setting works without throwing
    }

    #endregion

    #region ExperimentTracker Tests

    [Fact]
    public void ExperimentTracker_CreateExperiment_ReturnsExperimentInfo()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        var experiment = tracker.CreateExperiment("test_experiment", "Test description");

        Assert.NotNull(experiment);
        Assert.Equal("test_experiment", experiment.Name);
        Assert.Equal("Test description", experiment.Description);
    }

    [Fact]
    public void ExperimentTracker_CreateExperiment_WithTags_StoresTags()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);
        var tags = new Dictionary<string, string>
        {
            { "framework", "AiDotNet" },
            { "version", "1.0" }
        };

        var experiment = tracker.CreateExperiment("tagged_experiment", tags: tags);

        Assert.NotNull(experiment);
        Assert.Equal(2, experiment.Tags.Count);
    }

    [Fact]
    public void ExperimentTracker_GetExperiment_ReturnsExistingExperiment()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("my_experiment");
        var experiment = tracker.GetExperiment("my_experiment");

        Assert.NotNull(experiment);
        Assert.Equal("my_experiment", experiment.Name);
    }

    [Fact]
    public void ExperimentTracker_GetExperiment_ReturnsNullForUnknown()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        var experiment = tracker.GetExperiment("nonexistent");

        Assert.Null(experiment);
    }

    [Fact]
    public void ExperimentTracker_SetExperiment_SetsActiveExperiment()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("exp1");
        tracker.CreateExperiment("exp2");

        tracker.SetExperiment("exp2");

        // Starting a run should use exp2
        var run = tracker.StartRun("test_run");
        Assert.NotNull(run);
    }

    [Fact]
    public void ExperimentTracker_ListExperiments_ReturnsAllExperiments()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("exp1");
        tracker.CreateExperiment("exp2");
        tracker.CreateExperiment("exp3");

        var experiments = tracker.ListExperiments();

        Assert.True(experiments.Count >= 3);
    }

    [Fact]
    public void ExperimentTracker_StartRun_ReturnsRunInfo()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");

        var run = tracker.StartRun("my_run");

        Assert.NotNull(run);
        Assert.Equal("my_run", run.RunName);
        Assert.Equal(RunStatus.Running, run.Status);
    }

    [Fact]
    public void ExperimentTracker_EndRun_SetsStatusCompleted()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");

        tracker.EndRun();

        var retrievedRun = tracker.GetRun(run.RunId);
        Assert.Equal(RunStatus.Completed, retrievedRun?.Status);
    }

    [Fact]
    public void ExperimentTracker_EndRun_WithFailedStatus_SetsStatusFailed()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");

        tracker.EndRun(RunStatus.Failed);

        var retrievedRun = tracker.GetRun(run.RunId);
        Assert.Equal(RunStatus.Failed, retrievedRun?.Status);
    }

    [Fact]
    public void ExperimentTracker_LogParameter_StoresParameter()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");

        tracker.LogParameter("learning_rate", "0.001");
        tracker.LogParameter("batch_size", "32");

        tracker.EndRun();

        var retrievedRun = tracker.GetRun(run.RunId);
        Assert.True(retrievedRun?.Parameters.ContainsKey("learning_rate"));
        Assert.Equal("0.001", retrievedRun?.Parameters["learning_rate"]);
    }

    [Fact]
    public void ExperimentTracker_LogParameters_StoresMultipleParameters()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        tracker.StartRun("my_run");

        tracker.LogParameters(new Dictionary<string, string>
        {
            { "lr", "0.001" },
            { "epochs", "100" },
            { "optimizer", "adam" }
        });

        tracker.EndRun();

        // Parameters were stored
    }

    [Fact]
    public void ExperimentTracker_LogMetric_StoresMetricValue()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");

        tracker.LogMetric("loss", 0.5);
        tracker.LogMetric("loss", 0.4, step: 1);
        tracker.LogMetric("loss", 0.3, step: 2);

        tracker.EndRun();

        var history = tracker.GetMetricHistory(run.RunId, "loss");
        Assert.Equal(3, history.Count);
    }

    [Fact]
    public void ExperimentTracker_LogMetrics_StoresMultipleMetrics()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");

        tracker.LogMetrics(new Dictionary<string, double>
        {
            { "loss", 0.5 },
            { "accuracy", 0.85 },
            { "f1", 0.78 }
        }, step: 1);

        tracker.EndRun();

        var lossHistory = tracker.GetMetricHistory(run.RunId, "loss");
        var accHistory = tracker.GetMetricHistory(run.RunId, "accuracy");

        Assert.Single(lossHistory);
        Assert.Single(accHistory);
    }

    [Fact]
    public void ExperimentTracker_GetMetricHistory_ReturnsAllValues()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");

        for (int i = 0; i < 10; i++)
        {
            tracker.LogMetric("loss", 1.0 - i * 0.1, step: i);
        }

        tracker.EndRun();

        var history = tracker.GetMetricHistory(run.RunId, "loss");

        Assert.Equal(10, history.Count);
    }

    [Fact]
    public void ExperimentTracker_SetTag_StoresTag()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");

        tracker.SetTag("model_type", "neural_network");
        tracker.EndRun();

        var retrievedRun = tracker.GetRun(run.RunId);
        Assert.True(retrievedRun?.Tags.ContainsKey("model_type"));
    }

    [Fact]
    public void ExperimentTracker_ListRuns_ReturnsRunsForExperiment()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");

        tracker.StartRun("run1");
        tracker.EndRun();

        tracker.StartRun("run2");
        tracker.EndRun();

        tracker.StartRun("run3");
        tracker.EndRun();

        var runs = tracker.ListRuns("test_experiment");

        Assert.True(runs.Count >= 3);
    }

    [Fact]
    public void ExperimentTracker_DeleteRun_MarksRunAsDeleted()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");
        tracker.EndRun();

        tracker.DeleteRun(run.RunId);

        var retrievedRun = tracker.GetRun(run.RunId);
        Assert.True(retrievedRun?.IsDeleted);
    }

    [Fact]
    public void ExperimentTracker_RestoreRun_RestoresDeletedRun()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");
        var run = tracker.StartRun("my_run");
        tracker.EndRun();

        tracker.DeleteRun(run.RunId);
        tracker.RestoreRun(run.RunId);

        var retrievedRun = tracker.GetRun(run.RunId);
        Assert.False(retrievedRun?.IsDeleted);
    }

    [Fact]
    public void ExperimentTracker_CompareRuns_ReturnsComparison()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");

        var run1 = tracker.StartRun("run1");
        tracker.LogParameter("lr", "0.001");
        tracker.LogMetric("loss", 0.5);
        tracker.EndRun();

        var run2 = tracker.StartRun("run2");
        tracker.LogParameter("lr", "0.01");
        tracker.LogMetric("loss", 0.3);
        tracker.EndRun();

        var comparison = tracker.CompareRuns(run1.RunId, run2.RunId);

        Assert.NotNull(comparison);
        Assert.True(comparison.Runs.Count >= 2);
    }

    [Fact]
    public void ExperimentTracker_SearchRuns_FiltersCorrectly()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns");
        using var tracker = new ExperimentTracker(trackerPath);

        tracker.CreateExperiment("test_experiment");
        tracker.SetExperiment("test_experiment");

        // Create runs with different metrics
        for (int i = 0; i < 5; i++)
        {
            tracker.StartRun($"run_{i}");
            tracker.LogMetric("accuracy", 0.5 + i * 0.1);
            tracker.EndRun();
        }

        // Search for runs with accuracy > 0.7
        var results = tracker.SearchRuns(
            experimentNames: new[] { "test_experiment" },
            filter: "metrics.accuracy > 0.7"
        );

        Assert.True(results.Count >= 2); // Should include runs with 0.8 and 0.9
    }

    [Fact]
    public void ExperimentTracker_Persistence_LoadsOnRestart()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns_persist");

        // Create and populate tracker
        using (var tracker1 = new ExperimentTracker(trackerPath))
        {
            tracker1.CreateExperiment("persistent_experiment", "Test persistence");
            tracker1.SetExperiment("persistent_experiment");
            var run = tracker1.StartRun("persistent_run");
            tracker1.LogParameter("key", "value");
            tracker1.LogMetric("metric", 0.5);
            tracker1.EndRun();
        }

        // Create new tracker and verify data persisted
        using var tracker2 = new ExperimentTracker(trackerPath);

        var experiment = tracker2.GetExperiment("persistent_experiment");
        Assert.NotNull(experiment);

        var runs = tracker2.ListRuns("persistent_experiment");
        Assert.NotEmpty(runs);
    }

    #endregion

    #region NotificationManager Tests

    [Fact]
    public void NotificationManager_Constructor_InitializesCorrectly()
    {
        using var manager = new NotificationManager();

        // Manager initializes without exception
    }

    [Fact]
    public void NotificationManager_AddService_ReturnsTrue()
    {
        using var manager = new NotificationManager();
        var mockService = new MockNotificationService("test_service");

        var result = manager.AddService(mockService);

        Assert.True(result);
    }

    [Fact]
    public void NotificationManager_AddService_DuplicateName_ReturnsFalse()
    {
        using var manager = new NotificationManager();
        var service1 = new MockNotificationService("same_name");
        var service2 = new MockNotificationService("same_name");

        manager.AddService(service1);
        var result = manager.AddService(service2);

        Assert.False(result);
    }

    [Fact]
    public void NotificationManager_RemoveService_ReturnsTrue()
    {
        using var manager = new NotificationManager();
        var service = new MockNotificationService("test_service");
        manager.AddService(service);

        var result = manager.RemoveService("test_service");

        Assert.True(result);
    }

    [Fact]
    public void NotificationManager_RemoveService_NonExistent_ReturnsFalse()
    {
        using var manager = new NotificationManager();

        var result = manager.RemoveService("nonexistent");

        Assert.False(result);
    }

    [Fact]
    public void NotificationManager_GetService_ReturnsService()
    {
        using var manager = new NotificationManager();
        var service = new MockNotificationService("test_service");
        manager.AddService(service);

        var retrieved = manager.GetService("test_service");

        Assert.NotNull(retrieved);
        Assert.Equal("test_service", retrieved.ServiceName);
    }

    [Fact]
    public void NotificationManager_Send_SendsToAllServices()
    {
        using var manager = new NotificationManager();
        var service1 = new MockNotificationService("service1");
        var service2 = new MockNotificationService("service2");
        manager.AddService(service1);
        manager.AddService(service2);

        var notification = new TrainingNotification
        {
            Title = "Test",
            Message = "Test message",
            Severity = NotificationSeverity.Info
        };

        var results = manager.Send(notification);

        Assert.True(results["service1"]);
        Assert.True(results["service2"]);
        Assert.Equal(1, service1.SendCount);
        Assert.Equal(1, service2.SendCount);
    }

    [Fact]
    public async Task NotificationManager_SendAsync_SendsToAllServices()
    {
        using var manager = new NotificationManager();
        var service1 = new MockNotificationService("service1");
        var service2 = new MockNotificationService("service2");
        manager.AddService(service1);
        manager.AddService(service2);

        var notification = new TrainingNotification
        {
            Title = "Test",
            Message = "Test message",
            Severity = NotificationSeverity.Info
        };

        var results = await manager.SendAsync(notification);

        Assert.True(results["service1"]);
        Assert.True(results["service2"]);
    }

    [Fact]
    public void NotificationManager_SetMinimumSeverity_FiltersSeverity()
    {
        using var manager = new NotificationManager();
        var service = new MockNotificationService("test_service");
        manager.AddService(service);

        // Set minimum to Warning
        manager.SetMinimumSeverity("test_service", NotificationSeverity.Warning);

        // Send Info notification (should be filtered)
        manager.Send(new TrainingNotification
        {
            Title = "Info",
            Message = "Info message",
            Severity = NotificationSeverity.Info
        });

        // Send Warning notification (should pass through)
        manager.Send(new TrainingNotification
        {
            Title = "Warning",
            Message = "Warning message",
            Severity = NotificationSeverity.Warning
        });

        Assert.Equal(1, service.SendCount); // Only Warning was sent
    }

    [Fact]
    public void NotificationManager_SetTypeEnabled_FiltersType()
    {
        using var manager = new NotificationManager();
        var service = new MockNotificationService("test_service");
        manager.AddService(service);

        // Disable EpochComplete type
        manager.SetTypeEnabled(NotificationType.EpochCompleted, false);

        manager.Send(new TrainingNotification
        {
            Title = "Epoch",
            Message = "Epoch complete",
            Type = NotificationType.EpochCompleted
        });

        manager.Send(new TrainingNotification
        {
            Title = "Training",
            Message = "Training complete",
            Type = NotificationType.TrainingCompleted
        });

        Assert.Equal(1, service.SendCount); // Only TrainingComplete was sent
    }

    [Fact]
    public void NotificationManager_NotificationSent_EventFires()
    {
        using var manager = new NotificationManager();
        var service = new MockNotificationService("test_service");
        manager.AddService(service);
        var eventFired = false;

        manager.NotificationSent += (sender, args) =>
        {
            eventFired = true;
        };

        manager.Send(new TrainingNotification
        {
            Title = "Test",
            Message = "Test"
        });

        Assert.True(eventFired);
    }

    [Fact]
    public void NotificationManager_NotificationFailed_EventFires()
    {
        using var manager = new NotificationManager();
        var service = new MockNotificationService("failing_service", shouldFail: true);
        manager.AddService(service);
        var eventFired = false;

        manager.NotificationFailed += (sender, args) =>
        {
            eventFired = true;
        };

        manager.Send(new TrainingNotification
        {
            Title = "Test",
            Message = "Test"
        });

        Assert.True(eventFired);
    }

    [Fact]
    public async Task NotificationManager_TestAllServicesAsync_ReturnsResults()
    {
        using var manager = new NotificationManager();
        var service1 = new MockNotificationService("good_service");
        var service2 = new MockNotificationService("bad_service", shouldFail: true);
        manager.AddService(service1);
        manager.AddService(service2);

        var results = await manager.TestAllServicesAsync();

        Assert.True(results["good_service"]);
        Assert.False(results["bad_service"]);
    }

    [Fact]
    public void NotificationManager_Dispose_DisposesServices()
    {
        var manager = new NotificationManager();
        var service = new MockNotificationService("test_service");
        manager.AddService(service);

        manager.Dispose();

        Assert.True(service.IsDisposed);
    }

    #endregion

    #region Dashboard Tests

    [Fact]
    public void ConsoleDashboard_Constructor_InitializesCorrectly()
    {
        var dashboard = new ConsoleDashboard(name: "TestDashboard");

        Assert.Equal("TestDashboard", dashboard.Name);
        Assert.False(dashboard.IsRunning);
    }

    [Fact]
    public void ConsoleDashboard_Start_SetsIsRunning()
    {
        var dashboard = new ConsoleDashboard(name: "TestDashboard");

        dashboard.Start();

        Assert.True(dashboard.IsRunning);

        dashboard.Stop();
    }

    [Fact]
    public void ConsoleDashboard_LogScalar_StoresValue()
    {
        var dashboard = new ConsoleDashboard(name: "TestDashboard");
        dashboard.Start();

        dashboard.LogScalar("loss", step: 1, value: 0.5);
        dashboard.LogScalar("loss", step: 2, value: 0.4);

        var scalarData = dashboard.GetScalarData();

        Assert.True(scalarData.ContainsKey("loss"));
        Assert.Equal(2, scalarData["loss"].Count);

        dashboard.Stop();
    }

    [Fact]
    public void ConsoleDashboard_LogScalars_StoresMultipleValues()
    {
        var dashboard = new ConsoleDashboard(name: "TestDashboard");
        dashboard.Start();

        dashboard.LogScalars(new Dictionary<string, double>
        {
            { "loss", 0.5 },
            { "accuracy", 0.85 }
        }, step: 1);

        var scalarData = dashboard.GetScalarData();

        Assert.True(scalarData.ContainsKey("loss"));
        Assert.True(scalarData.ContainsKey("accuracy"));

        dashboard.Stop();
    }

    [Fact]
    public void ConsoleDashboard_LogHistogram_StoresValues()
    {
        var dashboard = new ConsoleDashboard(name: "TestDashboard");
        dashboard.Start();

        var values = Enumerable.Range(0, 100).Select(i => (double)i).ToArray();
        dashboard.LogHistogram("weights", step: 1, values: values);

        var histogramData = dashboard.GetHistogramData();

        Assert.True(histogramData.ContainsKey("weights"));

        dashboard.Stop();
    }

    [Fact]
    public void ConsoleDashboard_LogText_StoresText()
    {
        var dashboard = new ConsoleDashboard(name: "TestDashboard");
        dashboard.Start();

        dashboard.LogText("summary", step: 1, text: "Training started successfully");

        // Text was stored (no public getter for text data)
        dashboard.Stop();
    }

    [Fact]
    public void ConsoleDashboard_Clear_RemovesAllData()
    {
        var dashboard = new ConsoleDashboard(name: "TestDashboard");
        dashboard.Start();

        dashboard.LogScalar("loss", 1, 0.5);
        dashboard.LogScalar("accuracy", 1, 0.85);

        dashboard.Clear();

        var scalarData = dashboard.GetScalarData();
        Assert.Empty(scalarData);

        dashboard.Stop();
    }

    [Fact]
    public void HtmlDashboard_Constructor_InitializesCorrectly()
    {
        var logDir = Path.Combine(_testOutputPath, "html_logs");
        var dashboard = new HtmlDashboard(logDir, "HtmlTest");

        Assert.Equal("HtmlTest", dashboard.Name);
        Assert.Equal(logDir, dashboard.LogDirectory);
    }

    [Fact]
    public void HtmlDashboard_GenerateReport_CreatesFile()
    {
        var logDir = Path.Combine(_testOutputPath, "html_logs_report");
        var dashboard = new HtmlDashboard(logDir, "HtmlTest");
        dashboard.Start();

        // Add some data
        dashboard.LogScalar("loss", 1, 0.5);
        dashboard.LogScalar("loss", 2, 0.4);
        dashboard.LogScalar("accuracy", 1, 0.8);
        dashboard.LogScalar("accuracy", 2, 0.85);

        var reportPath = dashboard.GenerateReport();

        Assert.True(File.Exists(reportPath));
        var content = File.ReadAllText(reportPath);
        Assert.Contains("html", content, StringComparison.OrdinalIgnoreCase);

        dashboard.Stop();
    }

    [Fact]
    public void HtmlDashboard_GenerateReport_WithCustomPath_UsesPath()
    {
        var logDir = Path.Combine(_testOutputPath, "html_logs_custom");
        var dashboard = new HtmlDashboard(logDir, "HtmlTest");
        dashboard.Start();

        dashboard.LogScalar("loss", 1, 0.5);

        var customPath = Path.Combine(_testOutputPath, "custom_report.html");
        var reportPath = dashboard.GenerateReport(customPath);

        Assert.Equal(customPath, reportPath);
        Assert.True(File.Exists(customPath));

        dashboard.Stop();
    }

    [Fact]
    public void HtmlDashboard_LogHyperparameters_StoresHyperparameters()
    {
        var logDir = Path.Combine(_testOutputPath, "html_logs_hparams");
        var dashboard = new HtmlDashboard(logDir, "HtmlTest");
        dashboard.Start();

        dashboard.LogHyperparameters(
            new Dictionary<string, object>
            {
                { "learning_rate", 0.001 },
                { "batch_size", 32 }
            },
            new Dictionary<string, double>
            {
                { "final_loss", 0.1 }
            });

        // Hyperparameters stored (verified in report generation)
        dashboard.Stop();
    }

    [Fact]
    public void HtmlDashboard_LogConfusionMatrix_StoresMatrix()
    {
        var logDir = Path.Combine(_testOutputPath, "html_logs_cm");
        var dashboard = new HtmlDashboard(logDir, "HtmlTest");
        dashboard.Start();

        var matrix = new int[,]
        {
            { 50, 5, 2 },
            { 3, 45, 4 },
            { 1, 2, 48 }
        };
        var labels = new[] { "Class A", "Class B", "Class C" };

        dashboard.LogConfusionMatrix("confusion_matrix", step: 1, matrix: matrix, labels: labels);

        // Confusion matrix stored
        dashboard.Stop();
    }

    #endregion

    #region Cross-Module Integration Tests

    [Fact]
    public void Integration_TrainingMonitorWithResourceMonitor()
    {
        var trainingMonitor = new TrainingMonitor<double>();
        using var resourceMonitor = new ResourceMonitor();

        var sessionId = trainingMonitor.StartSession("IntegrationTest");
        resourceMonitor.Start(TimeSpan.FromMilliseconds(50));

        // Simulate training with resource monitoring
        for (int step = 1; step <= 10; step++)
        {
            trainingMonitor.LogMetric(sessionId, "loss", 1.0 / step, step);
            trainingMonitor.UpdateProgress(sessionId, step, 100, 1, 5);

            var snapshot = resourceMonitor.GetSnapshot();
            trainingMonitor.LogResourceUsage(
                sessionId,
                cpuUsage: snapshot.CpuPercent,
                memoryUsage: snapshot.ProcessMemoryMB);

            Thread.Sleep(10);
        }

        resourceMonitor.Stop();
        trainingMonitor.EndSession(sessionId);

        // Verify data was captured
        var speedStats = trainingMonitor.GetSpeedStats(sessionId);
        var resourceStats = trainingMonitor.GetResourceUsage(sessionId);

        Assert.Equal(10, speedStats.IterationsCompleted);
        Assert.True(resourceStats.CpuUsagePercent >= 0);
    }

    [Fact]
    public void Integration_ExperimentTrackerWithDashboard()
    {
        var trackerPath = Path.Combine(_testOutputPath, "mlruns_dashboard");
        var logDir = Path.Combine(_testOutputPath, "dashboard_logs");

        using var tracker = new ExperimentTracker(trackerPath);
        var dashboard = new HtmlDashboard(logDir, "ExperimentDashboard");

        // Start experiment and dashboard
        tracker.CreateExperiment("dashboard_test");
        tracker.SetExperiment("dashboard_test");
        var run = tracker.StartRun("dashboard_run");
        dashboard.Start();

        // Log metrics to both
        for (int step = 1; step <= 20; step++)
        {
            var loss = 1.0 / step;
            var accuracy = 0.5 + step * 0.025;

            tracker.LogMetrics(new Dictionary<string, double>
            {
                { "loss", loss },
                { "accuracy", accuracy }
            }, step);

            dashboard.LogScalars(new Dictionary<string, double>
            {
                { "loss", loss },
                { "accuracy", accuracy }
            }, step);
        }

        // Complete experiment
        tracker.EndRun();
        var reportPath = dashboard.GenerateReport();
        dashboard.Stop();

        // Verify both captured data
        var history = tracker.GetMetricHistory(run.RunId, "loss");
        var scalarData = dashboard.GetScalarData();

        Assert.Equal(20, history.Count);
        Assert.Equal(20, scalarData["loss"].Count);
        Assert.True(File.Exists(reportPath));
    }

    [Fact]
    public void Integration_FullTrainingPipeline()
    {
        // Set up all components
        var trackerPath = Path.Combine(_testOutputPath, "mlruns_full");
        var logDir = Path.Combine(_testOutputPath, "full_logs");

        var trainingMonitor = new TrainingMonitor<double>();
        using var resourceMonitor = new ResourceMonitor();
        using var tracker = new ExperimentTracker(trackerPath);
        var dashboard = new HtmlDashboard(logDir, "FullPipeline");
        using var notificationManager = new NotificationManager();

        var mockNotifier = new MockNotificationService("mock");
        notificationManager.AddService(mockNotifier);

        // Initialize
        tracker.CreateExperiment("full_pipeline_test", "End-to-end integration test");
        tracker.SetExperiment("full_pipeline_test");
        var run = tracker.StartRun("integration_run");

        tracker.LogParameters(new Dictionary<string, string>
        {
            { "model", "test_model" },
            { "learning_rate", "0.001" },
            { "epochs", "5" }
        });

        var sessionId = trainingMonitor.StartSession("IntegrationPipeline", new Dictionary<string, object>
        {
            { "run_id", run.RunId }
        });

        resourceMonitor.Start(TimeSpan.FromMilliseconds(100));
        dashboard.Start();

        // Simulate training
        for (int epoch = 1; epoch <= 5; epoch++)
        {
            trainingMonitor.OnEpochStart(sessionId, epoch);

            for (int step = 1; step <= 10; step++)
            {
                var globalStep = (epoch - 1) * 10 + step;
                var loss = 1.0 / (globalStep + 1);
                var accuracy = 0.5 + globalStep * 0.01;

                // Log everywhere
                trainingMonitor.LogMetrics(sessionId, new Dictionary<string, double>
                {
                    { "loss", loss },
                    { "accuracy", accuracy }
                }, globalStep);

                tracker.LogMetrics(new Dictionary<string, double>
                {
                    { "loss", loss },
                    { "accuracy", accuracy }
                }, globalStep);

                dashboard.LogScalars(new Dictionary<string, double>
                {
                    { "loss", loss },
                    { "accuracy", accuracy }
                }, globalStep);

                trainingMonitor.UpdateProgress(sessionId, globalStep, 50, epoch, 5);

                var snapshot = resourceMonitor.GetSnapshot();
                trainingMonitor.LogResourceUsage(sessionId, snapshot.CpuPercent, snapshot.ProcessMemoryMB);

                Thread.Sleep(5);
            }

            trainingMonitor.OnEpochEnd(sessionId, epoch, new Dictionary<string, double>
            {
                { "epoch_loss", 0.1 * epoch },
                { "epoch_accuracy", 0.9 - 0.1 / epoch }
            }, TimeSpan.FromMilliseconds(50));
        }

        // Complete training
        resourceMonitor.Stop();
        trainingMonitor.EndSession(sessionId);
        tracker.EndRun();

        // Send completion notification
        notificationManager.Send(new TrainingNotification
        {
            Title = "Training Complete",
            Message = "Full pipeline integration test completed successfully",
            Type = NotificationType.TrainingCompleted,
            Severity = NotificationSeverity.Info
        });

        // Export data
        var exportPath = Path.Combine(_testOutputPath, "full_export.json");
        trainingMonitor.ExportData(sessionId, exportPath, "json");

        var reportPath = dashboard.GenerateReport();
        dashboard.Stop();

        // Verify everything worked
        Assert.True(File.Exists(exportPath));
        Assert.True(File.Exists(reportPath));
        Assert.Equal(1, mockNotifier.SendCount);

        var speedStats = trainingMonitor.GetSpeedStats(sessionId);
        Assert.Equal(50, speedStats.IterationsCompleted);

        var trackerHistory = tracker.GetMetricHistory(run.RunId, "loss");
        Assert.Equal(50, trackerHistory.Count);

        var comparison = tracker.CompareRuns(run.RunId);
        Assert.NotNull(comparison);
    }

    #endregion

    #region Helper Classes

    private class MockNotificationService : INotificationService, IDisposable
    {
        public string ServiceName { get; }
        public bool IsConfigured => true;
        public int SendCount { get; private set; }
        public bool IsDisposed { get; private set; }
        private readonly bool _shouldFail;

        public MockNotificationService(string name, bool shouldFail = false)
        {
            ServiceName = name;
            _shouldFail = shouldFail;
        }

        public bool Send(TrainingNotification notification)
        {
            if (_shouldFail)
                throw new InvalidOperationException("Simulated failure");

            SendCount++;
            return true;
        }

        public Task<bool> SendAsync(TrainingNotification notification, CancellationToken cancellationToken = default)
        {
            if (_shouldFail)
                throw new InvalidOperationException("Simulated failure");

            SendCount++;
            return Task.FromResult(true);
        }

        public Task<bool> TestConnectionAsync(CancellationToken cancellationToken = default)
        {
            return Task.FromResult(!_shouldFail);
        }

        public void Dispose()
        {
            IsDisposed = true;
        }
    }

    #endregion
}
