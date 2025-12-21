using AiDotNet.Interfaces;
using AiDotNet.TrainingMonitoring;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TrainingInfrastructure;

/// <summary>
/// Unit tests for TrainingMonitor training progress tracking.
/// </summary>
public class TrainingMonitorTests : IDisposable
{
    private readonly TrainingMonitor<double> _monitor;
    private readonly string _testDirectory;

    public TrainingMonitorTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"training_monitor_tests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);
        _monitor = new TrainingMonitor<double>();
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

    #region Session Management Tests

    [Fact]
    public void StartSession_WithValidName_ReturnsSessionId()
    {
        // Arrange & Act
        var sessionId = _monitor.StartSession("test-session");

        // Assert
        Assert.NotNull(sessionId);
        Assert.NotEmpty(sessionId);
    }

    [Fact]
    public void StartSession_WithMetadata_StoresMetadata()
    {
        // Arrange
        var metadata = new Dictionary<string, object>
        {
            ["model_type"] = "neural-network",
            ["learning_rate"] = 0.001
        };

        // Act
        var sessionId = _monitor.StartSession("test-session", metadata);

        // Assert
        Assert.NotNull(sessionId);
    }

    [Fact]
    public void StartSession_WithNullName_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _monitor.StartSession(null!));
    }

    [Fact]
    public void StartSession_WithEmptyName_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _monitor.StartSession(""));
    }

    [Fact]
    public void EndSession_WithValidId_CompletesSession()
    {
        // Arrange
        var sessionId = _monitor.StartSession("test-session");

        // Act
        _monitor.EndSession(sessionId);

        // Assert - Session should still be accessible after ending
        var metrics = _monitor.GetCurrentMetrics(sessionId);
        Assert.NotNull(metrics);
    }

    [Fact]
    public void EndSession_WithInvalidId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _monitor.EndSession("nonexistent-session"));
    }

    #endregion

    #region Metric Logging Tests

    [Fact]
    public void LogMetric_StoresMetricValue()
    {
        // Arrange
        var sessionId = _monitor.StartSession("metric-test");

        // Act
        _monitor.LogMetric(sessionId, "loss", 0.5, step: 1);

        // Assert
        var currentMetrics = _monitor.GetCurrentMetrics(sessionId);
        Assert.True(currentMetrics.ContainsKey("loss"));
        Assert.Equal(0.5, currentMetrics["loss"]);
    }

    [Fact]
    public void LogMetric_WithMultipleSteps_StoresHistory()
    {
        // Arrange
        var sessionId = _monitor.StartSession("metric-history-test");

        // Act
        _monitor.LogMetric(sessionId, "loss", 0.5, step: 1);
        _monitor.LogMetric(sessionId, "loss", 0.3, step: 2);
        _monitor.LogMetric(sessionId, "loss", 0.1, step: 3);

        // Assert
        var history = _monitor.GetMetricHistory(sessionId, "loss");
        Assert.Equal(3, history.Count);
        Assert.Equal(0.5, history[0].Value);  // Tuple has .Value property
        Assert.Equal(0.3, history[1].Value);
        Assert.Equal(0.1, history[2].Value);
    }

    [Fact]
    public void LogMetric_WithNullName_ThrowsArgumentException()
    {
        // Arrange
        var sessionId = _monitor.StartSession("test-session");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _monitor.LogMetric(sessionId, null!, 0.5, step: 1));
    }

    [Fact]
    public void LogMetrics_StoresMultipleMetrics()
    {
        // Arrange
        var sessionId = _monitor.StartSession("multi-metric-test");
        var metrics = new Dictionary<string, double>
        {
            ["loss"] = 0.5,
            ["accuracy"] = 0.85,
            ["f1_score"] = 0.78
        };

        // Act
        _monitor.LogMetrics(sessionId, metrics, step: 1);

        // Assert
        var currentMetrics = _monitor.GetCurrentMetrics(sessionId);
        Assert.Equal(3, currentMetrics.Count);
        Assert.Equal(0.5, currentMetrics["loss"]);
        Assert.Equal(0.85, currentMetrics["accuracy"]);
        Assert.Equal(0.78, currentMetrics["f1_score"]);
    }

    [Fact]
    public void LogMetrics_WithNullDictionary_ThrowsArgumentNullException()
    {
        // Arrange
        var sessionId = _monitor.StartSession("test-session");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _monitor.LogMetrics(sessionId, null!, step: 1));
    }

    [Fact]
    public void GetMetricHistory_WithNonexistentMetric_ReturnsEmptyList()
    {
        // Arrange
        var sessionId = _monitor.StartSession("test-session");

        // Act
        var history = _monitor.GetMetricHistory(sessionId, "nonexistent-metric");

        // Assert
        Assert.Empty(history);
    }

    #endregion

    #region Resource Usage Tests

    [Fact]
    public void LogResourceUsage_StoresResourceSnapshot()
    {
        // Arrange
        var sessionId = _monitor.StartSession("resource-test");

        // Act
        _monitor.LogResourceUsage(sessionId, cpuUsage: 45.5, memoryUsage: 8192);

        // Assert
        var resourceUsage = _monitor.GetResourceUsage(sessionId);
        Assert.Equal(45.5, resourceUsage.CpuUsagePercent);
        Assert.Equal(8192, resourceUsage.MemoryUsageMB);
    }

    [Fact]
    public void LogResourceUsage_WithGpu_StoresGpuMetrics()
    {
        // Arrange
        var sessionId = _monitor.StartSession("gpu-test");

        // Act
        _monitor.LogResourceUsage(sessionId, cpuUsage: 30, memoryUsage: 4096, gpuUsage: 85, gpuMemory: 11000);

        // Assert
        var resourceUsage = _monitor.GetResourceUsage(sessionId);
        Assert.Equal(85, resourceUsage.GpuUsagePercent);
        Assert.Equal(11000, resourceUsage.GpuMemoryUsageMB);
    }

    #endregion

    #region Progress Tracking Tests

    [Fact]
    public void UpdateProgress_StoresProgressInfo()
    {
        // Arrange
        var sessionId = _monitor.StartSession("progress-test");

        // Act
        _monitor.UpdateProgress(sessionId, currentStep: 100, totalSteps: 1000, currentEpoch: 1, totalEpochs: 10);

        // Assert
        var speedStats = _monitor.GetSpeedStats(sessionId);
        Assert.Equal(100, speedStats.IterationsCompleted);
        Assert.Equal(1000, speedStats.TotalIterations);
    }

    [Fact]
    public void GetSpeedStats_CalculatesIterationsPerSecond()
    {
        // Arrange
        var sessionId = _monitor.StartSession("speed-test");
        _monitor.UpdateProgress(sessionId, currentStep: 100, totalSteps: 1000, currentEpoch: 1, totalEpochs: 10);

        // Act
        var speedStats = _monitor.GetSpeedStats(sessionId);

        // Assert
        Assert.True(speedStats.IterationsPerSecond >= 0);
        Assert.True(speedStats.ElapsedTime > TimeSpan.Zero);
    }

    #endregion

    #region Message Logging Tests

    [Fact]
    public void LogMessage_StoresMessage()
    {
        // Arrange
        var sessionId = _monitor.StartSession("message-test");

        // Act
        _monitor.LogMessage(sessionId, LogLevel.Info, "Training started");

        // Assert - Message logged successfully (no exception thrown)
    }

    [Fact]
    public void LogMessage_WithEmptyMessage_ThrowsArgumentException()
    {
        // Arrange
        var sessionId = _monitor.StartSession("test-session");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _monitor.LogMessage(sessionId, LogLevel.Info, ""));
    }

    [Fact]
    public void LogMessage_WithDifferentLevels_StoresAllLevels()
    {
        // Arrange
        var sessionId = _monitor.StartSession("levels-test");

        // Act
        _monitor.LogMessage(sessionId, LogLevel.Debug, "Debug message");
        _monitor.LogMessage(sessionId, LogLevel.Info, "Info message");
        _monitor.LogMessage(sessionId, LogLevel.Warning, "Warning message");
        _monitor.LogMessage(sessionId, LogLevel.Error, "Error message");

        // Assert - All messages logged successfully (no exception thrown)
    }

    #endregion

    #region Epoch Tracking Tests

    [Fact]
    public void OnEpochStart_RecordsEpochStart()
    {
        // Arrange
        var sessionId = _monitor.StartSession("epoch-start-test");

        // Act
        _monitor.OnEpochStart(sessionId, epochNumber: 1);

        // Assert - Epoch start recorded (no exception thrown)
    }

    [Fact]
    public void OnEpochEnd_RecordsEpochEnd()
    {
        // Arrange
        var sessionId = _monitor.StartSession("epoch-end-test");
        var epochMetrics = new Dictionary<string, double>
        {
            ["train_loss"] = 0.25,
            ["val_loss"] = 0.28
        };

        // Act
        _monitor.OnEpochStart(sessionId, epochNumber: 1);
        _monitor.OnEpochEnd(sessionId, epochNumber: 1, epochMetrics, TimeSpan.FromSeconds(30));

        // Assert - Epoch end recorded (no exception thrown)
    }

    #endregion

    #region Issue Detection Tests

    [Fact]
    public void CheckForIssues_WithNoIssues_ReturnsEmptyList()
    {
        // Arrange
        var sessionId = _monitor.StartSession("no-issues-test");
        _monitor.UpdateProgress(sessionId, currentStep: 10, totalSteps: 100, currentEpoch: 1, totalEpochs: 10);

        // Act
        var issues = _monitor.CheckForIssues(sessionId);

        // Assert
        Assert.Empty(issues);
    }

    [Fact]
    public void CheckForIssues_WithRecentErrors_ReportsIssue()
    {
        // Arrange
        var sessionId = _monitor.StartSession("error-test");
        _monitor.LogMessage(sessionId, LogLevel.Error, "Training error occurred");

        // Act
        var issues = _monitor.CheckForIssues(sessionId);

        // Assert
        Assert.Single(issues);
        Assert.Contains("error", issues[0].ToLower());
    }

    #endregion

    #region Export Tests

    [Fact]
    public void ExportData_ToJson_CreatesFile()
    {
        // Arrange
        var sessionId = _monitor.StartSession("export-json-test");
        _monitor.LogMetric(sessionId, "loss", 0.5, step: 1);
        var exportPath = Path.Combine(_testDirectory, "export.json");

        // Act
        _monitor.ExportData(sessionId, exportPath, "json");

        // Assert
        Assert.True(File.Exists(exportPath));
        var content = File.ReadAllText(exportPath);
        Assert.Contains("loss", content);
    }

    [Fact]
    public void ExportData_ToCsv_CreatesFile()
    {
        // Arrange
        var sessionId = _monitor.StartSession("export-csv-test");
        _monitor.LogMetric(sessionId, "loss", 0.5, step: 1);
        _monitor.LogMetric(sessionId, "accuracy", 0.85, step: 1);
        var exportPath = Path.Combine(_testDirectory, "export.csv");

        // Act
        _monitor.ExportData(sessionId, exportPath, "csv");

        // Assert
        Assert.True(File.Exists(exportPath));
        var content = File.ReadAllText(exportPath);
        Assert.Contains("loss", content);
        Assert.Contains("accuracy", content);
    }

    [Fact]
    public void ExportData_WithInvalidFormat_ThrowsArgumentException()
    {
        // Arrange
        var sessionId = _monitor.StartSession("export-format-test");
        var exportPath = Path.Combine(_testDirectory, "export.xyz");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _monitor.ExportData(sessionId, exportPath, "xyz"));
    }

    [Fact]
    public void CreateVisualization_CreatesVisualizationFile()
    {
        // Arrange
        var sessionId = _monitor.StartSession("visualization-test");
        _monitor.LogMetric(sessionId, "loss", 0.5, step: 1);
        _monitor.LogMetric(sessionId, "loss", 0.3, step: 2);
        var outputPath = Path.Combine(_testDirectory, "viz.json");

        // Act
        _monitor.CreateVisualization(sessionId, new List<string> { "loss" }, outputPath);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("loss", content);
        Assert.Contains("visualization-test", content);
    }

    [Fact]
    public void CreateVisualization_WithEmptyMetricList_ThrowsArgumentException()
    {
        // Arrange
        var sessionId = _monitor.StartSession("test-session");
        var outputPath = Path.Combine(_testDirectory, "viz.json");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _monitor.CreateVisualization(sessionId, new List<string>(), outputPath));
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void LogMetric_FromMultipleThreads_IsThreadSafe()
    {
        // Arrange
        var sessionId = _monitor.StartSession("concurrent-metric-test");
        var tasks = new List<Task>();
        var metricCount = 100;

        // Act
        for (int i = 0; i < metricCount; i++)
        {
            var step = i;
            tasks.Add(Task.Run(() => _monitor.LogMetric(sessionId, "concurrent_loss", step * 0.01, step)));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        var history = _monitor.GetMetricHistory(sessionId, "concurrent_loss");
        Assert.Equal(metricCount, history.Count);
    }

    [Fact]
    public void StartSession_FromMultipleThreads_IsThreadSafe()
    {
        // Arrange
        var tasks = new List<Task<string>>();
        var sessionCount = 10;

        // Act
        for (int i = 0; i < sessionCount; i++)
        {
            var index = i;
            tasks.Add(Task.Run(() => _monitor.StartSession($"concurrent-session-{index}")));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert - All sessions created successfully
        Assert.Equal(sessionCount, tasks.Count(t => !string.IsNullOrEmpty(t.Result)));
    }

    #endregion
}
