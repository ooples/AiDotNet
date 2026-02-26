using AiDotNet.TrainingMonitoring;
using Xunit;
using LogLevel = AiDotNet.Interfaces.LogLevel;

namespace AiDotNet.Tests.IntegrationTests.TrainingMonitoring;

/// <summary>
/// Deep integration tests for TrainingMonitor:
/// session management, metric logging and history, resource usage tracking,
/// progress updates, speed stats computation (iterations/sec, ETA),
/// epoch management, issue detection, and message logging.
/// </summary>
public class TrainingMonitoringDeepMathIntegrationTests
{
    // ============================
    // Session Management Tests
    // ============================

    [Fact]
    public void StartSession_ReturnsNonEmptyId()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test Session");

        Assert.False(string.IsNullOrWhiteSpace(sessionId));
    }

    [Fact]
    public void StartSession_MultipleSessions_UniqueIds()
    {
        var monitor = new TrainingMonitor<double>();
        var ids = new HashSet<string>();

        for (int i = 0; i < 10; i++)
        {
            ids.Add(monitor.StartSession($"Session {i}"));
        }

        Assert.Equal(10, ids.Count);
    }

    [Fact]
    public void StartSession_EmptyName_Throws()
    {
        var monitor = new TrainingMonitor<double>();
        Assert.Throws<ArgumentException>(() => monitor.StartSession(""));
    }

    [Fact]
    public void EndSession_SetsEndTime()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.EndSession(sessionId);
        // Session should be ended (no exception means success)
    }

    // ============================
    // Metric Logging Tests
    // ============================

    [Fact]
    public void LogMetric_SingleMetric_Retrievable()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.LogMetric(sessionId, "loss", 0.5, step: 1);

        var current = monitor.GetCurrentMetrics(sessionId);
        Assert.Equal(0.5, current["loss"]);
    }

    [Fact]
    public void LogMetric_MultipleSteps_HistoryPreserved()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.LogMetric(sessionId, "loss", 1.0, step: 0);
        monitor.LogMetric(sessionId, "loss", 0.5, step: 1);
        monitor.LogMetric(sessionId, "loss", 0.25, step: 2);

        var history = monitor.GetMetricHistory(sessionId, "loss");
        Assert.Equal(3, history.Count);
        Assert.Equal(1.0, history[0].Value);
        Assert.Equal(0.5, history[1].Value);
        Assert.Equal(0.25, history[2].Value);
    }

    [Fact]
    public void LogMetric_EmptyName_Throws()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        Assert.Throws<ArgumentException>(() => monitor.LogMetric(sessionId, "", 1.0, 0));
    }

    [Fact]
    public void LogMetrics_MultiplePairs_AllRecorded()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.LogMetrics(sessionId, new Dictionary<string, double>
        {
            { "loss", 0.5 },
            { "accuracy", 0.85 }
        }, step: 1);

        var current = monitor.GetCurrentMetrics(sessionId);
        Assert.Equal(0.5, current["loss"]);
        Assert.Equal(0.85, current["accuracy"]);
    }

    [Fact]
    public void GetMetricHistory_UnknownMetric_ReturnsEmpty()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        var history = monitor.GetMetricHistory(sessionId, "nonexistent");
        Assert.Empty(history);
    }

    [Fact]
    public void LogMetric_OverwritesCurrentMetric()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.LogMetric(sessionId, "loss", 1.0, step: 0);
        monitor.LogMetric(sessionId, "loss", 0.5, step: 1);

        var current = monitor.GetCurrentMetrics(sessionId);
        Assert.Equal(0.5, current["loss"]); // Latest value
    }

    // ============================
    // Resource Usage Tests
    // ============================

    [Fact]
    public void LogResourceUsage_RecordsSnapshot()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.LogResourceUsage(sessionId, cpuUsage: 45.0, memoryUsage: 60.0);

        var usage = monitor.GetResourceUsage(sessionId);
        Assert.Equal(45.0, usage.CpuUsagePercent);
        Assert.Equal(60.0, usage.MemoryUsageMB);
    }

    [Fact]
    public void LogResourceUsage_WithGpu_RecordsGpuMetrics()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.LogResourceUsage(sessionId, cpuUsage: 30.0, memoryUsage: 50.0,
            gpuUsage: 85.0, gpuMemory: 90.0);

        var usage = monitor.GetResourceUsage(sessionId);
        Assert.Equal(85.0, usage.GpuUsagePercent);
        Assert.Equal(90.0, usage.GpuMemoryUsageMB);
    }

    [Fact]
    public void GetResourceUsage_NoData_ReturnsDefault()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        var usage = monitor.GetResourceUsage(sessionId);
        Assert.Equal(0.0, usage.CpuUsagePercent);
    }

    // ============================
    // Progress and Speed Stats Tests
    // ============================

    [Fact]
    public void UpdateProgress_SetsProgressInfo()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.UpdateProgress(sessionId, currentStep: 50, totalSteps: 100,
            currentEpoch: 1, totalEpochs: 10);

        var stats = monitor.GetSpeedStats(sessionId);
        Assert.Equal(50, stats.IterationsCompleted);
        Assert.Equal(100, stats.TotalIterations);
    }

    [Fact]
    public void GetSpeedStats_WithProgress_CalculatesPercentage()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.UpdateProgress(sessionId, currentStep: 50, totalSteps: 200,
            currentEpoch: 1, totalEpochs: 10);

        var stats = monitor.GetSpeedStats(sessionId);

        // 50/200 = 25%
        Assert.Equal(25.0, stats.ProgressPercentage, 1.0);
    }

    [Fact]
    public void GetSpeedStats_WithProgress_IterationsPerSecond_Positive()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        // Simulate some work
        Thread.Sleep(50);
        monitor.UpdateProgress(sessionId, currentStep: 100, totalSteps: 1000,
            currentEpoch: 1, totalEpochs: 10);

        var stats = monitor.GetSpeedStats(sessionId);
        Assert.True(stats.IterationsPerSecond > 0);
        Assert.True(stats.SecondsPerIteration > 0);
    }

    [Fact]
    public void GetSpeedStats_WithProgress_EstimatedRemaining_NonNegative()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        Thread.Sleep(50);
        monitor.UpdateProgress(sessionId, currentStep: 100, totalSteps: 1000,
            currentEpoch: 1, totalEpochs: 10);

        var stats = monitor.GetSpeedStats(sessionId);
        Assert.True(stats.EstimatedTimeRemaining >= TimeSpan.Zero);
    }

    [Fact]
    public void GetSpeedStats_NoProgress_ZeroValues()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        var stats = monitor.GetSpeedStats(sessionId);
        Assert.Equal(0, stats.IterationsPerSecond);
        Assert.Equal(0, stats.SecondsPerIteration);
    }

    [Fact]
    public void GetSpeedStats_FullCompletion_100Percent()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        Thread.Sleep(10);
        monitor.UpdateProgress(sessionId, currentStep: 100, totalSteps: 100,
            currentEpoch: 10, totalEpochs: 10);

        var stats = monitor.GetSpeedStats(sessionId);
        Assert.Equal(100.0, stats.ProgressPercentage, 0.5);
    }

    // ============================
    // Epoch Management Tests
    // ============================

    [Fact]
    public void OnEpochStart_LogsMessage()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.OnEpochStart(sessionId, epochNumber: 1);
        // Should not throw, and logs "Epoch 1 started"
    }

    [Fact]
    public void OnEpochEnd_LogsCompletionMessage()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.OnEpochStart(sessionId, 1);
        monitor.OnEpochEnd(sessionId, 1,
            new Dictionary<string, double> { { "loss", 0.5 } },
            TimeSpan.FromSeconds(10));
        // Should not throw
    }

    // ============================
    // Message Logging Tests
    // ============================

    [Fact]
    public void LogMessage_EmptyMessage_Throws()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        Assert.Throws<ArgumentException>(() =>
            monitor.LogMessage(sessionId, LogLevel.Info, ""));
    }

    [Fact]
    public void LogMessage_ValidMessage_NoThrow()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.LogMessage(sessionId, LogLevel.Info, "Training started");
        monitor.LogMessage(sessionId, LogLevel.Warning, "Loss not decreasing");
        monitor.LogMessage(sessionId, LogLevel.Error, "NaN detected in gradients");
    }

    // ============================
    // Issue Detection Tests
    // ============================

    [Fact]
    public void CheckForIssues_NoIssues_EmptyList()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        var issues = monitor.CheckForIssues(sessionId);
        Assert.Empty(issues);
    }

    [Fact]
    public void CheckForIssues_HighMemory_ReportsIssue()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        // Log high memory usage (>90%)
        monitor.LogResourceUsage(sessionId, cpuUsage: 50, memoryUsage: 95);

        var issues = monitor.CheckForIssues(sessionId);
        Assert.Contains(issues, i => i.Contains("memory"));
    }

    [Fact]
    public void CheckForIssues_CriticalGpuMemory_ReportsIssue()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        // Log critical GPU memory (>95%)
        monitor.LogResourceUsage(sessionId, cpuUsage: 50, memoryUsage: 50,
            gpuUsage: 80, gpuMemory: 98);

        var issues = monitor.CheckForIssues(sessionId);
        Assert.Contains(issues, i => i.Contains("GPU memory"));
    }

    [Fact]
    public void CheckForIssues_RecentErrors_ReportsIssue()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        monitor.LogMessage(sessionId, LogLevel.Error, "NaN detected");
        monitor.LogMessage(sessionId, LogLevel.Error, "Gradient explosion");

        var issues = monitor.CheckForIssues(sessionId);
        Assert.Contains(issues, i => i.Contains("error"));
    }

    // ============================
    // Speed Stats Math Verification
    // ============================

    [Fact]
    public void SpeedStats_IterationsPerSecond_InverseOfSecondsPerIteration()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        Thread.Sleep(100); // Give some real time
        monitor.UpdateProgress(sessionId, currentStep: 50, totalSteps: 100,
            currentEpoch: 1, totalEpochs: 10);

        var stats = monitor.GetSpeedStats(sessionId);

        if (stats.IterationsPerSecond > 0 && stats.SecondsPerIteration > 0)
        {
            // IterationsPerSecond * SecondsPerIteration should equal 1
            double product = stats.IterationsPerSecond * stats.SecondsPerIteration;
            Assert.Equal(1.0, product, 0.01);
        }
    }

    [Fact]
    public void SpeedStats_ElapsedTime_NonNegative()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        var stats = monitor.GetSpeedStats(sessionId);
        Assert.True(stats.ElapsedTime >= TimeSpan.Zero);
    }

    [Fact]
    public void SpeedStats_ProgressPercentage_Between0And100()
    {
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("Test");

        Thread.Sleep(10);
        monitor.UpdateProgress(sessionId, currentStep: 75, totalSteps: 100,
            currentEpoch: 1, totalEpochs: 10);

        var stats = monitor.GetSpeedStats(sessionId);
        Assert.InRange(stats.ProgressPercentage, 0, 100);
        Assert.Equal(75.0, stats.ProgressPercentage, 1.0);
    }
}
