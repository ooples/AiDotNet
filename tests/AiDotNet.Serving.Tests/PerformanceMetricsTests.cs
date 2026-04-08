using AiDotNet.Serving.Monitoring;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for performance metrics tracking.
/// </summary>
public class PerformanceMetricsTests
{
    [Fact]
    public void PerformanceMetrics_ShouldRecordLatency()
    {
        // Arrange
        var metrics = new PerformanceMetrics();

        // Act
        metrics.RecordLatency(10.0);
        metrics.RecordLatency(20.0);
        metrics.RecordLatency(30.0);

        // Assert
        var averageLatency = metrics.GetAverageLatency();
        Assert.Equal(20.0, averageLatency, precision: 1);
    }

    [Fact]
    public void PerformanceMetrics_ShouldCalculateLatencyPercentiles()
    {
        // Arrange
        var metrics = new PerformanceMetrics();

        // Add 100 latency samples
        for (int i = 1; i <= 100; i++)
        {
            metrics.RecordLatency(i);
        }

        // Act
        var p50 = metrics.GetLatencyPercentile(50);
        var p95 = metrics.GetLatencyPercentile(95);
        var p99 = metrics.GetLatencyPercentile(99);

        // Assert
        Assert.InRange(p50, 45, 55); // Around 50th percentile
        Assert.InRange(p95, 90, 100); // Around 95th percentile
        Assert.InRange(p99, 95, 100); // Around 99th percentile
    }

    [Fact]
    public void PerformanceMetrics_ShouldRecordBatchMetrics()
    {
        // Arrange
        var metrics = new PerformanceMetrics();

        // Act
        metrics.RecordBatch(batchSize: 10, latencyMs: 15.0);
        metrics.RecordBatch(batchSize: 20, latencyMs: 25.0);
        metrics.RecordBatch(batchSize: 15, latencyMs: 20.0);

        // Assert
        var allMetrics = metrics.GetAllMetrics();
        Assert.Equal(45L, allMetrics["totalRequests"]); // 10 + 20 + 15
        Assert.Equal(3L, allMetrics["totalBatches"]);
        Assert.Equal(15.0, (double)allMetrics["averageBatchSize"]); // 45 / 3
    }

    [Fact]
    public void PerformanceMetrics_ShouldCalculateThroughput()
    {
        // Arrange
        var metrics = new PerformanceMetrics();

        // Act
        for (int i = 0; i < 100; i++)
        {
            metrics.RecordBatch(batchSize: 10, latencyMs: 10.0);
        }

        Thread.Sleep(100); // Wait 100ms to ensure elapsed time

        // Assert
        var throughput = metrics.GetThroughput();
        Assert.True(throughput > 0); // Should have some throughput
    }

    [Fact]
    public void PerformanceMetrics_ShouldRecordQueueDepth()
    {
        // Arrange
        var metrics = new PerformanceMetrics();

        // Act
        metrics.RecordQueueDepth(5);
        metrics.RecordQueueDepth(10);
        metrics.RecordQueueDepth(15);

        // Assert
        var avgQueueDepth = metrics.GetAverageQueueDepth();
        Assert.Equal(10.0, avgQueueDepth, precision: 1);
    }

    [Fact]
    public void PerformanceMetrics_ShouldCalculateBatchUtilization()
    {
        // Arrange
        var metrics = new PerformanceMetrics();

        // Act
        metrics.RecordBatchUtilization(actualElements: 80, paddingElements: 20);
        metrics.RecordBatchUtilization(actualElements: 90, paddingElements: 10);

        // Assert
        var utilization = metrics.GetBatchUtilization();
        Assert.Equal(85.0, utilization, precision: 1); // (170 / 200) * 100
    }

    [Fact]
    public void PerformanceMetrics_ShouldReturnAllMetrics()
    {
        // Arrange
        var metrics = new PerformanceMetrics();

        // Act
        metrics.RecordBatch(batchSize: 10, latencyMs: 15.0);
        metrics.RecordQueueDepth(5);
        metrics.RecordBatchUtilization(actualElements: 80, paddingElements: 20);

        var allMetrics = metrics.GetAllMetrics();

        // Assert
        Assert.Contains("totalRequests", allMetrics.Keys);
        Assert.Contains("totalBatches", allMetrics.Keys);
        Assert.Contains("throughputRequestsPerSecond", allMetrics.Keys);
        Assert.Contains("averageBatchSize", allMetrics.Keys);
        Assert.Contains("latencyP50Ms", allMetrics.Keys);
        Assert.Contains("latencyP95Ms", allMetrics.Keys);
        Assert.Contains("latencyP99Ms", allMetrics.Keys);
        Assert.Contains("averageLatencyMs", allMetrics.Keys);
        Assert.Contains("averageQueueDepth", allMetrics.Keys);
        Assert.Contains("batchUtilizationPercent", allMetrics.Keys);
        Assert.Contains("uptimeSeconds", allMetrics.Keys);
    }

    [Fact]
    public void PerformanceMetrics_ShouldReset()
    {
        // Arrange
        var metrics = new PerformanceMetrics();
        metrics.RecordBatch(batchSize: 10, latencyMs: 15.0);
        metrics.RecordQueueDepth(5);

        // Act
        metrics.Reset();

        // Assert
        var allMetrics = metrics.GetAllMetrics();
        Assert.Equal(0L, allMetrics["totalRequests"]);
        Assert.Equal(0L, allMetrics["totalBatches"]);
        Assert.Equal(0.0, (double)allMetrics["averageLatencyMs"]);
    }

    [Fact]
    public void PerformanceMetrics_ShouldLimitSampleSize()
    {
        // Arrange
        var metrics = new PerformanceMetrics(maxSamples: 100);

        // Act - Add more samples than the limit
        for (int i = 0; i < 200; i++)
        {
            metrics.RecordLatency(i);
        }

        // Assert - Should only keep the most recent 100 samples
        var p50 = metrics.GetLatencyPercentile(50);
        Assert.True(p50 >= 100); // Should be from the most recent samples
    }

    #region PR #758 Bug Fix Tests - Parameter Validation

    [Fact]
    public void PerformanceMetrics_Constructor_ThrowsOnInvalidMaxSamples()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PerformanceMetrics(maxSamples: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PerformanceMetrics(maxSamples: -1));
    }

    [Fact]
    public void PerformanceMetrics_Constructor_ThrowsOnInvalidMaxQueueDepthSamples()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PerformanceMetrics(maxSamples: 100, maxQueueDepthSamples: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PerformanceMetrics(maxSamples: 100, maxQueueDepthSamples: -1));
    }

    #endregion
}
