using AiDotNet.LinearAlgebra;
using AiDotNet.Serving.Batching;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Monitoring;
using AiDotNet.Serving.Padding;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// Integration tests for the AiDotNet.Serving module.
/// Tests configuration options, monitoring/metrics, and additional batching/padding scenarios.
/// </summary>
public class ServingIntegrationTests
{
    #region ServingOptions Tests

    [Fact]
    public void ServingOptions_DefaultValues_AreCorrect()
    {
        var options = new ServingOptions();

        Assert.Equal(52432, options.Port);
        Assert.Equal(10, options.BatchingWindowMs);
        Assert.Equal(100, options.MaxBatchSize);
        Assert.Equal(1, options.MinBatchSize);
        Assert.True(options.AdaptiveBatchSize);
        Assert.Equal(BatchingStrategyType.Adaptive, options.BatchingStrategy);
        Assert.Equal(20.0, options.TargetLatencyMs);
        Assert.Equal(2.0, options.LatencyToleranceFactor);
        Assert.Equal(1000, options.MaxQueueSize);
        Assert.False(options.EnablePriorityScheduling);
        Assert.Equal(PaddingStrategyType.Minimal, options.PaddingStrategy);
        Assert.Equal(new[] { 32, 64, 128, 256, 512 }, options.BucketSizes);
        Assert.Equal(512, options.FixedPaddingSize);
        Assert.True(options.EnablePerformanceMetrics);
        Assert.Equal(10000, options.MaxLatencySamples);
        Assert.Equal("models", options.ModelDirectory);
        Assert.NotNull(options.StartupModels);
        Assert.Empty(options.StartupModels);
    }

    [Fact]
    public void ServingOptions_CustomValues_ArePreserved()
    {
        var options = new ServingOptions
        {
            Port = 8080,
            BatchingWindowMs = 50,
            MaxBatchSize = 200,
            MinBatchSize = 5,
            AdaptiveBatchSize = false,
            BatchingStrategy = BatchingStrategyType.Size,
            TargetLatencyMs = 30.0,
            LatencyToleranceFactor = 1.5,
            MaxQueueSize = 500,
            EnablePriorityScheduling = true,
            PaddingStrategy = PaddingStrategyType.Fixed,
            BucketSizes = new[] { 16, 32, 64 },
            FixedPaddingSize = 256,
            EnablePerformanceMetrics = false,
            MaxLatencySamples = 5000,
            ModelDirectory = "/custom/models"
        };

        Assert.Equal(8080, options.Port);
        Assert.Equal(50, options.BatchingWindowMs);
        Assert.Equal(200, options.MaxBatchSize);
        Assert.Equal(5, options.MinBatchSize);
        Assert.False(options.AdaptiveBatchSize);
        Assert.Equal(BatchingStrategyType.Size, options.BatchingStrategy);
        Assert.Equal(30.0, options.TargetLatencyMs);
        Assert.Equal(1.5, options.LatencyToleranceFactor);
        Assert.Equal(500, options.MaxQueueSize);
        Assert.True(options.EnablePriorityScheduling);
        Assert.Equal(PaddingStrategyType.Fixed, options.PaddingStrategy);
        Assert.Equal(new[] { 16, 32, 64 }, options.BucketSizes);
        Assert.Equal(256, options.FixedPaddingSize);
        Assert.False(options.EnablePerformanceMetrics);
        Assert.Equal(5000, options.MaxLatencySamples);
        Assert.Equal("/custom/models", options.ModelDirectory);
    }

    [Fact]
    public void ServingOptions_StartupModels_CanBeAdded()
    {
        var options = new ServingOptions();
        options.StartupModels.Add(new StartupModel
        {
            Path = "models/model1.bin",
            Name = "TestModel",
            NumericType = NumericType.Double
        });

        Assert.Single(options.StartupModels);
        Assert.Equal("TestModel", options.StartupModels[0].Name);
    }

    #endregion

    #region StartupModel Tests

    [Fact]
    public void StartupModel_DefaultValues_AreCorrect()
    {
        var model = new StartupModel();

        Assert.Equal(string.Empty, model.Path);
        Assert.Equal(string.Empty, model.Name);
        Assert.Equal(NumericType.Double, model.NumericType);
        Assert.Null(model.Sha256);
    }

    [Fact]
    public void StartupModel_CustomValues_ArePreserved()
    {
        var model = new StartupModel
        {
            Path = "/path/to/model.bin",
            Name = "MyModel",
            NumericType = NumericType.Float,
            Sha256 = "abc123"
        };

        Assert.Equal("/path/to/model.bin", model.Path);
        Assert.Equal("MyModel", model.Name);
        Assert.Equal(NumericType.Float, model.NumericType);
        Assert.Equal("abc123", model.Sha256);
    }

    #endregion

    #region PerformanceMetrics Tests

    [Fact]
    public void PerformanceMetrics_RecordLatency_StoresSamples()
    {
        var metrics = new PerformanceMetrics(maxSamples: 100);

        metrics.RecordLatency(10.0);
        metrics.RecordLatency(20.0);
        metrics.RecordLatency(30.0);

        var avgLatency = metrics.GetAverageLatency();
        Assert.Equal(20.0, avgLatency);
    }

    [Fact]
    public void PerformanceMetrics_RecordLatency_TrimsSamplesWhenExceeded()
    {
        var metrics = new PerformanceMetrics(maxSamples: 5);

        for (int i = 0; i < 10; i++)
        {
            metrics.RecordLatency(i * 10.0);
        }

        // Should have trimmed to 5 samples (the last 5: 50, 60, 70, 80, 90)
        // Average should be (50+60+70+80+90)/5 = 70
        var avgLatency = metrics.GetAverageLatency();
        Assert.Equal(70.0, avgLatency);
    }

    [Fact]
    public void PerformanceMetrics_GetLatencyPercentile_P50_ReturnsMedian()
    {
        var metrics = new PerformanceMetrics();

        // Add samples 1 through 100
        for (int i = 1; i <= 100; i++)
        {
            metrics.RecordLatency(i);
        }

        var p50 = metrics.GetLatencyPercentile(50);
        Assert.True(p50 >= 49 && p50 <= 51); // Should be around 50
    }

    [Fact]
    public void PerformanceMetrics_GetLatencyPercentile_P99_ReturnsHighValue()
    {
        var metrics = new PerformanceMetrics();

        for (int i = 1; i <= 100; i++)
        {
            metrics.RecordLatency(i);
        }

        var p99 = metrics.GetLatencyPercentile(99);
        Assert.True(p99 >= 98); // Should be near the top
    }

    [Fact]
    public void PerformanceMetrics_GetLatencyPercentile_EmptySamples_ReturnsZero()
    {
        var metrics = new PerformanceMetrics();

        var p50 = metrics.GetLatencyPercentile(50);
        Assert.Equal(0.0, p50);
    }

    [Fact]
    public void PerformanceMetrics_GetLatencyPercentile_InvalidPercentile_ThrowsException()
    {
        var metrics = new PerformanceMetrics();
        metrics.RecordLatency(10.0);

        Assert.Throws<ArgumentException>(() => metrics.GetLatencyPercentile(-1));
        Assert.Throws<ArgumentException>(() => metrics.GetLatencyPercentile(101));
    }

    [Fact]
    public void PerformanceMetrics_RecordBatch_TracksTotals()
    {
        var metrics = new PerformanceMetrics();

        metrics.RecordBatch(10, 15.0);
        metrics.RecordBatch(20, 25.0);
        metrics.RecordBatch(15, 20.0);

        var allMetrics = metrics.GetAllMetrics();
        Assert.Equal(45L, allMetrics["totalRequests"]);
        Assert.Equal(3L, allMetrics["totalBatches"]);
        Assert.Equal(15.0, allMetrics["averageBatchSize"]);
    }

    [Fact]
    public void PerformanceMetrics_RecordQueueDepth_StoresSamples()
    {
        var metrics = new PerformanceMetrics(maxQueueDepthSamples: 100);

        metrics.RecordQueueDepth(5);
        metrics.RecordQueueDepth(10);
        metrics.RecordQueueDepth(15);

        var avgQueueDepth = metrics.GetAverageQueueDepth();
        Assert.Equal(10.0, avgQueueDepth);
    }

    [Fact]
    public void PerformanceMetrics_RecordBatchUtilization_CalculatesCorrectly()
    {
        var metrics = new PerformanceMetrics();

        metrics.RecordBatchUtilization(80, 20); // 80% utilization
        metrics.RecordBatchUtilization(90, 10); // 90% utilization

        var utilization = metrics.GetBatchUtilization();
        // (80 + 90) / (80 + 20 + 90 + 10) = 170 / 200 = 85%
        Assert.Equal(85.0, utilization);
    }

    [Fact]
    public void PerformanceMetrics_GetBatchUtilization_NoSamples_ReturnsHundred()
    {
        var metrics = new PerformanceMetrics();

        var utilization = metrics.GetBatchUtilization();
        Assert.Equal(100.0, utilization);
    }

    [Fact]
    public void PerformanceMetrics_GetThroughput_CalculatesCorrectly()
    {
        var metrics = new PerformanceMetrics();

        metrics.RecordBatch(100, 10.0);

        // Small sleep to ensure elapsed time is positive
        Thread.Sleep(10);

        var throughput = metrics.GetThroughput();
        Assert.True(throughput > 0);
    }

    [Fact]
    public void PerformanceMetrics_GetAllMetrics_ReturnsAllKeys()
    {
        var metrics = new PerformanceMetrics();
        metrics.RecordBatch(10, 15.0);
        metrics.RecordQueueDepth(5);
        metrics.RecordBatchUtilization(80, 20);

        var allMetrics = metrics.GetAllMetrics();

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
    public void PerformanceMetrics_Reset_ClearsAllData()
    {
        var metrics = new PerformanceMetrics();

        metrics.RecordBatch(100, 15.0);
        metrics.RecordQueueDepth(50);
        metrics.RecordBatchUtilization(80, 20);

        metrics.Reset();

        var allMetrics = metrics.GetAllMetrics();
        Assert.Equal(0L, allMetrics["totalRequests"]);
        Assert.Equal(0L, allMetrics["totalBatches"]);
        Assert.Equal(0.0, metrics.GetAverageLatency());
        Assert.Equal(0.0, metrics.GetAverageQueueDepth());
    }

    #endregion

    #region ContinuousBatchingStrategy Tests

    [Fact]
    public void ContinuousBatchingStrategy_ShouldProcessBatch_WhenRequestsExist()
    {
        var strategy = new ContinuousBatchingStrategy(
            maxConcurrency: 32,
            minWaitMs: 0, // Set to 0 for testing
            targetLatencyMs: 50,
            adaptiveConcurrency: true);

        // No requests - should not process
        var shouldProcess1 = strategy.ShouldProcessBatch(
            queuedRequests: 0,
            timeInQueueMs: 10,
            averageLatencyMs: 15,
            queueDepth: 0);

        // Has requests - should process
        var shouldProcess2 = strategy.ShouldProcessBatch(
            queuedRequests: 5,
            timeInQueueMs: 10,
            averageLatencyMs: 15,
            queueDepth: 5);

        Assert.False(shouldProcess1);
        Assert.True(shouldProcess2);
    }

    [Fact]
    public void ContinuousBatchingStrategy_GetOptimalBatchSize_ReturnsMinOfQueueAndConcurrency()
    {
        var strategy = new ContinuousBatchingStrategy(
            maxConcurrency: 32,
            minWaitMs: 1,
            targetLatencyMs: 50,
            adaptiveConcurrency: false);

        // Current concurrency starts at maxConcurrency/2 = 16
        var batchSize1 = strategy.GetOptimalBatchSize(100, 15.0);
        var batchSize2 = strategy.GetOptimalBatchSize(3, 15.0);

        Assert.True(batchSize1 <= 32); // Should be limited by concurrency
        Assert.Equal(3, batchSize2); // Should be limited by queue size
    }

    [Fact]
    public void ContinuousBatchingStrategy_Name_IsCorrect()
    {
        var strategy = new ContinuousBatchingStrategy();

        Assert.Equal("Continuous", strategy.Name);
    }

    [Fact]
    public void ContinuousBatchingStrategy_CurrentConcurrency_StartsAtHalfMax()
    {
        var strategy = new ContinuousBatchingStrategy(maxConcurrency: 32);

        Assert.Equal(16, strategy.CurrentConcurrency);
    }

    [Fact]
    public void ContinuousBatchingStrategy_AdaptiveConcurrency_IncreaseOnGoodLatency()
    {
        var strategy = new ContinuousBatchingStrategy(
            maxConcurrency: 32,
            targetLatencyMs: 50,
            adaptiveConcurrency: true);

        var initialConcurrency = strategy.CurrentConcurrency;

        // Provide feedback with good latency (below 40% of target)
        strategy.UpdatePerformanceFeedback(10, 10.0); // per-request = 1ms, which is < 50*0.8=40ms

        Assert.True(strategy.CurrentConcurrency >= initialConcurrency);
    }

    [Fact]
    public void ContinuousBatchingStrategy_GetStatistics_ContainsExpectedKeys()
    {
        var strategy = new ContinuousBatchingStrategy(
            maxConcurrency: 32,
            minWaitMs: 1,
            targetLatencyMs: 50,
            adaptiveConcurrency: true);

        var stats = strategy.GetStatistics();

        Assert.Contains("currentConcurrency", stats.Keys);
        Assert.Contains("maxConcurrency", stats.Keys);
        Assert.Contains("targetLatencyMs", stats.Keys);
        Assert.Contains("adaptiveConcurrency", stats.Keys);
    }

    #endregion

    #region Additional TimeoutBatchingStrategy Tests

    [Fact]
    public void TimeoutBatchingStrategy_ShouldProcessBatch_OnlyWhenTimeoutReached()
    {
        var strategy = new TimeoutBatchingStrategy(timeoutMs: 100, maxBatchSize: 10);

        // Even with max batch size reached, should not process before timeout
        var shouldProcessBeforeTimeout = strategy.ShouldProcessBatch(
            queuedRequests: 10,
            timeInQueueMs: 5,
            averageLatencyMs: 15,
            queueDepth: 10);

        // Should process once timeout is reached
        var shouldProcessAfterTimeout = strategy.ShouldProcessBatch(
            queuedRequests: 5,
            timeInQueueMs: 100,
            averageLatencyMs: 15,
            queueDepth: 5);

        Assert.False(shouldProcessBeforeTimeout);
        Assert.True(shouldProcessAfterTimeout);
    }

    [Fact]
    public void TimeoutBatchingStrategy_GetOptimalBatchSize_ReturnsMaxBatchSize()
    {
        var strategy = new TimeoutBatchingStrategy(timeoutMs: 100, maxBatchSize: 50);

        var batchSize = strategy.GetOptimalBatchSize(100, 15.0);

        Assert.Equal(50, batchSize);
    }

    [Fact]
    public void TimeoutBatchingStrategy_Name_IsCorrect()
    {
        var strategy = new TimeoutBatchingStrategy(100, 50);

        Assert.Equal("Timeout", strategy.Name);
    }

    #endregion

    #region Additional SizeBatchingStrategy Tests

    [Fact]
    public void SizeBatchingStrategy_GetOptimalBatchSize_ReturnsBatchSize()
    {
        var strategy = new SizeBatchingStrategy(batchSize: 32, maxWaitMs: 100);

        var batchSize = strategy.GetOptimalBatchSize(100, 15.0);

        Assert.Equal(32, batchSize);
    }

    [Fact]
    public void SizeBatchingStrategy_Name_IsCorrect()
    {
        var strategy = new SizeBatchingStrategy(32, 100);

        Assert.Equal("Size", strategy.Name);
    }

    #endregion

    #region Additional AdaptiveBatchingStrategy Tests

    [Fact]
    public void AdaptiveBatchingStrategy_GetOptimalBatchSize_RespectsMinMax()
    {
        var strategy = new AdaptiveBatchingStrategy(
            minBatchSize: 5,
            maxBatchSize: 50,
            maxWaitMs: 100,
            targetLatencyMs: 20.0,
            latencyToleranceFactor: 2.0);

        // Get multiple batch sizes to verify they stay within bounds
        for (int i = 0; i < 10; i++)
        {
            var batchSize = strategy.GetOptimalBatchSize(100, 15.0);
            Assert.True(batchSize >= 5 && batchSize <= 50);
            strategy.UpdatePerformanceFeedback(batchSize, 15.0);
        }
    }

    [Fact]
    public void AdaptiveBatchingStrategy_Name_IsCorrect()
    {
        var strategy = new AdaptiveBatchingStrategy(1, 100, 50, 20.0, 2.0);

        Assert.Equal("Adaptive", strategy.Name);
    }

    #endregion

    #region Additional BucketBatchingStrategy Tests

    [Fact]
    public void BucketBatchingStrategy_GetBucketIndex_HandlesEdgeCases()
    {
        var bucketSizes = new[] { 8, 16, 32 };
        var strategy = new BucketBatchingStrategy(bucketSizes, maxBatchSize: 100, maxWaitMs: 50);

        // Minimum size
        Assert.Equal(0, strategy.GetBucketIndex(1));

        // Exactly at bucket boundary
        Assert.Equal(0, strategy.GetBucketIndex(8));
        Assert.Equal(1, strategy.GetBucketIndex(9));
        Assert.Equal(1, strategy.GetBucketIndex(16));
        Assert.Equal(2, strategy.GetBucketIndex(17));
    }

    [Fact]
    public void BucketBatchingStrategy_ShouldProcessBatch_WhenMaxWaitReached()
    {
        var bucketSizes = new[] { 32, 64, 128 };
        var strategy = new BucketBatchingStrategy(bucketSizes, maxBatchSize: 100, maxWaitMs: 50);

        // Max wait exceeded
        var shouldProcess = strategy.ShouldProcessBatch(
            queuedRequests: 5,
            timeInQueueMs: 60,
            averageLatencyMs: 15,
            queueDepth: 5);

        Assert.True(shouldProcess);
    }

    [Fact]
    public void BucketBatchingStrategy_Name_IsCorrect()
    {
        var bucketSizes = new[] { 32, 64, 128 };
        var strategy = new BucketBatchingStrategy(bucketSizes, 100, 50);

        Assert.Equal("Bucket", strategy.Name);
    }

    #endregion

    #region Additional Padding Strategy Tests

    [Fact]
    public void MinimalPaddingStrategy_PadBatch_WithSingleVector()
    {
        var strategy = new MinimalPaddingStrategy();
        var vectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3, 4, 5 })
        };

        var paddedMatrix = strategy.PadBatch(vectors, out var attentionMask);

        Assert.Equal(1, paddedMatrix.Rows);
        Assert.Equal(5, paddedMatrix.Columns);
        Assert.NotNull(attentionMask);
        Assert.Equal(1.0, attentionMask![0, 0]);
        Assert.Equal(1.0, attentionMask[0, 4]);
    }

    [Fact]
    public void BucketPaddingStrategy_PadBatch_SelectsCorrectBucket()
    {
        var bucketSizes = new[] { 8, 16, 32, 64 };
        var strategy = new BucketPaddingStrategy(bucketSizes);
        var vectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }) // Length 9, should use 16
        };

        var paddedMatrix = strategy.PadBatch(vectors, out var attentionMask);

        Assert.Equal(1, paddedMatrix.Rows);
        Assert.Equal(16, paddedMatrix.Columns);
    }

    [Fact]
    public void BucketPaddingStrategy_PadBatch_HandlesLargerThanAllBuckets()
    {
        var bucketSizes = new[] { 8, 16, 32 };
        var strategy = new BucketPaddingStrategy(bucketSizes);
        var vectors = new[]
        {
            new Vector<double>(new double[50]) // Larger than max bucket
        };

        var paddedMatrix = strategy.PadBatch(vectors, out var attentionMask);

        // Should pad to next power-of-2 like size or just the max length
        Assert.Equal(1, paddedMatrix.Rows);
        Assert.True(paddedMatrix.Columns >= 50);
    }

    [Fact]
    public void FixedSizePaddingStrategy_PadBatch_ConsistentSize()
    {
        var strategy = new FixedSizePaddingStrategy(fixedLength: 20);
        var vectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3 }),
            new Vector<double>(new double[] { 4, 5, 6, 7, 8, 9 }),
            new Vector<double>(new double[] { 10 })
        };

        var paddedMatrix = strategy.PadBatch(vectors, out var attentionMask);

        Assert.Equal(3, paddedMatrix.Rows);
        Assert.Equal(20, paddedMatrix.Columns); // Fixed at 20
        Assert.NotNull(attentionMask);
    }

    [Fact]
    public void PaddingStrategies_PreserveOriginalData()
    {
        var strategies = new IPaddingStrategy[]
        {
            new MinimalPaddingStrategy(),
            new BucketPaddingStrategy(new[] { 8, 16, 32 }),
            new FixedSizePaddingStrategy(20)
        };

        var originalData = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        var vectors = new[] { new Vector<double>(originalData) };

        foreach (var strategy in strategies)
        {
            var paddedMatrix = strategy.PadBatch(vectors, out _);

            // Verify original data is preserved
            for (int i = 0; i < originalData.Length; i++)
            {
                Assert.Equal(originalData[i], paddedMatrix[0, i]);
            }
        }
    }

    #endregion

    #region Configuration Enum Tests

    [Fact]
    public void BatchingStrategyType_AllValuesExist()
    {
        var values = Enum.GetValues<BatchingStrategyType>();

        Assert.Contains(BatchingStrategyType.Timeout, values);
        Assert.Contains(BatchingStrategyType.Size, values);
        Assert.Contains(BatchingStrategyType.Adaptive, values);
        Assert.Contains(BatchingStrategyType.Bucket, values);
        Assert.Contains(BatchingStrategyType.Continuous, values);
    }

    [Fact]
    public void PaddingStrategyType_AllValuesExist()
    {
        var values = Enum.GetValues<PaddingStrategyType>();

        Assert.Contains(PaddingStrategyType.Minimal, values);
        Assert.Contains(PaddingStrategyType.Bucket, values);
        Assert.Contains(PaddingStrategyType.Fixed, values);
    }

    [Fact]
    public void NumericType_AllValuesExist()
    {
        var values = Enum.GetValues<NumericType>();

        Assert.Contains(NumericType.Double, values);
        Assert.Contains(NumericType.Float, values);
    }

    #endregion
}
