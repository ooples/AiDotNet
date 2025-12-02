using AiDotNet.Serving.Batching;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for batching strategies.
/// </summary>
public class BatchingStrategyTests
{
    [Fact]
    public void TimeoutBatchingStrategy_ShouldProcessBatch_WhenTimeoutReached()
    {
        // Arrange
        var strategy = new TimeoutBatchingStrategy(timeoutMs: 10, maxBatchSize: 100);

        // Act - Time not reached
        var shouldProcess1 = strategy.ShouldProcessBatch(
            queuedRequests: 5,
            timeInQueueMs: 5,
            averageLatencyMs: 15,
            queueDepth: 5);

        // Act - Time reached
        var shouldProcess2 = strategy.ShouldProcessBatch(
            queuedRequests: 5,
            timeInQueueMs: 11,
            averageLatencyMs: 15,
            queueDepth: 5);

        // Assert
        Assert.False(shouldProcess1);
        Assert.True(shouldProcess2);
    }

    [Fact]
    public void SizeBatchingStrategy_ShouldProcessBatch_WhenSizeReached()
    {
        // Arrange
        var strategy = new SizeBatchingStrategy(batchSize: 10, maxWaitMs: 100);

        // Act - Size not reached
        var shouldProcess1 = strategy.ShouldProcessBatch(
            queuedRequests: 5,
            timeInQueueMs: 5,
            averageLatencyMs: 15,
            queueDepth: 5);

        // Act - Size reached
        var shouldProcess2 = strategy.ShouldProcessBatch(
            queuedRequests: 10,
            timeInQueueMs: 5,
            averageLatencyMs: 15,
            queueDepth: 10);

        // Assert
        Assert.False(shouldProcess1);
        Assert.True(shouldProcess2);
    }

    [Fact]
    public void SizeBatchingStrategy_ShouldProcessBatch_WhenMaxWaitReached()
    {
        // Arrange
        var strategy = new SizeBatchingStrategy(batchSize: 10, maxWaitMs: 100);

        // Act - Max wait reached with smaller batch
        var shouldProcess = strategy.ShouldProcessBatch(
            queuedRequests: 5,
            timeInQueueMs: 101,
            averageLatencyMs: 15,
            queueDepth: 5);

        // Assert
        Assert.True(shouldProcess);
    }

    [Fact]
    public void AdaptiveBatchingStrategy_ShouldAdaptBatchSize_BasedOnLatency()
    {
        // Arrange
        var strategy = new AdaptiveBatchingStrategy(
            minBatchSize: 1,
            maxBatchSize: 100,
            maxWaitMs: 50,
            targetLatencyMs: 20.0,
            latencyToleranceFactor: 2.0);

        // Act - Low latency should increase batch size
        var initialBatchSize = strategy.GetOptimalBatchSize(100, 10.0);
        strategy.UpdatePerformanceFeedback(initialBatchSize, 10.0);
        var increasedBatchSize = strategy.GetOptimalBatchSize(100, 10.0);

        // Assert - Batch size should increase
        Assert.True(increasedBatchSize >= initialBatchSize);
    }

    [Fact]
    public void AdaptiveBatchingStrategy_ShouldDecreaseBatchSize_WhenLatencyTooHigh()
    {
        // Arrange
        var strategy = new AdaptiveBatchingStrategy(
            minBatchSize: 1,
            maxBatchSize: 100,
            maxWaitMs: 50,
            targetLatencyMs: 20.0,
            latencyToleranceFactor: 2.0);

        // Warm up with good latency
        for (int i = 0; i < 5; i++)
        {
            strategy.UpdatePerformanceFeedback(50, 15.0);
        }

        var batchSizeBeforeHighLatency = strategy.GetOptimalBatchSize(100, 15.0);

        // Act - High latency should decrease batch size
        for (int i = 0; i < 5; i++)
        {
            strategy.UpdatePerformanceFeedback(50, 50.0); // Latency > 2x target
        }

        var decreasedBatchSize = strategy.GetOptimalBatchSize(100, 50.0);

        // Assert - Batch size should decrease
        Assert.True(decreasedBatchSize < batchSizeBeforeHighLatency);
    }

    [Fact]
    public void AdaptiveBatchingStrategy_ShouldProcessBatch_WhenBackpressureDetected()
    {
        // Arrange
        var strategy = new AdaptiveBatchingStrategy(
            minBatchSize: 10,
            maxBatchSize: 100,
            maxWaitMs: 50,
            targetLatencyMs: 20.0,
            latencyToleranceFactor: 2.0);

        // Act - High queue depth indicates backpressure
        var shouldProcess = strategy.ShouldProcessBatch(
            queuedRequests: 5,
            timeInQueueMs: 5,
            averageLatencyMs: 15,
            queueDepth: 200); // Much higher than optimal batch size

        // Assert
        Assert.True(shouldProcess);
    }

    [Fact]
    public void BucketBatchingStrategy_ShouldReturnCorrectBucketIndex()
    {
        // Arrange
        var bucketSizes = new[] { 32, 64, 128, 256, 512 };
        var strategy = new BucketBatchingStrategy(bucketSizes, maxBatchSize: 100, maxWaitMs: 50);

        // Act & Assert
        Assert.Equal(0, strategy.GetBucketIndex(20));
        Assert.Equal(0, strategy.GetBucketIndex(32));
        Assert.Equal(1, strategy.GetBucketIndex(50));
        Assert.Equal(2, strategy.GetBucketIndex(100));
        Assert.Equal(4, strategy.GetBucketIndex(500));
        Assert.Equal(5, strategy.GetBucketIndex(1000)); // Overflow bucket
    }

    [Fact]
    public void BucketBatchingStrategy_ShouldReturnCorrectBucketSize()
    {
        // Arrange
        var bucketSizes = new[] { 32, 64, 128, 256, 512 };
        var strategy = new BucketBatchingStrategy(bucketSizes, maxBatchSize: 100, maxWaitMs: 50);

        // Act & Assert
        Assert.Equal(32, strategy.GetBucketSize(0));
        Assert.Equal(64, strategy.GetBucketSize(1));
        Assert.Equal(512, strategy.GetBucketSize(4));
        Assert.Equal(1024, strategy.GetBucketSize(5)); // Double for overflow
    }
}
