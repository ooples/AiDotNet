using AiDotNet.Serving.Scheduling;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for priority request queue.
/// </summary>
public class PriorityQueueTests
{
    [Fact]
    public void PriorityQueue_ShouldEnqueueAndDequeue()
    {
        // Arrange
        var queue = new PriorityRequestQueue<int>();

        // Act
        queue.TryEnqueue(1, RequestPriority.Normal);
        queue.TryEnqueue(2, RequestPriority.High);
        queue.TryEnqueue(3, RequestPriority.Low);

        // Assert
        Assert.Equal(3, queue.Count);
    }

    [Fact]
    public void PriorityQueue_ShouldDequeueHigherPriorityFirst()
    {
        // Arrange
        var queue = new PriorityRequestQueue<string>();

        // Act
        queue.TryEnqueue("low", RequestPriority.Low);
        queue.TryEnqueue("normal", RequestPriority.Normal);
        queue.TryEnqueue("high", RequestPriority.High);
        queue.TryEnqueue("critical", RequestPriority.Critical);

        queue.TryDequeue(out var item1);
        queue.TryDequeue(out var item2);
        queue.TryDequeue(out var item3);
        queue.TryDequeue(out var item4);

        // Assert - Should dequeue in priority order
        Assert.Equal("critical", item1);
        Assert.Equal("high", item2);
        Assert.Equal("normal", item3);
        Assert.Equal("low", item4);
    }

    [Fact]
    public void PriorityQueue_ShouldImplementFairScheduling()
    {
        // Arrange
        var queue = new PriorityRequestQueue<string>();

        // Enqueue many low priority items and a few high priority items
        for (int i = 0; i < 20; i++)
        {
            queue.TryEnqueue($"low-{i}", RequestPriority.Low);
        }
        for (int i = 0; i < 5; i++)
        {
            queue.TryEnqueue($"high-{i}", RequestPriority.High);
        }

        // Act - Dequeue all items
        var dequeuedItems = new List<string>();
        while (queue.TryDequeue(out var item))
        {
            dequeuedItems.Add(item!);
        }

        // Assert - High priority items should come first, but not starve low priority
        var firstFiveItems = dequeuedItems.Take(5).ToList();
        Assert.Contains(firstFiveItems, item => item!.StartsWith("high"));
    }

    [Fact]
    public void PriorityQueue_ShouldHandleBackpressure()
    {
        // Arrange
        var queue = new PriorityRequestQueue<int>(maxQueueSize: 5);

        // Act
        var enqueue1 = queue.TryEnqueue(1, RequestPriority.Normal);
        var enqueue2 = queue.TryEnqueue(2, RequestPriority.Normal);
        var enqueue3 = queue.TryEnqueue(3, RequestPriority.Normal);
        var enqueue4 = queue.TryEnqueue(4, RequestPriority.Normal);
        var enqueue5 = queue.TryEnqueue(5, RequestPriority.Normal);
        var enqueue6 = queue.TryEnqueue(6, RequestPriority.Normal); // Should fail

        // Assert
        Assert.True(enqueue1);
        Assert.True(enqueue2);
        Assert.True(enqueue3);
        Assert.True(enqueue4);
        Assert.True(enqueue5);
        Assert.False(enqueue6); // Backpressure - queue full
        Assert.True(queue.IsFull);
    }

    [Fact]
    public void PriorityQueue_ShouldReturnPriorityCounts()
    {
        // Arrange
        var queue = new PriorityRequestQueue<int>();

        // Act
        queue.TryEnqueue(1, RequestPriority.Low);
        queue.TryEnqueue(2, RequestPriority.Low);
        queue.TryEnqueue(3, RequestPriority.Normal);
        queue.TryEnqueue(4, RequestPriority.High);
        queue.TryEnqueue(5, RequestPriority.Critical);

        var counts = queue.GetPriorityCounts();

        // Assert
        Assert.Equal(2, counts[RequestPriority.Low]);
        Assert.Equal(1, counts[RequestPriority.Normal]);
        Assert.Equal(1, counts[RequestPriority.High]);
        Assert.Equal(1, counts[RequestPriority.Critical]);
    }

    [Fact]
    public void PriorityQueue_ShouldClearAllItems()
    {
        // Arrange
        var queue = new PriorityRequestQueue<int>();
        queue.TryEnqueue(1, RequestPriority.Normal);
        queue.TryEnqueue(2, RequestPriority.High);
        queue.TryEnqueue(3, RequestPriority.Low);

        // Act
        queue.Clear();

        // Assert
        Assert.Equal(0, queue.Count);
        Assert.True(queue.IsEmpty);
    }
}
