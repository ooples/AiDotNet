using System.Collections.Concurrent;

namespace AiDotNet.Serving.Scheduling;

/// <summary>
/// Priority-based request queue that implements fair scheduling with backpressure handling.
/// Requests are dequeued in priority order, with fairness guarantees to prevent starvation.
/// </summary>
/// <typeparam name="T">The type of items in the queue</typeparam>
public class PriorityRequestQueue<T>
{
    private readonly ConcurrentQueue<T>[] _queues;
    private readonly int[] _dequeueCounts;
    private readonly int _maxQueueSize;
    private int _totalCount;
    private readonly object _lock = new();

    // Fair scheduling weights: how many items to dequeue from each priority before moving to lower priority
    private readonly int[] _fairnessWeights = { 8, 4, 2, 1 }; // Critical:High:Normal:Low = 8:4:2:1

    /// <summary>
    /// Initializes a new instance of the PriorityRequestQueue.
    /// </summary>
    /// <param name="maxQueueSize">Maximum total queue size for backpressure handling (0 = unlimited)</param>
    public PriorityRequestQueue(int maxQueueSize = 0)
    {
        _maxQueueSize = maxQueueSize;
        _queues = new ConcurrentQueue<T>[4]; // One queue per priority level
        _dequeueCounts = new int[4];

        for (int i = 0; i < _queues.Length; i++)
        {
            _queues[i] = new ConcurrentQueue<T>();
            _dequeueCounts[i] = 0;
        }
    }

    /// <summary>
    /// Gets the total number of items in all queues.
    /// </summary>
    public int Count => _totalCount;

    /// <summary>
    /// Gets whether the queue is empty.
    /// </summary>
    public bool IsEmpty => _totalCount == 0;

    /// <summary>
    /// Gets whether the queue has reached maximum capacity.
    /// </summary>
    public bool IsFull => _maxQueueSize > 0 && _totalCount >= _maxQueueSize;

    /// <summary>
    /// Enqueues an item with the specified priority.
    /// </summary>
    /// <param name="item">The item to enqueue</param>
    /// <param name="priority">The priority level</param>
    /// <returns>True if the item was enqueued; false if the queue is full (backpressure)</returns>
    public bool TryEnqueue(T item, RequestPriority priority)
    {
        lock (_lock)
        {
            if (IsFull)
                return false;

            _queues[(int)priority].Enqueue(item);
            _totalCount++;
            return true;
        }
    }

    /// <summary>
    /// Attempts to dequeue an item using fair scheduling across priority levels.
    /// </summary>
    /// <param name="item">The dequeued item</param>
    /// <returns>True if an item was dequeued; false if the queue is empty</returns>
    public bool TryDequeue(out T? item)
    {
        item = default;

        lock (_lock)
        {
            if (IsEmpty)
                return false;

            // Try to dequeue from queues in priority order, respecting fairness weights
            for (int priorityIndex = _queues.Length - 1; priorityIndex >= 0; priorityIndex--)
            {
                var queue = _queues[priorityIndex];
                if (queue.IsEmpty)
                    continue;

                // Check if we've exceeded the fairness quota for this priority
                int weight = _fairnessWeights[priorityIndex];
                if (_dequeueCounts[priorityIndex] < weight)
                {
                    if (queue.TryDequeue(out item))
                    {
                        _dequeueCounts[priorityIndex]++;
                        _totalCount--;

                        // Reset counts when we complete a full cycle
                        if (_dequeueCounts.Sum() >= _fairnessWeights.Sum())
                        {
                            Array.Clear(_dequeueCounts, 0, _dequeueCounts.Length);
                        }

                        return true;
                    }
                }
            }

            // If we couldn't dequeue respecting fairness, reset and try again
            Array.Clear(_dequeueCounts, 0, _dequeueCounts.Length);

            // Simple priority dequeue as fallback
            for (int priorityIndex = _queues.Length - 1; priorityIndex >= 0; priorityIndex--)
            {
                if (_queues[priorityIndex].TryDequeue(out item))
                {
                    _totalCount--;
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>
    /// Gets the count of items at each priority level.
    /// </summary>
    /// <returns>Dictionary mapping priority to count</returns>
    public Dictionary<RequestPriority, int> GetPriorityCounts()
    {
        return new Dictionary<RequestPriority, int>
        {
            [RequestPriority.Critical] = _queues[3].Count,
            [RequestPriority.High] = _queues[2].Count,
            [RequestPriority.Normal] = _queues[1].Count,
            [RequestPriority.Low] = _queues[0].Count
        };
    }

    /// <summary>
    /// Clears all items from the queue.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            foreach (var queue in _queues)
            {
                while (queue.TryDequeue(out _)) { }
            }
            _totalCount = 0;
            Array.Clear(_dequeueCounts, 0, _dequeueCounts.Length);
        }
    }
}
