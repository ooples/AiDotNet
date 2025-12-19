namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// Schedules sequences for continuous batching based on priority and resource constraints.
/// </summary>
/// <remarks>
/// <para>
/// The scheduler determines which sequences should be processed in each iteration,
/// balancing priorities, fairness, and memory constraints.
/// </para>
/// <para><b>For Beginners:</b> The scheduler is like a restaurant host.
///
/// When new customers arrive (requests), the host must decide:
/// - Who gets seated next? (priority)
/// - How many can we serve at once? (batch size)
/// - Do we have enough tables/kitchen capacity? (memory/compute)
/// - Should we pause someone's meal to serve urgent customers? (preemption)
///
/// The scheduler makes these decisions to maximize throughput while
/// ensuring fairness and meeting priority requirements.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
public class BatchScheduler<T>
{
    private readonly BatchSchedulerConfig _config;
    private readonly object _lock = new();

    // Queues for different states
    private readonly PriorityQueue<SequenceState<T>, int> _waitingQueue;
    private readonly List<SequenceState<T>> _runningSequences;
    private readonly List<SequenceState<T>> _preemptedSequences;

    // Resource tracking
    private int _usedCacheSlots;
    private long _usedMemoryBytes;

    /// <summary>
    /// Gets the number of sequences currently waiting to be processed.
    /// </summary>
    public int WaitingCount
    {
        get { lock (_lock) return _waitingQueue.Count; }
    }

    /// <summary>
    /// Gets the number of sequences currently being processed.
    /// </summary>
    public int RunningCount
    {
        get { lock (_lock) return _runningSequences.Count; }
    }

    /// <summary>
    /// Gets the number of preempted sequences waiting to resume.
    /// </summary>
    public int PreemptedCount
    {
        get { lock (_lock) return _preemptedSequences.Count; }
    }

    /// <summary>
    /// Gets the scheduler configuration.
    /// </summary>
    public BatchSchedulerConfig Config => _config;

    /// <summary>
    /// Creates a new batch scheduler with the specified configuration.
    /// </summary>
    public BatchScheduler(BatchSchedulerConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _waitingQueue = new PriorityQueue<SequenceState<T>, int>();
        _runningSequences = new List<SequenceState<T>>();
        _preemptedSequences = new List<SequenceState<T>>();
    }

    /// <summary>
    /// Creates a new batch scheduler with default configuration.
    /// </summary>
    public BatchScheduler()
        : this(new BatchSchedulerConfig())
    {
    }

    /// <summary>
    /// Adds a new sequence to the waiting queue.
    /// </summary>
    /// <param name="sequence">The sequence to add.</param>
    public void AddSequence(SequenceState<T> sequence)
    {
        if (sequence == null)
            throw new ArgumentNullException(nameof(sequence));

        lock (_lock)
        {
            // Negate priority so higher priority comes first (PriorityQueue is min-heap)
            _waitingQueue.Enqueue(sequence, -sequence.Priority);
        }
    }

    /// <summary>
    /// Schedules the next batch of sequences to process.
    /// </summary>
    /// <returns>List of sequences to process in this iteration.</returns>
    public List<SequenceState<T>> ScheduleNextBatch()
    {
        lock (_lock)
        {
            var batch = new List<SequenceState<T>>();

            // Always include already-running sequences (continuous batching).
            // These sequences must be processed every iteration until they complete or are preempted.
            foreach (var seq in _runningSequences)
            {
                if (batch.Count >= _config.MaxBatchSize)
                    break;

                if (seq.Status is SequenceStatus.Generating or SequenceStatus.Prefilling)
                    batch.Add(seq);
            }

            int availableSlots = _config.MaxBatchSize - batch.Count;
            long availableMemory = _config.MaxMemoryBytes - _usedMemoryBytes;

            // First, try to resume preempted sequences (FIFO order)
            var resumedSequences = new List<SequenceState<T>>();
            foreach (var seq in _preemptedSequences.ToList())
            {
                if (batch.Count >= availableSlots) break;

                long memNeeded = EstimateMemoryForSequence(seq);
                if (memNeeded <= availableMemory)
                {
                    seq.Status = SequenceStatus.Generating;
                    batch.Add(seq);
                    resumedSequences.Add(seq);
                    availableMemory -= memNeeded;
                    availableSlots--;
                }
            }

            foreach (var seq in resumedSequences)
            {
                _preemptedSequences.Remove(seq);
                _runningSequences.Add(seq);
            }

            // Then, add new sequences from waiting queue
            while (_waitingQueue.Count > 0 && batch.Count < _config.MaxBatchSize)
            {
                // Peek without removing
                if (!_waitingQueue.TryPeek(out var nextSeq, out _))
                    break;

                long memNeeded = EstimateMemoryForSequence(nextSeq);

                // Check if we have resources
                if (memNeeded > availableMemory)
                {
                    // Try to preempt lower priority sequences
                    if (_config.AllowPreemption && TryPreemptForSequence(nextSeq, memNeeded - availableMemory))
                    {
                        availableMemory = _config.MaxMemoryBytes - _usedMemoryBytes;
                    }
                    else
                    {
                        break; // Can't fit this sequence, stop scheduling
                    }
                }

                // Check cache slot availability
                if (_usedCacheSlots >= _config.MaxCacheSlots)
                {
                    if (_config.AllowPreemption && TryPreemptForCacheSlot())
                    {
                        // Slot freed
                    }
                    else
                    {
                        break;
                    }
                }

                // Remove from queue and add to batch
                _waitingQueue.Dequeue();
                nextSeq.Status = SequenceStatus.Prefilling;
                nextSeq.CacheSlot = AllocateCacheSlot();
                batch.Add(nextSeq);
                _runningSequences.Add(nextSeq);
                _usedMemoryBytes += memNeeded;
            }

            // Assign batch indices
            for (int i = 0; i < batch.Count; i++)
            {
                batch[i].BatchIndex = i;
            }

            return batch;
        }
    }

    /// <summary>
    /// Gets all currently running sequences.
    /// </summary>
    public List<SequenceState<T>> GetRunningSequences()
    {
        lock (_lock)
        {
            return new List<SequenceState<T>>(_runningSequences);
        }
    }

    /// <summary>
    /// Marks a sequence as completed and removes it from the running set.
    /// </summary>
    /// <param name="sequence">The completed sequence.</param>
    public void CompleteSequence(SequenceState<T> sequence)
    {
        if (sequence == null) return;

        lock (_lock)
        {
            _runningSequences.Remove(sequence);
            if (sequence.CacheSlot >= 0)
            {
                FreeCacheSlot(sequence.CacheSlot);
                sequence.CacheSlot = -1;
            }
            _usedMemoryBytes -= EstimateMemoryForSequence(sequence);
            sequence.BatchIndex = -1;
        }
    }

    /// <summary>
    /// Preempts a running sequence, moving it to the preempted queue.
    /// </summary>
    /// <param name="sequence">The sequence to preempt.</param>
    public void PreemptSequence(SequenceState<T> sequence)
    {
        if (sequence == null) return;

        lock (_lock)
        {
            if (_runningSequences.Remove(sequence))
            {
                sequence.Status = SequenceStatus.Paused;
                sequence.BatchIndex = -1;
                _preemptedSequences.Add(sequence);
                _usedMemoryBytes -= EstimateMemoryForSequence(sequence);
                // Note: Cache slot is retained for quick resume
            }
        }
    }

    /// <summary>
    /// Cancels a sequence, removing it from all queues.
    /// </summary>
    /// <param name="sequenceId">The ID of the sequence to cancel.</param>
    /// <returns>True if the sequence was found and cancelled.</returns>
    public bool CancelSequence(long sequenceId)
    {
        lock (_lock)
        {
            // Check running sequences
            var running = _runningSequences.Find(s => s.SequenceId == sequenceId);
            if (running != null)
            {
                running.Cancel();
                _runningSequences.Remove(running);
                if (running.CacheSlot >= 0)
                {
                    FreeCacheSlot(running.CacheSlot);
                }
                _usedMemoryBytes -= EstimateMemoryForSequence(running);
                return true;
            }

            // Check preempted sequences
            var preempted = _preemptedSequences.Find(s => s.SequenceId == sequenceId);
            if (preempted != null)
            {
                preempted.Cancel();
                _preemptedSequences.Remove(preempted);
                if (preempted.CacheSlot >= 0)
                {
                    FreeCacheSlot(preempted.CacheSlot);
                }
                return true;
            }

            // Can't efficiently remove from priority queue, mark for removal
            return false;
        }
    }

    /// <summary>
    /// Gets statistics about the scheduler state.
    /// </summary>
    public SchedulerStatistics GetStatistics()
    {
        lock (_lock)
        {
            return new SchedulerStatistics
            {
                WaitingSequences = _waitingQueue.Count,
                RunningSequences = _runningSequences.Count,
                PreemptedSequences = _preemptedSequences.Count,
                UsedCacheSlots = _usedCacheSlots,
                MaxCacheSlots = _config.MaxCacheSlots,
                UsedMemoryBytes = _usedMemoryBytes,
                MaxMemoryBytes = _config.MaxMemoryBytes,
                MemoryUtilization = _config.MaxMemoryBytes > 0
                    ? (double)_usedMemoryBytes / _config.MaxMemoryBytes
                    : 0
            };
        }
    }

    /// <summary>
    /// Reorders running sequences by priority.
    /// </summary>
    public void ReorderByPriority()
    {
        lock (_lock)
        {
            _runningSequences.Sort((a, b) => b.Priority.CompareTo(a.Priority));
            for (int i = 0; i < _runningSequences.Count; i++)
            {
                _runningSequences[i].BatchIndex = i;
            }
        }
    }

    private bool TryPreemptForSequence(SequenceState<T> newSequence, long memoryNeeded)
    {
        // Find lowest priority running sequence that can be preempted
        var candidates = _runningSequences
            .Where(s => s.Priority < newSequence.Priority)
            .OrderBy(s => s.Priority)
            .ThenByDescending(s => s.GeneratedLength) // Prefer preempting further along
            .ToList();

        long freedMemory = 0;
        var toPreempt = new List<SequenceState<T>>();

        foreach (var candidate in candidates)
        {
            if (freedMemory >= memoryNeeded) break;

            toPreempt.Add(candidate);
            freedMemory += EstimateMemoryForSequence(candidate);
        }

        if (freedMemory >= memoryNeeded)
        {
            foreach (var seq in toPreempt)
            {
                PreemptSequence(seq);
            }
            return true;
        }

        return false;
    }

    private bool TryPreemptForCacheSlot()
    {
        // Find lowest priority running sequence
        var candidate = _runningSequences
            .OrderBy(s => s.Priority)
            .FirstOrDefault();

        if (candidate != null)
        {
            PreemptSequence(candidate);
            return true;
        }

        return false;
    }

    private long EstimateMemoryForSequence(SequenceState<T> sequence)
    {
        // Estimate memory based on sequence length and model configuration
        // This is a simplified estimate; real implementation would use model config
        int seqLen = sequence.TokenIds.Count + sequence.MaxNewTokens;
        long elementsPerLayer = _config.NumHeads * seqLen * _config.HeadDimension * 2; // K and V
        return elementsPerLayer * _config.NumLayers * sizeof(float);
    }

    private int AllocateCacheSlot()
    {
        // Simple linear allocation
        return _usedCacheSlots++;
    }

    private void FreeCacheSlot(int slot)
    {
        _usedCacheSlots = Math.Max(0, _usedCacheSlots - 1);
    }
}

/// <summary>
/// Configuration for the batch scheduler.
/// </summary>
public class BatchSchedulerConfig
{
    /// <summary>
    /// Maximum number of sequences in a batch.
    /// </summary>
    public int MaxBatchSize { get; set; } = 8;

    /// <summary>
    /// Maximum number of KV-cache slots available.
    /// </summary>
    public int MaxCacheSlots { get; set; } = 256;

    /// <summary>
    /// Maximum memory available for KV-cache (bytes).
    /// </summary>
    public long MaxMemoryBytes { get; set; } = 8L * 1024 * 1024 * 1024; // 8GB default

    /// <summary>
    /// Whether to allow preempting lower-priority sequences.
    /// </summary>
    public bool AllowPreemption { get; set; } = true;

    /// <summary>
    /// Scheduling policy to use.
    /// </summary>
    public SchedulingPolicy Policy { get; set; } = SchedulingPolicy.Priority;

    /// <summary>
    /// Number of attention heads (for memory estimation).
    /// </summary>
    public int NumHeads { get; set; } = 32;

    /// <summary>
    /// Dimension of each attention head (for memory estimation).
    /// </summary>
    public int HeadDimension { get; set; } = 128;

    /// <summary>
    /// Number of transformer layers (for memory estimation).
    /// </summary>
    public int NumLayers { get; set; } = 32;

    /// <summary>
    /// Creates config for a specific model.
    /// </summary>
    public static BatchSchedulerConfig ForModel(string modelName, int maxBatchSize = 8)
    {
        return modelName.ToLowerInvariant() switch
        {
            "llama-7b" => new BatchSchedulerConfig
            {
                MaxBatchSize = maxBatchSize,
                NumHeads = 32,
                HeadDimension = 128,
                NumLayers = 32,
                MaxMemoryBytes = 4L * 1024 * 1024 * 1024
            },
            "llama-13b" => new BatchSchedulerConfig
            {
                MaxBatchSize = maxBatchSize,
                NumHeads = 40,
                HeadDimension = 128,
                NumLayers = 40,
                MaxMemoryBytes = 8L * 1024 * 1024 * 1024
            },
            "llama-70b" => new BatchSchedulerConfig
            {
                MaxBatchSize = Math.Min(maxBatchSize, 4),
                NumHeads = 64,
                HeadDimension = 128,
                NumLayers = 80,
                MaxMemoryBytes = 16L * 1024 * 1024 * 1024
            },
            _ => new BatchSchedulerConfig { MaxBatchSize = maxBatchSize }
        };
    }
}

/// <summary>
/// Scheduling policies for batch scheduling.
/// </summary>
public enum SchedulingPolicy
{
    /// <summary>First-come, first-served ordering.</summary>
    FCFS,

    /// <summary>Priority-based ordering (higher priority first).</summary>
    Priority,

    /// <summary>Shortest job first (shorter sequences first).</summary>
    ShortestFirst,

    /// <summary>Fair scheduling with time-based preemption.</summary>
    Fair
}

/// <summary>
/// Statistics about the scheduler state.
/// </summary>
public class SchedulerStatistics
{
    /// <summary>Number of sequences waiting to be processed.</summary>
    public int WaitingSequences { get; set; }

    /// <summary>Number of sequences currently being processed.</summary>
    public int RunningSequences { get; set; }

    /// <summary>Number of preempted sequences.</summary>
    public int PreemptedSequences { get; set; }

    /// <summary>Number of cache slots in use.</summary>
    public int UsedCacheSlots { get; set; }

    /// <summary>Maximum number of cache slots.</summary>
    public int MaxCacheSlots { get; set; }

    /// <summary>Memory currently in use (bytes).</summary>
    public long UsedMemoryBytes { get; set; }

    /// <summary>Maximum memory available (bytes).</summary>
    public long MaxMemoryBytes { get; set; }

    /// <summary>Memory utilization (0-1).</summary>
    public double MemoryUtilization { get; set; }

    /// <summary>Cache slot utilization (0-1).</summary>
    public double SlotUtilization => MaxCacheSlots > 0
        ? (double)UsedCacheSlots / MaxCacheSlots
        : 0;

    /// <summary>Total sequences in system.</summary>
    public int TotalSequences => WaitingSequences + RunningSequences + PreemptedSequences;
}
