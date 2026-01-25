using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides an in-memory implementation of distributed communication for testing and single-machine scenarios.
/// </summary>
/// <remarks>
/// <para><b>⚠️ WARNING - Static Shared State:</b>
/// This implementation uses STATIC shared dictionaries to simulate cross-process communication.
/// This design has important implications:
/// </para>
/// <list type="bullet">
/// <item>All instances in the same process share the SAME static state</item>
/// <item>Unit tests using this backend CANNOT run in parallel without isolation via environmentId</item>
/// <item>Multiple training sessions in the same process can interfere unless using unique environmentIds</item>
/// <item>NOT suitable for production multi-process scenarios - use MPI/NCCL backends instead</item>
/// </list>
/// <para>
/// The static state includes: _sharedBuffers, _barrierCounters, _barrierGenerations, _operationCounters, _messageQueues.
/// These are namespaced by environmentId to enable concurrent independent sessions, but tests must ensure
/// unique environmentIds or run serially.
/// </para>
/// <para>
/// This backend simulates multiple processes by using shared memory and locks. It's perfect for testing
/// distributed code without needing actual MPI infrastructure or multiple machines. All "processes" run
/// within the same application instance, using static shared memory to simulate cross-process communication.
/// </para>
/// <para><b>For Beginners:</b> This is a "fake" distributed system that runs on a single machine.
///
/// It's perfect for testing your distributed code without needing multiple GPUs or machines.
/// Think of it as a practice mode - it simulates distributed behavior but everything runs
/// in one process.
///
/// Use this when:
/// - Testing distributed code locally
/// - Debugging distributed training logic
/// - Running unit tests
/// - Learning how distributed training works
///
/// For production with actual multiple GPUs/machines, use an MPI-based backend instead.
///
/// Example:
/// <code>
/// // Create a simulated distributed environment with 4 "processes"
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// backend.Initialize();
///
/// // Now you can test distributed operations locally
/// var data = new Vector&lt;double&gt;(new[] { 1.0, 2.0, 3.0 });
/// backend.AllReduce(data, ReductionOperation.Sum);
/// // data now contains the sum from all 4 simulated processes
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
public class InMemoryCommunicationBackend<T> : CommunicationBackendBase<T>
{
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly string _environmentId;

    // Shared state for simulating collective operations
    //
    // IMPORTANT: While these dictionaries are static (shared across all InMemoryCommunicationBackend instances),
    // they are NAMESPACED by _environmentId to enable concurrent, independent distributed training sessions
    // within the same process. Each session uses a unique _environmentId prefix in all dictionary keys
    // (e.g., "{environmentId}_allreduce_{counter}", "{environmentId}_barrier_{generation}").
    //
    // This design allows:
    // - Multiple unit tests running in parallel without interference
    // - Multiple concurrent training sessions with isolated communication
    // - Clean separation between different distributed training contexts
    //
    // In a real implementation, this would be handled by the MPI backend's process isolation.
    //
    // NOTE ON THREAD SAFETY AND CONCURRENT SESSIONS:
    // These static dictionaries ARE safe for concurrent training sessions because all keys
    // include the unique _environmentId prefix (see line 61). Different training sessions
    // get different environment IDs, so they operate on completely separate dictionary entries.
    // The _globalLock ensures thread-safe access to the shared dictionaries.
    private static readonly object _globalLock = new object();
    private static readonly Dictionary<string, List<Vector<T>>> _sharedBuffers = new();
    private static readonly Dictionary<string, int> _pendingConsumers = new();
    private static readonly Dictionary<string, int> _barrierCounters = new();
    private static readonly Dictionary<string, int> _barrierReleaseCounts = new();
    private static readonly Dictionary<string, int> _barrierGenerations = new();
    private static readonly Dictionary<string, int> _operationCounters = new();

    // Point-to-point message queues for Send/Receive operations
    // Key format: "{environmentId}_msg_{sourceRank}_{destRank}_{tag}"
    private static readonly Dictionary<string, Queue<Vector<T>>> _messageQueues = new();

    private const int BarrierTimeoutMs = 30000; // 30 seconds
    private const int MessageTimeoutMs = 30000; // 30 seconds for point-to-point


    /// <inheritdoc/>
    public override int Rank => _rank;

    /// <inheritdoc/>
    public override int WorldSize => _worldSize;

    /// <summary>
    /// Creates a new in-memory communication backend.
    /// </summary>
    /// <remarks>
    /// <para>
    /// You create one of these for each simulated "process". If you want to simulate 4 GPUs,
    /// you create 4 instances with ranks 0, 1, 2, 3, all with worldSize=4.
    /// </para>
    /// <para><b>For Beginners:</b> This creates one simulated process in your fake distributed system.
    ///
    /// Parameters:
    /// - rank: The ID of this process (0-based). Each process needs a unique rank.
    /// - worldSize: How many processes total are in your simulated system.
    ///
    /// Example: To simulate 4 GPUs, create 4 backends:
    /// <code>
    /// var process0 = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
    /// var process1 = new InMemoryCommunicationBackend&lt;double&gt;(rank: 1, worldSize: 4);
    /// var process2 = new InMemoryCommunicationBackend&lt;double&gt;(rank: 2, worldSize: 4);
    /// var process3 = new InMemoryCommunicationBackend&lt;double&gt;(rank: 3, worldSize: 4);
    /// </code>
    /// </para>
    /// </remarks>
    /// <param name="rank">The rank (ID) of this simulated process (0-based)</param>
    /// <param name="worldSize">The total number of simulated processes</param>
    /// <param name="environmentId">Optional environment ID for isolation (defaults to "default" for backwards compatibility)</param>
    /// <exception cref="ArgumentException">Thrown if rank or worldSize are invalid</exception>
    public InMemoryCommunicationBackend(int rank, int worldSize, string environmentId = "default")
    {
        if (rank < 0 || rank >= worldSize)
        {
            throw new ArgumentException(
                $"Invalid rank {rank}. Must be between 0 and {worldSize - 1}.",
                nameof(rank));
        }

        if (worldSize <= 0)
        {
            throw new ArgumentException(
                $"Invalid worldSize {worldSize}. Must be positive.",
                nameof(worldSize));
        }

        if (string.IsNullOrWhiteSpace(environmentId))
        {
            throw new ArgumentException("Environment ID cannot be null or empty.", nameof(environmentId));
        }

        _rank = rank;
        _worldSize = worldSize;
        _environmentId = environmentId;

        // Initialize environment-specific counters
        lock (_globalLock)
        {
            // Use ContainsKey check for .NET Framework 4.62 compatibility (TryAdd was added in .NET Core 2.0)
            if (!_barrierGenerations.ContainsKey(_environmentId))
            {
                _barrierGenerations[_environmentId] = 0;
            }
            if (!_operationCounters.ContainsKey(_environmentId))
            {
                _operationCounters[_environmentId] = 0;
            }
        }
    }

    /// <inheritdoc/>
    protected override void OnInitialize()
    {
        // Base class handles initialization state
        // No additional initialization required for in-memory backend
    }

    /// <inheritdoc/>
    protected override void OnShutdown()
    {
        lock (_globalLock)
        {
            // Clear only this environment's shared state
            ClearEnvironmentState(_environmentId);
        }
    }

    /// <summary>
    /// Clears all shared state for a specific environment.
    /// Useful for test cleanup and isolation.
    /// </summary>
    /// <param name="environmentId">The environment ID to clear</param>
    public static void ClearEnvironment(string environmentId)
    {
        if (string.IsNullOrWhiteSpace(environmentId))
        {
            return;
        }

        lock (_globalLock)
        {
            ClearEnvironmentState(environmentId);
        }
    }

    private static void ClearEnvironmentState(string environmentId)
    {
        // Remove all keys that belong to this environment
        var buffersToRemove = _sharedBuffers.Keys.Where(k => k.StartsWith($"{environmentId}_")).ToList();
        foreach (var key in buffersToRemove)
        {
            _sharedBuffers.Remove(key);
        }

        var barriersToRemove = _barrierCounters.Keys.Where(k => k.StartsWith($"{environmentId}_")).ToList();
        foreach (var key in barriersToRemove)
        {
            _barrierCounters.Remove(key);
        }

        var messagesToRemove = _messageQueues.Keys.Where(k => k.StartsWith($"{environmentId}_")).ToList();
        foreach (var key in messagesToRemove)
        {
            _messageQueues.Remove(key);
        }

        var consumersToRemove = _pendingConsumers.Keys.Where(k => k.StartsWith($"{environmentId}_")).ToList();
        foreach (var key in consumersToRemove)
        {
            _pendingConsumers.Remove(key);
        }

        var releaseCountsToRemove = _barrierReleaseCounts.Keys.Where(k => k.StartsWith($"{environmentId}_")).ToList();
        foreach (var key in releaseCountsToRemove)
        {
            _barrierReleaseCounts.Remove(key);
        }

        // Reset environment counters
        _barrierGenerations[environmentId] = 0;
        _operationCounters[environmentId] = 0;
    }

    /// <inheritdoc/>
    public override void Barrier()
    {
        EnsureInitialized();

        lock (_globalLock)
        {
            // Use shared barrier generation counter so all ranks synchronize on same key
            int currentGeneration = _barrierGenerations[_environmentId];

            // Use environment-prefixed barrier ID
            string barrierId = $"{_environmentId}_barrier_{currentGeneration}";

            if (!_barrierCounters.ContainsKey(barrierId))
            {
                _barrierCounters[barrierId] = 0;
            }

            _barrierCounters[barrierId]++;

            var startTime = DateTime.UtcNow;

            try
            {
                // Wait until all processes have reached the barrier
                while (_barrierCounters.ContainsKey(barrierId) && _barrierCounters[barrierId] < _worldSize)
                {
                    Monitor.Wait(_globalLock, 10);
                    if ((DateTime.UtcNow - startTime).TotalMilliseconds > BarrierTimeoutMs)
                    {
                        throw new TimeoutException($"Barrier timeout after {BarrierTimeoutMs}ms. Only {_barrierCounters[barrierId]} of {_worldSize} processes reached the barrier.");
                    }
                }
            }
            finally
            {
                // CRITICAL: Always wake other ranks (especially on timeout for fail-fast)
                Monitor.PulseAll(_globalLock);

                // Track how many ranks have exited the barrier
                int released = _barrierReleaseCounts.TryGetValue(barrierId, out var current)
                    ? current + 1
                    : 1;
                _barrierReleaseCounts[barrierId] = released;

                // Only remove the barrier entries when ALL ranks have exited
                // This prevents KeyNotFoundException when ranks wake up and re-check the while condition
                if (released == _worldSize)
                {
                    _barrierCounters.Remove(barrierId);
                    _barrierReleaseCounts.Remove(barrierId);
                    _barrierGenerations[_environmentId]++;
                }
            }
        }
    }

    /// <inheritdoc/>
    /// <summary>
    /// Performs an AllReduce operation, combining data from all processes.
    /// </summary>
    /// <remarks>
    /// <para><b>IMPORTANT - In-Place Modification:</b>
    /// This method modifies the `data` parameter IN-PLACE. Unlike other collective operations
    /// (Broadcast, AllGather, Scatter, ReduceScatter) which return new vectors, AllReduce
    /// follows the standard MPI convention of modifying the input vector directly.
    /// </para>
    /// <para>
    /// This design choice:
    /// - Matches standard MPI AllReduce behavior (in-place modification)
    /// - Reduces memory allocations for large gradient vectors
    /// - Is consistent with ICommunicationBackend interface contract
    /// </para>
    /// <para><b>Thread Safety:</b>
    /// The implementation clones data before storing in shared buffers to prevent
    /// race conditions during the synchronization phase.
    /// </para>
    /// </remarks>
    public override void AllReduce(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();

        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        // For single process, no communication needed - data is already the "reduced" result
        // (e.g., Average of one value is itself, Sum of one value is itself)
        if (_worldSize == 1)
        {
            // No modification needed - data already contains the correct result
            return;
        }

        lock (_globalLock)
        {
            // Use shared operation counter so all ranks target same buffer key
            int currentCounter = _operationCounters[_environmentId];

            // Use environment-prefixed buffer ID
            string bufferId = $"{_environmentId}_allreduce_{currentCounter}";

            // Initialize shared buffer
            if (!_sharedBuffers.ContainsKey(bufferId))
            {
                _sharedBuffers[bufferId] = new List<Vector<T>>();
            }

            // Contribute local data
            _sharedBuffers[bufferId].Add(data.Clone());

            var startTime = DateTime.UtcNow;

            // Initialize pending consumers counter to track how many ranks need to read the result
            // This prevents race condition where rank 0 removes the buffer before other ranks finish reading
            if (!_pendingConsumers.ContainsKey(bufferId))
            {
                _pendingConsumers[bufferId] = _worldSize;
            }

            try
            {
                // Wait until all processes have contributed
                while (_sharedBuffers.ContainsKey(bufferId) && _sharedBuffers[bufferId].Count < _worldSize)
                {
                    Monitor.Wait(_globalLock, 10);
                    if ((DateTime.UtcNow - startTime).TotalMilliseconds > BarrierTimeoutMs)
                    {
                        throw new TimeoutException($"AllReduce timeout after {BarrierTimeoutMs}ms. Only {_sharedBuffers[bufferId].Count} of {_worldSize} processes contributed.");
                    }
                }

                // Perform reduction
                if (_sharedBuffers.ContainsKey(bufferId))
                {
                    var allData = _sharedBuffers[bufferId];
                    var result = PerformReduction(allData, operation);

                    // Copy result back to input data
                    for (int i = 0; i < data.Length; i++)
                    {
                        data[i] = result[i];
                    }
                }
            }
            finally
            {
                // CRITICAL: Decrement consumer count even on exception to prevent buffer leaks
                // This must happen in finally to ensure cleanup even if reduction or copy operations fail
                if (_pendingConsumers.ContainsKey(bufferId))
                {
                    int remaining = --_pendingConsumers[bufferId];
                    if (remaining == 0)
                    {
                        // All ranks have consumed the result, safe to cleanup
                        _sharedBuffers.Remove(bufferId);
                        _pendingConsumers.Remove(bufferId);
                        _operationCounters[_environmentId]++;
                    }
                }

                // CRITICAL: PulseAll must happen even on timeout to wake other waiting processes
                // Without this, a timeout in one process causes deadlock in others waiting at Monitor.Wait
                Monitor.PulseAll(_globalLock);
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> AllGather(Vector<T> sendData)
    {
        EnsureInitialized();

        if (sendData == null)
        {
            throw new ArgumentNullException(nameof(sendData));
        }

        // For single process, just return a copy
        if (_worldSize == 1)
        {
            return sendData.Clone();
        }

        lock (_globalLock)
        {
            // Use shared operation counter so all ranks target same buffer key
            int currentCounter = _operationCounters[_environmentId];

            // Use environment-prefixed buffer ID
            string bufferId = $"{_environmentId}_allgather_{currentCounter}";

            // Initialize shared buffer
            if (!_sharedBuffers.ContainsKey(bufferId))
            {
                _sharedBuffers[bufferId] = new List<Vector<T>>(new Vector<T>[_worldSize]);
            }

            // Initialize pending consumers counter to track how many ranks need to read the result
            // This prevents race condition where cleanup happens before other ranks finish reading
            if (!_pendingConsumers.ContainsKey(bufferId))
            {
                _pendingConsumers[bufferId] = _worldSize;
            }

            // Contribute local data
            _sharedBuffers[bufferId][_rank] = sendData.Clone();

            try
            {
                var startTime = DateTime.UtcNow;

                // Wait until all processes have contributed
                while (true)
                {
                    int contributedCount = _sharedBuffers[bufferId].Count(v => v != null);
                    if (contributedCount >= _worldSize)
                    {
                        break;
                    }

                    Monitor.Wait(_globalLock, 10);

                    if ((DateTime.UtcNow - startTime).TotalMilliseconds > BarrierTimeoutMs)
                    {
                        throw new TimeoutException(
                            $"AllGather timeout after {BarrierTimeoutMs}ms. Only {contributedCount} of {_worldSize} processes contributed.");
                    }
                }

                // Concatenate all data
                var allData = _sharedBuffers[bufferId];
                int totalLength = allData.Sum(v => v.Length);
                var result = new T[totalLength];
                int offset = 0;

                for (int i = 0; i < _worldSize; i++)
                {
                    var data = allData[i];
                    Array.Copy(data.ToArray(), 0, result, offset, data.Length);
                    offset += data.Length;
                }

                return new Vector<T>(result);
            }
            finally
            {
                // CRITICAL: Decrement consumer count even on exception to prevent buffer leaks
                // This must happen in finally to ensure cleanup even if concatenation or copy operations fail
                if (_pendingConsumers.ContainsKey(bufferId))
                {
                    int remaining = --_pendingConsumers[bufferId];
                    if (remaining == 0)
                    {
                        // All ranks have consumed the result, safe to cleanup
                        _sharedBuffers.Remove(bufferId);
                        _pendingConsumers.Remove(bufferId);
                        _operationCounters[_environmentId]++;
                    }
                }

                // CRITICAL: PulseAll must happen even on timeout to wake other waiting processes
                // Without this, a timeout in one process causes deadlock in others waiting at Monitor.Wait
                Monitor.PulseAll(_globalLock);
            }
        }
    }

    /// <inheritdoc/>
    /// <summary>
    /// Broadcasts data from the root process to all other processes.
    /// </summary>
    /// <remarks>
    /// <para><b>IMPORTANT - Returns New Vector (No In-Place Modification):</b>
    /// Unlike AllReduce which modifies the input in-place, Broadcast returns a NEW vector
    /// and does NOT modify the `data` parameter. The input `data` is only meaningful on the
    /// root process and is ignored on non-root processes.
    /// </para>
    /// <para>
    /// This design:
    /// - Follows standard MPI Broadcast semantics (returns broadcasted data)
    /// - Prevents unintended side effects on non-root processes
    /// - Is consistent with ICommunicationBackend interface contract
    /// </para>
    /// </remarks>
    public override Vector<T> Broadcast(Vector<T> data, int root = 0)
    {
        EnsureInitialized();

        if (root < 0 || root >= _worldSize)
        {
            throw new ArgumentException($"Invalid root {root}. Must be between 0 and {_worldSize - 1}.");
        }

        // For single process, return a clone (maintains consistency with multi-process behavior)
        if (_worldSize == 1)
        {
            return data?.Clone() ?? throw new ArgumentNullException(nameof(data));
        }

        lock (_globalLock)
        {
            // Use shared operation counter so all ranks target same buffer key
            int currentCounter = _operationCounters[_environmentId];

            // Use environment-prefixed buffer ID
            string bufferId = $"{_environmentId}_broadcast_{currentCounter}";
            Vector<T> result;
            List<Vector<T>>? buffer;

            // Root process stores the data
            if (_rank == root)
            {
                if (data == null)
                {
                    throw new ArgumentNullException(nameof(data), "Data cannot be null on root process.");
                }
                _sharedBuffers[bufferId] = new List<Vector<T>> { data.Clone() };
                _pendingConsumers[bufferId] = _worldSize;
            }

            var startTime = DateTime.UtcNow;

            try
            {
                // Wait for root to store data
                while (!_sharedBuffers.ContainsKey(bufferId))
                {
                    Monitor.Wait(_globalLock, 10);
                    if ((DateTime.UtcNow - startTime).TotalMilliseconds > BarrierTimeoutMs)
                    {
                        throw new TimeoutException($"Broadcast timeout after {BarrierTimeoutMs}ms waiting for root rank {root} to provide data.");
                    }
                }

                // Retrieve the broadcasted data
                if (_sharedBuffers.ContainsKey(bufferId))
                {
                    buffer = _sharedBuffers[bufferId];
                    if (buffer != null && buffer.Count > 0)
                    {
                        result = buffer[0].Clone();
                    }
                    else
                    {
                        throw new InvalidOperationException("Broadcast buffer is null or empty after synchronization.");
                    }
                }
                else
                {
                    throw new InvalidOperationException("Broadcast buffer was removed before data could be retrieved.");
                }

                return result;
            }
            finally
            {
                // CRITICAL: Decrement consumer count even on exception to prevent buffer leaks
                // This must happen in finally to ensure cleanup even if Clone() or other operations fail
                if (_pendingConsumers.ContainsKey(bufferId))
                {
                    int remaining = --_pendingConsumers[bufferId];
                    if (remaining == 0)
                    {
                        _sharedBuffers.Remove(bufferId);
                        _pendingConsumers.Remove(bufferId);
                        _operationCounters[_environmentId]++;
                    }
                }

                // CRITICAL: PulseAll must happen even on timeout to wake other waiting processes
                Monitor.PulseAll(_globalLock);
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Scatter(Vector<T> sendData, int root = 0)
    {
        EnsureInitialized();

        if (root < 0 || root >= _worldSize)
        {
            throw new ArgumentException($"Invalid root {root}. Must be between 0 and {_worldSize - 1}.");
        }

        // For single process, just return a copy
        if (_worldSize == 1)
        {
            return sendData?.Clone() ?? throw new ArgumentNullException(nameof(sendData));
        }

        lock (_globalLock)
        {
            // Use shared operation counter so all ranks target same buffer key
            int currentCounter = _operationCounters[_environmentId];

            // Use environment-prefixed buffer ID
            string bufferId = $"{_environmentId}_scatter_{currentCounter}";

            // Root process splits and stores the data
            if (_rank == root)
            {
                if (sendData == null)
                {
                    throw new ArgumentNullException(nameof(sendData), "Data cannot be null on root process.");
                }

                if (sendData.Length % _worldSize != 0)
                {
                    throw new ArgumentException(
                        $"Data length {sendData.Length} must be divisible by world size {_worldSize}.");
                }

                int chunkSize = sendData.Length / _worldSize;
                _sharedBuffers[bufferId] = new List<Vector<T>>();

                for (int i = 0; i < _worldSize; i++)
                {
                    var chunk = new T[chunkSize];
                    Array.Copy(sendData.ToArray(), i * chunkSize, chunk, 0, chunkSize);
                    _sharedBuffers[bufferId].Add(new Vector<T>(chunk));
                }

                _pendingConsumers[bufferId] = _worldSize;
            }

            List<Vector<T>>? buffer;
            Vector<T> result;
            var startTime = DateTime.UtcNow;

            try
            {
                // Wait for root to split data
                while (!_sharedBuffers.ContainsKey(bufferId))
                {
                    Monitor.Wait(_globalLock, 10);
                    if ((DateTime.UtcNow - startTime).TotalMilliseconds > BarrierTimeoutMs)
                    {
                        throw new TimeoutException($"Scatter timeout after {BarrierTimeoutMs}ms waiting for root rank {root} to split data.");
                    }
                }

                // Retrieve scattered chunk
                if (_sharedBuffers.ContainsKey(bufferId))
                {
                    buffer = _sharedBuffers[bufferId];
                    if (buffer == null || buffer.Count <= _rank)
                    {
                        throw new InvalidOperationException($"Scatter buffer is null or missing data for rank {_rank} after synchronization.");
                    }
                    result = buffer[_rank].Clone();
                    return result;
                }
                else
                {
                    throw new InvalidOperationException("Scatter buffer was removed before data could be retrieved.");
                }
            }
            finally
            {
                // CRITICAL: Decrement consumer count even on exception to prevent buffer leaks
                // This must happen in finally to ensure cleanup even if Clone() or other operations fail
                if (_pendingConsumers.ContainsKey(bufferId))
                {
                    int remaining = --_pendingConsumers[bufferId];
                    if (remaining == 0)
                    {
                        _sharedBuffers.Remove(bufferId);
                        _pendingConsumers.Remove(bufferId);
                        _operationCounters[_environmentId]++;
                    }
                }

                // CRITICAL: PulseAll must happen even on timeout to wake other waiting processes
                Monitor.PulseAll(_globalLock);
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();

        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        // For single process, just return a copy
        if (_worldSize == 1)
        {
            if (operation == ReductionOperation.Average)
            {
                // Already averaged (only one value)
            }
            return data.Clone();
        }

        if (data.Length % _worldSize != 0)
        {
            throw new ArgumentException(
                $"Data length {data.Length} must be divisible by world size {_worldSize}.");
        }

        // Perform AllReduce then extract local chunk
        var reducedData = data.Clone();
        AllReduce(reducedData, operation);

        int chunkSize = reducedData.Length / _worldSize;
        var chunk = new T[chunkSize];
        Array.Copy(reducedData.ToArray(), _rank * chunkSize, chunk, 0, chunkSize);

        return new Vector<T>(chunk);
    }

    /// <inheritdoc/>
    public override void Send(Vector<T> data, int destinationRank, int tag = 0)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));
        ValidateRank(destinationRank, nameof(destinationRank));

        if (tag < 0)
        {
            throw new ArgumentException("Tag must be non-negative.", nameof(tag));
        }

        lock (_globalLock)
        {
            // Message queue key: {environmentId}_msg_{sourceRank}_{destRank}_{tag}
            string queueKey = $"{_environmentId}_msg_{_rank}_{destinationRank}_{tag}";

            if (!_messageQueues.ContainsKey(queueKey))
            {
                _messageQueues[queueKey] = new Queue<Vector<T>>();
            }

            // Enqueue a clone of the data
            _messageQueues[queueKey].Enqueue(data.Clone());

            // Wake up any waiting receivers
            Monitor.PulseAll(_globalLock);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Receive(int sourceRank, int count, int tag = 0)
    {
        EnsureInitialized();
        ValidateRank(sourceRank, nameof(sourceRank));

        if (count <= 0)
        {
            throw new ArgumentException("Count must be positive.", nameof(count));
        }

        if (tag < 0)
        {
            throw new ArgumentException("Tag must be non-negative.", nameof(tag));
        }

        lock (_globalLock)
        {
            // Message queue key: {environmentId}_msg_{sourceRank}_{destRank}_{tag}
            string queueKey = $"{_environmentId}_msg_{sourceRank}_{_rank}_{tag}";

            var startTime = DateTime.UtcNow;

            // Wait for a message to arrive
            while (!_messageQueues.ContainsKey(queueKey) || _messageQueues[queueKey].Count == 0)
            {
                Monitor.Wait(_globalLock, 10);

                if ((DateTime.UtcNow - startTime).TotalMilliseconds > MessageTimeoutMs)
                {
                    throw new TimeoutException(
                        $"Receive timeout after {MessageTimeoutMs}ms waiting for message from rank {sourceRank} with tag {tag}.");
                }
            }

            // Peek to validate size BEFORE dequeuing (prevents data loss on size mismatch)
            var message = _messageQueues[queueKey].Peek();

            // Validate size matches expected count
            if (message.Length != count)
            {
                throw new InvalidOperationException(
                    $"Received message size {message.Length} does not match expected count {count}. " +
                    "Message remains in queue for potential recovery.");
            }

            // Size validated, now safe to dequeue
            _messageQueues[queueKey].Dequeue();

            // Cleanup empty queues
            if (_messageQueues[queueKey].Count == 0)
            {
                _messageQueues.Remove(queueKey);
            }

            return message;
        }
    }

    /// <summary>
    /// Performs the actual reduction operation on a collection of vectors.
    /// </summary>
    private Vector<T> PerformReduction(List<Vector<T>> vectors, ReductionOperation operation)
    {
        if (vectors == null || vectors.Count == 0)
        {
            throw new ArgumentException("Cannot reduce empty vector list.");
        }

        int length = vectors[0].Length;

        // Validate all vectors have the same length to prevent IndexOutOfRangeException
        for (int i = 1; i < vectors.Count; i++)
        {
            if (vectors[i].Length != length)
            {
                throw new ArgumentException(
                    $"All vectors must have the same length. Vector 0 has length {length}, but vector {i} has length {vectors[i].Length}.");
            }
        }
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            T value = vectors[0][i];

            for (int j = 1; j < vectors.Count; j++)
            {
                value = ApplyReductionOperation(value, vectors[j][i], operation);
            }

            // Apply averaging if needed
            // Average operation: First accumulates using Sum (via ApplyReductionOperation which treats
            // Average same as Sum per CommunicationBackendBase.cs:296), then divides by the count of
            // vectors to compute the mean.
            // AVERAGE OPERATION CLARIFICATION:
            // This is mathematically correct: average = (v0 + v1 + ... + vn-1) / n
            // Implementation: First accumulate using Sum logic (lines above), then divide by count.
            // This is the standard and most efficient way to compute element-wise averages.
            // The division applies to the accumulated sum, not to each element before summing.
            if (operation == ReductionOperation.Average)
            {
                // CRITICAL: Ensure we use the proper numeric type conversion for division
                // vectors.Count is int, must convert to T to ensure type-safe division
                var count = NumOps.FromDouble(vectors.Count);
                value = NumOps.Divide(value, count);
            }

            result[i] = value;
        }

        return new Vector<T>(result);
    }
}
