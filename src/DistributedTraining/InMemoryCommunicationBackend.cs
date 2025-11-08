using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides an in-memory implementation of distributed communication for testing and single-machine scenarios.
/// </summary>
/// <remarks>
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

    // Shared state for simulating collective operations
    // In a real implementation, this would be handled by the MPI backend
    private static readonly object _globalLock = new object();
    private static readonly Dictionary<string, List<Vector<T>>> _sharedBuffers = new();
    private static readonly Dictionary<string, int> _barrierCounters = new();

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
    /// <exception cref="ArgumentException">Thrown if rank or worldSize are invalid</exception>
    public InMemoryCommunicationBackend(int rank, int worldSize)
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

        _rank = rank;
        _worldSize = worldSize;
    }

    /// <inheritdoc/>
    protected override void OnShutdown()
    {
        lock (_globalLock)
        {
            // Clear any remaining shared state
            _sharedBuffers.Clear();
            _barrierCounters.Clear();
        }
    }

    /// <inheritdoc/>
    public override void Barrier()
    {
        EnsureInitialized();

        lock (_globalLock)
        {
            string barrierId = $"barrier_{DateTime.UtcNow.Ticks}";

            if (!_barrierCounters.ContainsKey(barrierId))
            {
                _barrierCounters[barrierId] = 0;
            }

            _barrierCounters[barrierId]++;

            // Wait until all processes have reached the barrier
            while (_barrierCounters[barrierId] < _worldSize)
            {
                Monitor.Wait(_globalLock, 10);
            }

            Monitor.PulseAll(_globalLock);

            // Cleanup
            if (_rank == 0)
            {
                _barrierCounters.Remove(barrierId);
            }
        }
    }

    /// <inheritdoc/>
    public override void AllReduce(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();

        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        // For single process, no communication needed
        if (_worldSize == 1)
        {
            if (operation == ReductionOperation.Average)
            {
                // Already averaged (only one value)
            }
            return;
        }

        string bufferId = $"allreduce_{Guid.NewGuid()}";

        lock (_globalLock)
        {
            // Initialize shared buffer
            if (!_sharedBuffers.ContainsKey(bufferId))
            {
                _sharedBuffers[bufferId] = new List<Vector<T>>();
            }

            // Contribute local data
            _sharedBuffers[bufferId].Add(data.Clone());

            // Wait until all processes have contributed
            while (_sharedBuffers[bufferId].Count < _worldSize)
            {
                Monitor.Wait(_globalLock, 10);
            }

            Monitor.PulseAll(_globalLock);

            // Perform reduction
            var allData = _sharedBuffers[bufferId];
            var result = PerformReduction(allData, operation);

            // Copy result back to input data
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = result[i];
            }

            // Cleanup (rank 0 cleans up)
            if (_rank == 0)
            {
                _sharedBuffers.Remove(bufferId);
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

        string bufferId = $"allgather_{Guid.NewGuid()}";

        lock (_globalLock)
        {
            // Initialize shared buffer
            if (!_sharedBuffers.ContainsKey(bufferId))
            {
                _sharedBuffers[bufferId] = new List<Vector<T>>(new Vector<T>[_worldSize]);
            }

            // Contribute local data
            _sharedBuffers[bufferId][_rank] = sendData.Clone();

            // Wait until all processes have contributed
            int contributedCount = 0;
            while (contributedCount < _worldSize)
            {
                contributedCount = _sharedBuffers[bufferId].Count(v => v != null);
                if (contributedCount < _worldSize)
                {
                    Monitor.Wait(_globalLock, 10);
                }
            }

            Monitor.PulseAll(_globalLock);

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

            // Cleanup
            if (_rank == 0)
            {
                _sharedBuffers.Remove(bufferId);
            }

            return new Vector<T>(result);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Broadcast(Vector<T> data, int root = 0)
    {
        EnsureInitialized();

        if (root < 0 || root >= _worldSize)
        {
            throw new ArgumentException($"Invalid root {root}. Must be between 0 and {_worldSize - 1}.");
        }

        // For single process, just return a copy
        if (_worldSize == 1)
        {
            return data?.Clone() ?? throw new ArgumentNullException(nameof(data));
        }

        string bufferId = $"broadcast_{Guid.NewGuid()}";

        lock (_globalLock)
        {
            Vector<T> result;

            // Root process stores the data
            if (_rank == root)
            {
                if (data == null)
                {
                    throw new ArgumentNullException(nameof(data), "Data cannot be null on root process.");
                }
                _sharedBuffers[bufferId] = new List<Vector<T>> { data.Clone() };
            }

            // Wait for root to store data
            while (!_sharedBuffers.ContainsKey(bufferId))
            {
                Monitor.Wait(_globalLock, 10);
            }

            Monitor.PulseAll(_globalLock);

            // All processes retrieve the data
            result = _sharedBuffers[bufferId][0].Clone();

            // Cleanup
            if (_rank == 0)
            {
                _sharedBuffers.Remove(bufferId);
            }

            return result;
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

        string bufferId = $"scatter_{Guid.NewGuid()}";

        lock (_globalLock)
        {
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
            }

            // Wait for root to split data
            while (!_sharedBuffers.ContainsKey(bufferId))
            {
                Monitor.Wait(_globalLock, 10);
            }

            Monitor.PulseAll(_globalLock);

            // Each process retrieves its chunk
            var result = _sharedBuffers[bufferId][_rank].Clone();

            // Cleanup
            if (_rank == 0)
            {
                _sharedBuffers.Remove(bufferId);
            }

            return result;
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
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            T value = vectors[0][i];

            for (int j = 1; j < vectors.Count; j++)
            {
                value = ApplyReductionOperation(value, vectors[j][i], operation);
            }

            // Apply averaging if needed
            if (operation == ReductionOperation.Average)
            {
                var count = NumOps.FromDouble(vectors.Count);
                value = NumOps.Divide(value, count);
            }

            result[i] = value;
        }

        return new Vector<T>(result);
    }
}
