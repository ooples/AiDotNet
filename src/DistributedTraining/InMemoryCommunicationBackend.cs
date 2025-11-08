using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;
using AiDotNet.Helpers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// A simple in-memory implementation of distributed communication for testing and single-machine scenarios.
/// This backend simulates multiple processes by using shared memory and locks.
///
/// For Beginners:
/// This is a "fake" distributed system that runs on a single machine.
/// It's perfect for testing your distributed code without needing multiple GPUs or machines.
/// Think of it as a practice mode - it simulates distributed behavior but everything
/// runs in one process.
///
/// Example:
/// <code>
/// // Create a simulated distributed environment with 4 "processes"
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// CommunicationManager.Initialize(backend);
/// </code>
///
/// Note: For production distributed training, you would use an MPI-based backend instead.
/// </summary>
/// <typeparam name="T">The numeric type for operations</typeparam>
public class InMemoryCommunicationBackend<T> : ICommunicationBackend<T> where T : struct
{
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly INumericOperations<T> _numOps;
    private bool _isInitialized;

    // Shared state for simulating collective operations
    // In a real implementation, this would be handled by the MPI backend
    private static readonly object _globalLock = new object();
    private static readonly Dictionary<string, List<Vector<T>>> _sharedBuffers = new();
    private static readonly Dictionary<string, int> _barrierCounters = new();
    private static int _barrierGeneration = 0;
    private static int _operationCounter = 0;

    /// <inheritdoc/>
    public int Rank => _rank;

    /// <inheritdoc/>
    public int WorldSize => _worldSize;

    /// <inheritdoc/>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Creates a new in-memory communication backend.
    ///
    /// For Beginners:
    /// You create one of these for each simulated "process". If you want to simulate
    /// 4 GPUs, you create 4 instances with ranks 0, 1, 2, 3, all with worldSize=4.
    /// </summary>
    /// <param name="rank">The rank (ID) of this simulated process (0-based)</param>
    /// <param name="worldSize">The total number of simulated processes</param>
    /// <exception cref="ArgumentException">Thrown if rank or worldSize are invalid</exception>
    public InMemoryCommunicationBackend(int rank, int worldSize)
    {
        if (rank < 0 || rank >= worldSize)
        {
            throw new ArgumentException(
                $"Invalid rank {rank}. Must be between 0 and {worldSize - 1}.");
        }

        if (worldSize <= 0)
        {
            throw new ArgumentException(
                $"Invalid worldSize {worldSize}. Must be positive.");
        }

        _rank = rank;
        _worldSize = worldSize;
        _numOps = MathHelper.GetNumericOperations<T>();
        _isInitialized = false;
    }

    /// <inheritdoc/>
    public void Initialize()
    {
        lock (_globalLock)
        {
            if (_isInitialized)
            {
                return;
            }

            _isInitialized = true;
        }
    }

    /// <inheritdoc/>
    public void Shutdown()
    {
        lock (_globalLock)
        {
            if (!_isInitialized)
            {
                return;
            }

            // Clear any remaining shared state
            _sharedBuffers.Clear();
            _barrierCounters.Clear();

            _isInitialized = false;
        }
    }

    /// <inheritdoc/>
    public void Barrier()
    {
        EnsureInitialized();

        lock (_globalLock)
        {
            // Use shared barrier generation counter so all ranks synchronize on same key
            string barrierId = $"barrier_{_barrierGeneration}";

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

            // Cleanup - rank 0 removes key and increments generation for next barrier
            if (_rank == 0)
            {
                _barrierCounters.Remove(barrierId);
                _barrierGeneration++;
            }
        }
    }

    /// <inheritdoc/>
    public void AllReduce(Vector<T> data, ReductionOperation operation)
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

        lock (_globalLock)
        {
            // Use shared operation counter so all ranks target same buffer key
            string bufferId = $"allreduce_{_operationCounter}";

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

            // Cleanup - rank 0 removes key and increments counter for next operation
            if (_rank == 0)
            {
                _sharedBuffers.Remove(bufferId);
                _operationCounter++;
            }
        }
    }

    /// <inheritdoc/>
    public Vector<T> AllGather(Vector<T> sendData)
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
            string bufferId = $"allgather_{_operationCounter}";

            // Initialize shared buffer
            if (!_sharedBuffers.ContainsKey(bufferId))
            {
                _sharedBuffers[bufferId] = new List<Vector<T>>(new Vector<T>[_worldSize]);
            }

            // Contribute local data
            _sharedBuffers[bufferId][_rank] = sendData.Clone();

            // Wait until all processes have contributed
            while (true)
            {
                int contributedCount = _sharedBuffers[bufferId].Count(v => v != null);
                if (contributedCount >= _worldSize)
                {
                    break;
                }
                Monitor.Wait(_globalLock, 10);
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

            // Cleanup - rank 0 removes key and increments counter for next operation
            if (_rank == 0)
            {
                _sharedBuffers.Remove(bufferId);
                _operationCounter++;
            }

            return new Vector<T>(result);
        }
    }

    /// <inheritdoc/>
    public Vector<T> Broadcast(Vector<T> data, int root = 0)
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

        lock (_globalLock)
        {
            // Use shared operation counter so all ranks target same buffer key
            string bufferId = $"broadcast_{_operationCounter}";
            Vector<T> result;
            List<Vector<T>> buffer;

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
            while (!_sharedBuffers.TryGetValue(bufferId, out buffer))
            {
                Monitor.Wait(_globalLock, 10);
            }

            Monitor.PulseAll(_globalLock);

            // All processes retrieve the data
            result = buffer[0].Clone();

            // Cleanup - rank 0 removes key and increments counter for next operation
            if (_rank == 0)
            {
                _sharedBuffers.Remove(bufferId);
                _operationCounter++;
            }

            return result;
        }
    }

    /// <inheritdoc/>
    public Vector<T> Scatter(Vector<T> sendData, int root = 0)
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
            string bufferId = $"scatter_{_operationCounter}";

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

            List<Vector<T>> buffer;

            // Wait for root to split data
            while (!_sharedBuffers.TryGetValue(bufferId, out buffer))
            {
                Monitor.Wait(_globalLock, 10);
            }

            Monitor.PulseAll(_globalLock);

            // Each process retrieves its chunk
            var result = buffer[_rank].Clone();

            // Cleanup - rank 0 removes key and increments counter for next operation
            if (_rank == 0)
            {
                _sharedBuffers.Remove(bufferId);
                _operationCounter++;
            }

            return result;
        }
    }

    /// <inheritdoc/>
    public Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation)
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
                value = ApplyOperation(value, vectors[j][i], operation);
            }

            // Apply averaging if needed
            if (operation == ReductionOperation.Average)
            {
                var count = _numOps.FromDouble(vectors.Count);
                value = _numOps.Divide(value, count);
            }

            result[i] = value;
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Applies the reduction operation to two values.
    /// </summary>
    private T ApplyOperation(T a, T b, ReductionOperation operation)
    {
        return operation switch
        {
            ReductionOperation.Sum or ReductionOperation.Average => _numOps.Add(a, b),
            ReductionOperation.Product => _numOps.Multiply(a, b),
            ReductionOperation.Min => _numOps.LessThan(a, b) ? a : b,
            ReductionOperation.Max => _numOps.GreaterThan(a, b) ? a : b,
            _ => throw new NotSupportedException($"Operation {operation} is not supported.")
        };
    }

    /// <summary>
    /// Ensures the backend is initialized before operations.
    /// </summary>
    private void EnsureInitialized()
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException(
                "Communication backend is not initialized. Call Initialize() first.");
        }
    }
}
