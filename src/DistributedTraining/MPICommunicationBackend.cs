using System;
using System.Linq;
using System.Reflection;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// MPI.NET-based communication backend for production distributed training.
/// </summary>
/// <remarks>
/// <para><b>Overview:</b>
/// MPI (Message Passing Interface) is the industry-standard communication framework for
/// high-performance computing. MPI.NET provides .NET bindings for MPI, enabling production-grade
/// distributed training on HPC clusters and supercomputers.
/// </para>
/// <para><b>Features:</b>
/// - Optimized collective operations (AllReduce, AllGather, etc.)
/// - Support for InfiniBand and other high-speed interconnects
/// - Battle-tested in HPC for decades
/// - Excellent performance and scalability
/// </para>
/// <para><b>Use Cases:</b>
/// - HPC cluster deployment
/// - Large-scale training (100s-1000s of nodes)
/// - InfiniBand or high-speed network infrastructure
/// - Production distributed training pipelines
/// </para>
/// <para><b>Requirements:</b>
/// - MPI.NET NuGet package
/// - MPI implementation (OpenMPI, MPICH, Intel MPI, etc.)
/// - MPI runtime environment
/// </para>
/// <para><b>Graceful Degradation:</b>
/// If MPI.NET is not available, this backend falls back to single-process mode
/// where all operations work correctly but without actual inter-process communication.
/// A warning is logged when fallback mode is active.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MPICommunicationBackend provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class MPICommunicationBackend<T> : CommunicationBackendBase<T>
{
    private int _rank;
    private int _worldSize;
    private bool _useMPI;
    private object? _mpiCommunicator;
    private Type? _mpiEnvironmentType;
    private Type? _mpiOperationType;

    /// <summary>
    /// Creates a new MPI communication backend.
    /// </summary>
    /// <param name="rank">This process's rank (will be obtained from MPI if available)</param>
    /// <param name="worldSize">Total number of processes (will be obtained from MPI if available)</param>
    public MPICommunicationBackend(int rank = 0, int worldSize = 1)
    {
        _rank = rank;
        _worldSize = worldSize;
        _useMPI = false;
    }

    /// <inheritdoc/>
    public override int Rank => _rank;

    /// <inheritdoc/>
    public override int WorldSize => _worldSize;

    /// <inheritdoc/>
    protected override void OnInitialize()
    {
        // Try to use MPI.NET if available via reflection
        try
        {
            // Attempt to load MPI.NET assembly
            _mpiEnvironmentType = Type.GetType("MPI.Environment, MPI");

            if (_mpiEnvironmentType != null)
            {
                // Check if MPI is already initialized
                var initializedProp = _mpiEnvironmentType.GetProperty("Initialized", BindingFlags.Public | BindingFlags.Static);
                if (initializedProp != null)
                {
                    var isInitialized = (bool?)initializedProp.GetValue(null);

                    if (isInitialized == true)
                    {
                        _useMPI = true;
                        Console.WriteLine("MPI.NET detected and initialized. Using MPI for communication.");

                        // Get communicator type and world communicator
                        var intracommunicatorType = Type.GetType("MPI.Intracommunicator, MPI");
                        if (intracommunicatorType != null)
                        {
                            var worldProp = intracommunicatorType.GetProperty("World", BindingFlags.Public | BindingFlags.Static);
                            _mpiCommunicator = worldProp?.GetValue(null);

                            // Query actual rank and world size from MPI communicator
                            if (_mpiCommunicator != null)
                            {
                                var rankProp = _mpiCommunicator.GetType().GetProperty("Rank");
                                var sizeProp = _mpiCommunicator.GetType().GetProperty("Size");

                                if (rankProp != null && sizeProp != null)
                                {
                                    _rank = (int)(rankProp.GetValue(_mpiCommunicator) ?? _rank);
                                    _worldSize = (int)(sizeProp.GetValue(_mpiCommunicator) ?? _worldSize);
                                    Console.WriteLine($"MPI Communicator: Rank={_rank}, WorldSize={_worldSize}");
                                }
                            }
                        }

                        return;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            // MPI.NET not available, will use fallback
            Console.WriteLine($"MPI.NET not available: {ex.Message}");
        }

        // Fallback to single-process mode
        _useMPI = false;
        Console.WriteLine("WARNING: MPI.NET not available. Running in single-process fallback mode.");
        Console.WriteLine("For production HPC deployment, install MPI.NET and an MPI implementation (OpenMPI, MPICH, etc.).");
    }

    /// <inheritdoc/>
    protected override void OnShutdown()
    {
        if (_useMPI && _mpiEnvironmentType != null)
        {
            try
            {
                // Check if already finalized
                var finalizedProp = _mpiEnvironmentType.GetProperty("Finalized", BindingFlags.Public | BindingFlags.Static);
                if (finalizedProp != null)
                {
                    var isFinalized = (bool?)finalizedProp.GetValue(null);

                    if (isFinalized == false)
                    {
                        // Call MPI.Environment.Finalize()
                        var finalizeMethod = _mpiEnvironmentType.GetMethod("Finalize", BindingFlags.Public | BindingFlags.Static);
                        finalizeMethod?.Invoke(null, null);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Error during MPI shutdown: {ex.Message}");
            }
        }

        _mpiCommunicator = null;
    }

    /// <inheritdoc/>
    public override void Barrier()
    {
        EnsureInitialized();

        if (!_useMPI)
        {
            // Single-process: barrier is a no-op
            return;
        }

        try
        {
            // Call _communicator.Barrier()
            var barrierMethod = _mpiCommunicator?.GetType().GetMethod("Barrier", Type.EmptyTypes);
            barrierMethod?.Invoke(_mpiCommunicator, null);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"MPI Barrier failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc/>
    public override void AllReduce(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));

        if (!_useMPI)
        {
            // Single-process: data already contains the result
            // For Average, divide by 1 (no-op)
            return;
        }

        try
        {
            // Get MPI operation
            var mpiOp = GetMPIOperation(operation);

            // Get the Allreduce method
            var allreduceMethod = _mpiCommunicator?.GetType().GetMethod("Allreduce", new[] { typeof(T[]), mpiOp.GetType() });

            if (allreduceMethod != null)
            {
                var array = data.ToArray();
                var result = (T[]?)allreduceMethod.Invoke(_mpiCommunicator, new object[] { array, mpiOp });

                if (result != null)
                {
                    // Handle average operation
                    if (operation == ReductionOperation.Average)
                    {
                        for (int i = 0; i < result.Length; i++)
                        {
                            result[i] = NumOps.Divide(result[i], NumOps.FromDouble(_worldSize));
                        }
                    }

                    // Copy result back to data
                    for (int i = 0; i < data.Length; i++)
                    {
                        data[i] = result[i];
                    }
                }
            }
            else
            {
                throw new InvalidOperationException("MPI Allreduce method not found.");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"MPI AllReduce failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> AllGather(Vector<T> sendData)
    {
        EnsureInitialized();
        ValidateData(sendData, nameof(sendData));

        if (!_useMPI)
        {
            // Single-process: return a copy
            return sendData.Clone();
        }

        try
        {
            // Get the Allgather method
            var allgatherMethod = _mpiCommunicator?.GetType().GetMethod("Allgather", new[] { typeof(T[]) });

            if (allgatherMethod != null)
            {
                var sendArray = sendData.ToArray();
                var result = (T[]?)allgatherMethod.Invoke(_mpiCommunicator, new object[] { sendArray });

                if (result != null)
                {
                    return new Vector<T>(result);
                }
            }

            throw new InvalidOperationException("MPI Allgather method not found or returned null.");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"MPI AllGather failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Broadcast(Vector<T> data, int root = 0)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));
        ValidateRoot(root);

        if (!_useMPI)
        {
            // Single-process: return a copy
            return data.Clone();
        }

        try
        {
            // Get the Broadcast method (ref T[] array, int root)
            var broadcastMethod = _mpiCommunicator?.GetType().GetMethod("Broadcast",
                new[] { typeof(T[]).MakeByRefType(), typeof(int) });

            if (broadcastMethod != null)
            {
                var array = data.ToArray();
                var parameters = new object[] { array, root };
                broadcastMethod.Invoke(_mpiCommunicator, parameters);

                // Get the modified array from parameters
                array = (T[])parameters[0];
                return new Vector<T>(array);
            }

            throw new InvalidOperationException("MPI Broadcast method not found.");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"MPI Broadcast failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Scatter(Vector<T> sendData, int root = 0)
    {
        EnsureInitialized();
        ValidateRoot(root);

        if (!_useMPI)
        {
            // Single-process: return a copy
            if (Rank == root)
            {
                ValidateData(sendData, nameof(sendData));
                return sendData.Clone();
            }
            return new Vector<T>(Array.Empty<T>());
        }

        try
        {
            T[]? sendArray = null;
            if (Rank == root)
            {
                ValidateData(sendData, nameof(sendData));
                sendArray = sendData.ToArray();
            }

            // Get the Scatter method
            var scatterMethod = _mpiCommunicator?.GetType().GetMethod("Scatter",
                new[] { typeof(T[]), typeof(int) });

            if (scatterMethod != null)
            {
                var result = (T[]?)scatterMethod.Invoke(_mpiCommunicator, new object[] { sendArray!, root });

                if (result != null)
                {
                    return new Vector<T>(result);
                }
            }

            throw new InvalidOperationException("MPI Scatter method not found or returned null.");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"MPI Scatter failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));

        if (!_useMPI)
        {
            // Single-process: return a copy
            return data.Clone();
        }

        if (data.Length % _worldSize != 0)
        {
            throw new ArgumentException(
                $"Data length {data.Length} must be divisible by world size {_worldSize}.");
        }

        try
        {
            var mpiOp = GetMPIOperation(operation);

            // Calculate counts for each process
            var counts = new int[_worldSize];
            int baseCount = data.Length / _worldSize;
            for (int i = 0; i < _worldSize; i++)
            {
                counts[i] = baseCount;
            }

            // Get the ReduceScatter method
            var reduceScatterMethod = _mpiCommunicator?.GetType().GetMethod("ReduceScatter",
                new[] { typeof(T[]), typeof(int[]), mpiOp.GetType() });

            if (reduceScatterMethod != null)
            {
                var array = data.ToArray();
                var result = (T[]?)reduceScatterMethod.Invoke(_mpiCommunicator,
                    new object[] { array, counts, mpiOp });

                if (result != null)
                {
                    // Handle average operation
                    if (operation == ReductionOperation.Average)
                    {
                        for (int i = 0; i < result.Length; i++)
                        {
                            result[i] = NumOps.Divide(result[i], NumOps.FromDouble(_worldSize));
                        }
                    }

                    return new Vector<T>(result);
                }
            }

            throw new InvalidOperationException("MPI ReduceScatter method not found or returned null.");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"MPI ReduceScatter failed: {ex.Message}", ex);
        }
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

        if (!_useMPI)
        {
            throw new InvalidOperationException("Cannot send in single-process mode.");
        }

        try
        {
            // Get the Send method: void Send<T>(T[] values, int dest, int tag)
            var sendMethod = _mpiCommunicator?.GetType().GetMethod("Send",
                new[] { typeof(T[]), typeof(int), typeof(int) });

            if (sendMethod != null)
            {
                var array = data.ToArray();
                sendMethod.Invoke(_mpiCommunicator, new object[] { array, destinationRank, tag });
            }
            else
            {
                throw new InvalidOperationException("MPI Send method not found.");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"MPI Send failed: {ex.Message}", ex);
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

        if (!_useMPI)
        {
            throw new InvalidOperationException("Cannot receive in single-process mode.");
        }

        try
        {
            // Get the Receive method: void Receive<T>(int source, int tag, ref T[] values)
            var receiveMethod = _mpiCommunicator?.GetType().GetMethod("Receive",
                new[] { typeof(int), typeof(int), typeof(T[]).MakeByRefType() });

            if (receiveMethod != null)
            {
                var array = new T[count];
                var parameters = new object[] { sourceRank, tag, array };
                receiveMethod.Invoke(_mpiCommunicator, parameters);

                // Get the received array from parameters
                array = (T[])parameters[2];
                return new Vector<T>(array);
            }

            throw new InvalidOperationException("MPI Receive method not found.");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"MPI Receive failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Gets the MPI operation object for the specified reduction operation.
    /// </summary>
    /// <param name="operation">The reduction operation</param>
    /// <returns>The MPI operation object</returns>
    /// <exception cref="NotSupportedException">Thrown if the operation is not supported</exception>
    private object GetMPIOperation(ReductionOperation operation)
    {
        if (_mpiOperationType == null)
        {
            // Get MPI.Operation<T> type
            var operationGenericType = Type.GetType("MPI.Operation`1, MPI");
            if (operationGenericType != null)
            {
                _mpiOperationType = operationGenericType.MakeGenericType(typeof(T));
            }
            else
            {
                throw new InvalidOperationException("MPI.Operation<T> type not found.");
            }
        }

        // Get the appropriate static operation field
        var fieldName = operation switch
        {
            ReductionOperation.Sum or ReductionOperation.Average => "Add",
            ReductionOperation.Product => "Multiply",
            ReductionOperation.Min => "Min",
            ReductionOperation.Max => "Max",
            _ => throw new NotSupportedException($"Operation {operation} is not supported.")
        };

        var field = _mpiOperationType.GetField(fieldName, BindingFlags.Public | BindingFlags.Static);
        if (field == null)
        {
            throw new InvalidOperationException($"MPI operation field '{fieldName}' not found.");
        }

        var mpiOp = field.GetValue(null);
        if (mpiOp == null)
        {
            throw new InvalidOperationException($"MPI operation '{fieldName}' is null.");
        }

        return mpiOp;
    }
}
