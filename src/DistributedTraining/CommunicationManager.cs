using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Central manager for distributed communication operations.
/// </summary>
/// <remarks>
/// <para>
/// <b>⚠️ WARNING - Static Mutable State:</b> This class uses static mutable state for managing
/// communication backends. This design choice has important implications for concurrent usage:
/// - Only ONE backend can be active per process
/// - Unit tests using this class CANNOT run in parallel
/// - Multiple training sessions in the same process share the same backend
/// See detailed thread-safety notes below for proper usage patterns.
/// </para>
/// <para>
/// Provides a static API for collective communication in distributed training scenarios.
/// </para>
/// <para><b>For Beginners:</b>
/// This is your main entry point for distributed training communication.
/// It's a "wrapper" that makes it easy to communicate between different processes/GPUs
/// without worrying about the underlying implementation details.
/// </para>
/// <para>
/// Example usage:
/// <code>
/// // Initialize communication (do this once at startup)
/// CommunicationManager.Initialize(new InMemoryCommunicationBackend&lt;double&gt;());
///
/// // Get your process ID and total number of processes
/// int myRank = CommunicationManager.GetRank();
/// int totalProcesses = CommunicationManager.GetWorldSize();
///
/// // Average gradients across all processes
/// Vector&lt;double&gt; gradients = ...; // Your local gradients
/// CommunicationManager.AllReduce(gradients, ReductionOperation.Sum);
/// // Now 'gradients' contains the sum from all processes
///
/// // Clean up when done
/// CommunicationManager.Shutdown();
/// </code>
///
/// IMPORTANT - Thread Safety and Testing Limitations:
/// This class uses STATIC MUTABLE STATE which has the following implications:
///
/// 1. SINGLE GLOBAL INSTANCE: Only ONE backend can be active per process at a time.
///    Multiple training sessions in the same process will share the same backend instance.
///
/// 2. PARALLEL TEST EXECUTION: Tests that use this class CANNOT run in parallel.
///    Use [Collection] attributes in xUnit or similar mechanisms to enforce sequential execution.
///
/// 3. TEST ISOLATION: Always call Shutdown() in test cleanup to reset state.
///    For better isolation in tests, use InMemoryCommunicationBackend with unique environment IDs
///    and inject the backend directly instead of using this static manager.
///
/// 4. CONCURRENT INITIALIZATION: Attempting to Initialize() from multiple threads concurrently
///    is protected by locks, but may result in exceptions if already initialized.
///
/// Recommended Test Pattern:
/// <code>
/// // Option 1: Use environment isolation (recommended for parallel tests)
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4, environmentId: "test-123");
/// // Use backend directly, don't call CommunicationManager.Initialize()
///
/// // Option 2: Sequential tests with proper cleanup
/// [Collection("DistributedTraining")] // Force sequential execution
/// public class MyDistributedTests
/// {
///     [Fact]
///     public void MyTest()
///     {
///         try
///         {
///             CommunicationManager.Initialize(...);
///             // Test code
///         }
///         finally
///         {
///             CommunicationManager.Shutdown(); // CRITICAL: Always cleanup
///         }
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public static class CommunicationManager
{
    private static readonly object _lock = new object();
    private static ICommunicationBackend<float>? _floatBackend;
    private static ICommunicationBackend<double>? _doubleBackend;
    private static bool _isInitialized = false;

    /// <summary>
    /// Gets whether the communication manager has been initialized.
    /// </summary>
    public static bool IsInitialized
    {
        get
        {
            lock (_lock)
            {
                return _isInitialized;
            }
        }
    }

    /// <summary>
    /// Initializes the communication manager with the specified backend.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This must be called before any other operations.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This sets up the communication system. You need to provide a "backend"
    /// which is the actual implementation that does the communication.
    /// For testing, use InMemoryCommunicationBackend. For real distributed training,
    /// you would use an MPI backend.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type (float or double)</typeparam>
    /// <param name="backend">The communication backend to use</param>
    /// <exception cref="ArgumentNullException">Thrown if backend is null</exception>
    /// <exception cref="InvalidOperationException">Thrown if already initialized</exception>
    public static void Initialize<T>(ICommunicationBackend<T> backend)
    {
        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        lock (_lock)
        {
            if (_isInitialized)
            {
                throw new InvalidOperationException(
                    "CommunicationManager is already initialized. Call Shutdown() first, or check IsInitialized before calling Initialize().");
            }

            // Initialize the backend
            backend.Initialize();

            // Store the backend (type-specific)
            if (typeof(T) == typeof(float))
            {
                _floatBackend = backend as ICommunicationBackend<float>;
            }
            else if (typeof(T) == typeof(double))
            {
                _doubleBackend = backend as ICommunicationBackend<double>;
            }
            else
            {
                throw new NotSupportedException(
                    $"Type {typeof(T).Name} is not supported for distributed communication. " +
                    "Only float and double are supported because of MPI type mapping constraints. " +
                    "Please use float or double for distributed operations.");
            }

            _isInitialized = true;
        }
    }

    /// <summary>
    /// Shuts down the communication manager and releases all resources.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Should be called when distributed training is complete.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is cleanup - call it when you're done with distributed training
    /// to free up resources and properly close connections.
    /// </para>
    /// </remarks>
    public static void Shutdown()
    {
        lock (_lock)
        {
            if (!_isInitialized)
            {
                return;
            }

            _floatBackend?.Shutdown();
            _doubleBackend?.Shutdown();

            _floatBackend = null;
            _doubleBackend = null;
            _isInitialized = false;
        }
    }

    /// <summary>
    /// Gets the rank (ID) of the current process.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This tells you which process you are. If you're running on 4 GPUs,
    /// one will be rank 0, one will be rank 1, etc.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <returns>The rank of the current process (0-based index)</returns>
    /// <exception cref="InvalidOperationException">Thrown if not initialized</exception>
    public static int GetRank<T>()
    {
        var backend = GetBackend<T>();
        return backend.Rank;
    }

    /// <summary>
    /// Gets the total number of processes in the distributed group.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This tells you how many processes (or GPUs) are working together total.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <returns>The total number of processes</returns>
    /// <exception cref="InvalidOperationException">Thrown if not initialized</exception>
    public static int GetWorldSize<T>()
    {
        var backend = GetBackend<T>();
        return backend.WorldSize;
    }

    /// <summary>
    /// Blocks until all processes reach this synchronization point.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is a "wait for everyone" checkpoint. All processes must reach
    /// this point before any can continue. Useful for making sure everyone
    /// is ready before starting the next step.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <exception cref="InvalidOperationException">Thrown if not initialized</exception>
    public static void Barrier<T>()
    {
        var backend = GetBackend<T>();
        backend.Barrier();
    }

    /// <summary>
    /// Performs an AllReduce operation - combines data from all processes and
    /// distributes the result to all processes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is the key operation for distributed training. It combines values
    /// from all processes (like adding gradients from all GPUs) and gives
    /// everyone the result. After this, everyone has the same combined data.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="data">The data to reduce (will be modified to contain the result)</param>
    /// <param name="operation">How to combine the data (Sum, Average, Max, etc.)</param>
    /// <exception cref="InvalidOperationException">Thrown if not initialized</exception>
    /// <exception cref="ArgumentNullException">Thrown if data is null</exception>
    public static void AllReduce<T>(Vector<T> data, ReductionOperation operation)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var backend = GetBackend<T>();
        backend.AllReduce(data, operation);
    }

    /// <summary>
    /// Gathers data from all processes and returns the concatenated result.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This collects data from all processes and combines it into one big array.
    /// Everyone gets the full combined result. Useful when you need to see
    /// all the pieces together (like reconstructing full parameters from shards).
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="sendData">The local data to contribute</param>
    /// <returns>The concatenated data from all processes</returns>
    /// <exception cref="InvalidOperationException">Thrown if not initialized</exception>
    /// <exception cref="ArgumentNullException">Thrown if sendData is null</exception>
    public static Vector<T> AllGather<T>(Vector<T> sendData)
    {
        if (sendData == null)
        {
            throw new ArgumentNullException(nameof(sendData));
        }

        var backend = GetBackend<T>();
        return backend.AllGather(sendData);
    }

    /// <summary>
    /// Broadcasts data from one process (root) to all other processes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// One process (the root) sends data to everyone else. Everyone ends up
    /// with the same data. Useful for distributing initial parameters or settings.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="data">The data to broadcast (only meaningful on root)</param>
    /// <param name="root">Which process is broadcasting (default: 0)</param>
    /// <returns>The broadcast data</returns>
    /// <exception cref="InvalidOperationException">Thrown if not initialized</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if root is out of valid range</exception>
    public static Vector<T> Broadcast<T>(Vector<T> data, int root = 0)
    {
        var backend = GetBackend<T>();

        if (root < 0 || root >= backend.WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(root),
                $"Root must be between 0 and {backend.WorldSize - 1}, but was {root}.");
        }

        return backend.Broadcast(data, root);
    }

    /// <summary>
    /// Scatters different chunks of data from root to each process.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// The root process splits data into chunks and gives each process
    /// a different chunk. This is how we distribute work across processes.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="sendData">The data to scatter (only used on root)</param>
    /// <param name="root">Which process is scattering (default: 0)</param>
    /// <returns>The chunk received by this process</returns>
    /// <exception cref="InvalidOperationException">Thrown if not initialized</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if root is out of valid range</exception>
    public static Vector<T> Scatter<T>(Vector<T> sendData, int root = 0)
    {
        var backend = GetBackend<T>();

        if (root < 0 || root >= backend.WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(root),
                $"Root must be between 0 and {backend.WorldSize - 1}, but was {root}.");
        }

        return backend.Scatter(sendData, root);
    }

    /// <summary>
    /// Performs a reduce-scatter operation - combines data and distributes chunks.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is a combined operation that's more efficient than doing
    /// AllReduce followed by Scatter. It reduces the data and immediately
    /// gives each process only their chunk of the result.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="data">The data to reduce and scatter</param>
    /// <param name="operation">How to combine the data</param>
    /// <returns>The reduced chunk for this process</returns>
    /// <exception cref="InvalidOperationException">Thrown if not initialized</exception>
    /// <exception cref="ArgumentNullException">Thrown if data is null</exception>
    public static Vector<T> ReduceScatter<T>(Vector<T> data, ReductionOperation operation)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var backend = GetBackend<T>();
        return backend.ReduceScatter(data, operation);
    }

    /// <summary>
    /// Gets the appropriate backend for the specified type.
    /// </summary>
    private static ICommunicationBackend<T> GetBackend<T>()
    {
        lock (_lock)
        {
            if (!_isInitialized)
            {
                throw new InvalidOperationException(
                    "CommunicationManager has not been initialized. Call Initialize() with a communication backend first, or check IsInitialized before using distributed operations.");
            }

            if (typeof(T) == typeof(float))
            {
                if (_floatBackend == null)
                {
                    throw new InvalidOperationException(
                        "CommunicationManager was not initialized with a float backend.");
                }
                return (ICommunicationBackend<T>)(object)_floatBackend;
            }
            else if (typeof(T) == typeof(double))
            {
                if (_doubleBackend == null)
                {
                    throw new InvalidOperationException(
                        "CommunicationManager was not initialized with a double backend.");
                }
                return (ICommunicationBackend<T>)(object)_doubleBackend;
            }
            else
            {
                throw new NotSupportedException(
                    $"Type {typeof(T).Name} is not supported. Use float or double.");
            }
        }
    }
}
