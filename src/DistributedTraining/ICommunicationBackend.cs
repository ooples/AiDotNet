using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Defines the contract for distributed communication backends.
/// This abstraction allows different implementations (in-memory, MPI.NET, NCCL, etc.)
/// to provide collective communication operations for distributed training.
///
/// For Beginners:
/// This interface defines how different processes (or GPUs) communicate with each other
/// during distributed training. Think of it as a "walkie-talkie" system where multiple
/// processes can send data to each other, synchronize, and perform collective operations.
/// </summary>
/// <typeparam name="T">The numeric type for operations (float, double, etc.)</typeparam>
public interface ICommunicationBackend<T> where T : struct
{
    /// <summary>
    /// Gets the rank (ID) of the current process in the distributed group.
    /// Rank 0 is typically the "master" or "coordinator" process.
    ///
    /// For Beginners:
    /// Think of rank as your process's unique ID number. If you have 4 GPUs,
    /// ranks will be 0, 1, 2, and 3. Rank 0 is usually the "boss" that coordinates everything.
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Gets the total number of processes in the distributed group.
    ///
    /// For Beginners:
    /// This is how many processes (or GPUs) are working together.
    /// If WorldSize is 4, you have 4 processes sharing the work.
    /// </summary>
    int WorldSize { get; }

    /// <summary>
    /// Gets whether this backend is initialized and ready for use.
    /// </summary>
    bool IsInitialized { get; }

    /// <summary>
    /// Initializes the communication backend.
    /// Must be called before any other operations.
    ///
    /// For Beginners:
    /// This is like turning on your walkie-talkie system. You need to do this
    /// once at the start before any processes can talk to each other.
    /// </summary>
    void Initialize();

    /// <summary>
    /// Shuts down the communication backend and releases resources.
    /// Should be called when distributed training is complete.
    /// </summary>
    void Shutdown();

    /// <summary>
    /// Synchronization barrier - blocks until all processes reach this point.
    ///
    /// For Beginners:
    /// This is like a meeting checkpoint. All processes must arrive at this point
    /// before any of them can continue. It ensures everyone is synchronized.
    /// Example: Before starting training, you want all GPUs to be ready.
    /// </summary>
    void Barrier();

    /// <summary>
    /// AllReduce operation - combines data from all processes using the specified operation
    /// and distributes the result back to all processes.
    ///
    /// For Beginners:
    /// Imagine 4 GPUs each calculated a gradient vector. AllReduce takes all 4 vectors,
    /// adds them together (if operation is Sum), and gives the result to all 4 GPUs.
    /// This is crucial for averaging gradients across GPUs during training.
    ///
    /// Common operations:
    /// - Sum: Add all values together (used for gradient averaging)
    /// - Max: Take the maximum value across all processes
    /// - Min: Take the minimum value across all processes
    /// </summary>
    /// <param name="data">The data to reduce. Will be replaced with the reduced result.</param>
    /// <param name="operation">The reduction operation (Sum, Max, Min, etc.)</param>
    void AllReduce(Vector<T> data, ReductionOperation operation);

    /// <summary>
    /// AllGather operation - gathers data from all processes and concatenates it.
    /// Each process receives the complete concatenated result.
    ///
    /// For Beginners:
    /// If GPU 0 has [1,2], GPU 1 has [3,4], GPU 2 has [5,6], GPU 3 has [7,8],
    /// then AllGather gives everyone [1,2,3,4,5,6,7,8].
    /// This is used to reconstruct the full model parameters from sharded pieces.
    /// </summary>
    /// <param name="sendData">The local data to contribute</param>
    /// <returns>The gathered data from all processes concatenated together</returns>
    Vector<T> AllGather(Vector<T> sendData);

    /// <summary>
    /// Broadcast operation - sends data from one process (root) to all other processes.
    ///
    /// For Beginners:
    /// This is like an announcement from the boss (root process). The root sends
    /// data to everyone else. Useful for distributing initial parameters or configurations.
    /// </summary>
    /// <param name="data">The data to broadcast (only meaningful on root process)</param>
    /// <param name="root">The rank of the process that is broadcasting</param>
    /// <returns>The broadcast data (received from root on non-root processes)</returns>
    Vector<T> Broadcast(Vector<T> data, int root = 0);

    /// <summary>
    /// Scatter operation - distributes different chunks of data from root to each process.
    ///
    /// For Beginners:
    /// The root has a big array and wants to give each GPU a different piece.
    /// If root has [1,2,3,4,5,6,7,8] and WorldSize=4, it gives:
    /// GPU 0 gets [1,2], GPU 1 gets [3,4], GPU 2 gets [5,6], GPU 3 gets [7,8]
    /// </summary>
    /// <param name="sendData">The data to scatter (only used on root process)</param>
    /// <param name="root">The rank of the process that is scattering</param>
    /// <returns>The chunk of data received by this process</returns>
    Vector<T> Scatter(Vector<T> sendData, int root = 0);

    /// <summary>
    /// ReduceScatter operation - reduces data and scatters the result.
    /// Combines AllReduce and Scatter in one operation for efficiency.
    ///
    /// For Beginners:
    /// This is an optimization that combines reduction and scattering.
    /// Instead of doing AllReduce (everyone gets everything) then Scatter (split it up),
    /// we directly compute and distribute only the needed chunks.
    /// </summary>
    /// <param name="data">The data to reduce and scatter</param>
    /// <param name="operation">The reduction operation</param>
    /// <returns>The reduced chunk for this process</returns>
    Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation);
}

/// <summary>
/// Defines the supported reduction operations for collective communication.
///
/// For Beginners:
/// These are different ways to combine values from multiple processes.
/// </summary>
public enum ReductionOperation
{
    /// <summary>Add all values together</summary>
    Sum,

    /// <summary>Multiply all values together</summary>
    Product,

    /// <summary>Take the minimum value</summary>
    Min,

    /// <summary>Take the maximum value</summary>
    Max,

    /// <summary>Compute average (sum divided by count)</summary>
    Average
}
