using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Defines the contract for distributed communication backends.
/// </summary>
/// <remarks>
/// <para>
/// This abstraction allows different implementations (in-memory, MPI.NET, NCCL, etc.)
/// to provide collective communication operations for distributed training.
/// </para>
/// <para><b>For Beginners:</b>
/// This interface defines how different processes (or GPUs) communicate with each other
/// during distributed training. Think of it as a "walkie-talkie" system where multiple
/// processes can send data to each other, synchronize, and perform collective operations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations (float, double, etc.)</typeparam>
[AiDotNet.Configuration.YamlConfigurable("CommunicationBackend")]
public interface ICommunicationBackend<T>
{
    /// <summary>
    /// Gets the rank (ID) of the current process in the distributed group.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Rank 0 is typically the "master" or "coordinator" process.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Think of rank as your process's unique ID number. If you have 4 GPUs,
    /// ranks will be 0, 1, 2, and 3. Rank 0 is usually the "boss" that coordinates everything.
    /// </para>
    /// </remarks>
    int Rank { get; }

    /// <summary>
    /// Gets the total number of processes in the distributed group.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is how many processes (or GPUs) are working together.
    /// If WorldSize is 4, you have 4 processes sharing the work.
    /// </para>
    /// </remarks>
    int WorldSize { get; }

    /// <summary>
    /// Gets whether this backend is initialized and ready for use.
    /// </summary>
    bool IsInitialized { get; }

    /// <summary>
    /// Initializes the communication backend.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Must be called before any other operations.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like turning on your walkie-talkie system. You need to do this
    /// once at the start before any processes can talk to each other.
    /// </para>
    /// </remarks>
    void Initialize();

    /// <summary>
    /// Shuts down the communication backend and releases resources.
    /// Should be called when distributed training is complete.
    /// </summary>
    void Shutdown();

    /// <summary>
    /// Synchronization barrier - blocks until all processes reach this point.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is like a meeting checkpoint. All processes must arrive at this point
    /// before any of them can continue. It ensures everyone is synchronized.
    /// Example: Before starting training, you want all GPUs to be ready.
    /// </para>
    /// </remarks>
    void Barrier();

    /// <summary>
    /// AllReduce operation - combines data from all processes using the specified operation
    /// and distributes the result back to all processes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Imagine 4 GPUs each calculated a gradient vector. AllReduce takes all 4 vectors,
    /// adds them together (if operation is Sum), and gives the result to all 4 GPUs.
    /// This is crucial for averaging gradients across GPUs during training.
    /// </para>
    /// <para>
    /// Common operations:
    /// - Sum: Add all values together (used for gradient averaging)
    /// - Max: Take the maximum value across all processes
    /// - Min: Take the minimum value across all processes
    /// </para>
    /// </remarks>
    /// <param name="data">The data to reduce. Will be replaced with the reduced result.</param>
    /// <param name="operation">The reduction operation (Sum, Max, Min, etc.)</param>
    void AllReduce(Vector<T> data, ReductionOperation operation);

    /// <summary>
    /// AllGather operation - gathers data from all processes and concatenates it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each process receives the complete concatenated result.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// If GPU 0 has [1,2], GPU 1 has [3,4], GPU 2 has [5,6], GPU 3 has [7,8],
    /// then AllGather gives everyone [1,2,3,4,5,6,7,8].
    /// This is used to reconstruct the full model parameters from sharded pieces.
    /// </para>
    /// </remarks>
    /// <param name="sendData">The local data to contribute</param>
    /// <returns>The gathered data from all processes concatenated together</returns>
    Vector<T> AllGather(Vector<T> sendData);

    /// <summary>
    /// Broadcast operation - sends data from one process (root) to all other processes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is like an announcement from the boss (root process). The root sends
    /// data to everyone else. Useful for distributing initial parameters or configurations.
    /// </para>
    /// </remarks>
    /// <param name="data">The data to broadcast (only meaningful on root process)</param>
    /// <param name="root">The rank of the process that is broadcasting</param>
    /// <returns>The broadcast data (received from root on non-root processes)</returns>
    Vector<T> Broadcast(Vector<T> data, int root = 0);

    /// <summary>
    /// Scatter operation - distributes different chunks of data from root to each process.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// The root has a big array and wants to give each GPU a different piece.
    /// If root has [1,2,3,4,5,6,7,8] and WorldSize=4, it gives:
    /// GPU 0 gets [1,2], GPU 1 gets [3,4], GPU 2 gets [5,6], GPU 3 gets [7,8]
    /// </para>
    /// </remarks>
    /// <param name="sendData">The data to scatter (only used on root process)</param>
    /// <param name="root">The rank of the process that is scattering</param>
    /// <returns>The chunk of data received by this process</returns>
    Vector<T> Scatter(Vector<T> sendData, int root = 0);

    /// <summary>
    /// ReduceScatter operation - reduces data and scatters the result.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Combines AllReduce and Scatter in one operation for efficiency.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is an optimization that combines reduction and scattering.
    /// Instead of doing AllReduce (everyone gets everything) then Scatter (split it up),
    /// we directly compute and distribute only the needed chunks.
    /// </para>
    /// </remarks>
    /// <param name="data">The data to reduce and scatter</param>
    /// <param name="operation">The reduction operation</param>
    /// <returns>The reduced chunk for this process</returns>
    Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation);

    /// <summary>
    /// Send operation - sends data from this process to a specific destination process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a point-to-point communication operation. Unlike collective operations
    /// (AllReduce, Broadcast, etc.), only two processes are involved: sender and receiver.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like sending a private message to one specific GPU. Unlike Broadcast
    /// (which sends to everyone), Send only sends to one receiver.
    ///
    /// Use cases:
    /// - Pipeline parallelism: sending activations from one stage to the next
    /// - Ring-based algorithms: sending data to neighbor in a ring
    /// - Custom communication patterns
    /// </para>
    /// <para><b>Important:</b>
    /// Send must be matched with a corresponding Receive on the destination process.
    /// The sender and receiver must agree on the message size, otherwise deadlock
    /// or incorrect data transfer can occur.
    /// </para>
    /// </remarks>
    /// <param name="data">The data to send</param>
    /// <param name="destinationRank">The rank of the process to send to</param>
    /// <param name="tag">Optional message tag to distinguish different messages (default=0)</param>
    void Send(Vector<T> data, int destinationRank, int tag = 0);

    /// <summary>
    /// Receive operation - receives data from a specific source process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a point-to-point communication operation that blocks until data arrives.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like waiting for a private message from a specific GPU. The process
    /// will wait (block) until the message arrives.
    ///
    /// Use cases:
    /// - Pipeline parallelism: receiving activations from previous stage
    /// - Ring-based algorithms: receiving data from neighbor
    /// - Custom communication patterns
    /// </para>
    /// <para><b>Important:</b>
    /// Receive must be matched with a corresponding Send from the source process.
    /// If the sender never sends, this will deadlock (hang forever). If the sizes
    /// don't match, data corruption or errors can occur.
    /// </para>
    /// </remarks>
    /// <param name="sourceRank">The rank of the process to receive from</param>
    /// <param name="count">The expected number of elements to receive</param>
    /// <param name="tag">Optional message tag to match with Send (default=0)</param>
    /// <returns>The received data</returns>
    Vector<T> Receive(int sourceRank, int count, int tag = 0);
}

/// <summary>
/// Defines the supported reduction operations for collective communication.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// These are different ways to combine values from multiple processes.
/// </para>
/// </remarks>
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
