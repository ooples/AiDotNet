namespace AiDotNet.DistributedTraining;

/// <summary>
/// Configuration for parameter sharding in distributed training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// This configuration tells the sharding system how to divide up parameters
/// and how to handle communication. Think of it as the "rules" for how the
/// team collaborates.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
public interface IShardingConfiguration<T>
{
    /// <summary>
    /// Gets whether to automatically synchronize gradients after backward pass.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// When true, gradients are automatically shared across all processes after
    /// each training step. This is usually what you want for standard training.
    /// You might set it to false if you want manual control over synchronization.
    /// </para>
    /// <para>Default: true</para>
    /// </remarks>
    bool AutoSyncGradients { get; }

    /// <summary>
    /// Gets the communication backend to use for distributed operations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is the "communication system" that processes use to talk to each other.
    /// It could be an in-memory backend for testing or an MPI backend for real
    /// distributed training across multiple machines.
    /// </para>
    /// </remarks>
    ICommunicationBackend<T> CommunicationBackend { get; }

    /// <summary>
    /// Gets the minimum parameter group size for sharding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Parameters smaller than this might be grouped together to reduce communication overhead.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Sending many tiny messages is inefficient. This setting groups small
    /// parameters together into larger chunks before communicating them.
    /// Think of it like sending one big box instead of 100 tiny envelopes.
    /// </para>
    /// <para>Default: 1024</para>
    /// </remarks>
    int MinimumParameterGroupSize { get; }

    /// <summary>
    /// Gets whether to enable gradient compression to reduce communication costs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Gradient compression reduces the size of data that needs to be sent
    /// between processes. It's like zipping a file before sending it - faster
    /// to send, but requires a tiny bit of extra work to compress/decompress.
    /// This can significantly speed up training on slower networks.
    /// </para>
    /// <para>Default: false</para>
    /// </remarks>
    bool EnableGradientCompression { get; }

    /// <summary>
    /// Gets the learning rate for gradient application during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// The learning rate controls how much to update model parameters based on
    /// computed gradients. A typical default is 0.01. Lower values mean slower
    /// but more stable learning; higher values mean faster but potentially unstable learning.
    /// </para>
    /// <para>Default: 0.01</para>
    /// </remarks>
    T LearningRate { get; }
}
