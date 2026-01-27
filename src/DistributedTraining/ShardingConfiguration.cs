

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Default implementation of sharding configuration for distributed training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// This class holds all the settings that control how distributed training works.
/// You can create an instance with default settings or customize it for your needs.
/// </para>
/// <para>
/// Example:
/// <code>
/// var config = new ShardingConfiguration&lt;double&gt;(backend)
/// {
///     AutoSyncGradients = true,      // Automatically sync after each step
///     MinimumParameterGroupSize = 1024,  // Group small parameters together
///     EnableGradientCompression = false  // No compression for now
/// };
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
public class ShardingConfiguration<T> : IShardingConfiguration<T>
{
    /// <inheritdoc/>
    public bool AutoSyncGradients { get; set; } = true;

    /// <inheritdoc/>
    public ICommunicationBackend<T> CommunicationBackend { get; }

    /// <inheritdoc/>
    public int MinimumParameterGroupSize { get; set; } = 1024;

    /// <inheritdoc/>
    public bool EnableGradientCompression { get; set; } = false;

    /// <inheritdoc/>
    public T LearningRate { get; set; }

    /// <summary>
    /// Creates a new sharding configuration with the specified communication backend.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This creates the configuration object that tells the system how to handle
    /// distributed training. You must provide a communication backend (the system
    /// that allows processes to talk to each other).
    /// </para>
    /// </remarks>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <param name="learningRate">Learning rate for gradient application. Defaults to 0.01.</param>
    /// <exception cref="ArgumentNullException">Thrown if backend is null</exception>
    public ShardingConfiguration(ICommunicationBackend<T> communicationBackend, double learningRate = 0.01)
    {
        CommunicationBackend = communicationBackend ??
            throw new ArgumentNullException(nameof(communicationBackend));

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be greater than zero.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        LearningRate = ops.FromDouble(learningRate);
    }

    /// <summary>
    /// Creates a new sharding configuration with default settings and the specified backend.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is a convenient way to create a configuration with sensible defaults.
    /// The defaults are:
    /// - AutoSyncGradients = true (automatically sync gradients)
    /// - MinimumParameterGroupSize = 1024 (group small parameters)
    /// - EnableGradientCompression = false (no compression for simplicity)
    /// </para>
    /// </remarks>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <returns>A new configuration with default settings</returns>
    /// <exception cref="ArgumentNullException">Thrown if communicationBackend is null</exception>
    public static ShardingConfiguration<T> CreateDefault(ICommunicationBackend<T> communicationBackend)
    {
        if (communicationBackend == null)
        {
            throw new ArgumentNullException(nameof(communicationBackend));
        }

        return new ShardingConfiguration<T>(communicationBackend);
    }

    /// <summary>
    /// Creates a configuration optimized for high-bandwidth networks (like NVLink between GPUs).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Use this when your GPUs or machines are connected with very fast networks.
    /// It disables compression (not needed with fast networks) and uses smaller
    /// parameter groups (communication is fast enough to handle many messages).
    /// </para>
    /// </remarks>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <returns>A configuration optimized for high-bandwidth scenarios</returns>
    /// <exception cref="ArgumentNullException">Thrown if communicationBackend is null</exception>
    public static ShardingConfiguration<T> CreateForHighBandwidth(ICommunicationBackend<T> communicationBackend)
    {
        if (communicationBackend == null)
        {
            throw new ArgumentNullException(nameof(communicationBackend));
        }

        return new ShardingConfiguration<T>(communicationBackend)
        {
            AutoSyncGradients = true,
            MinimumParameterGroupSize = 512,  // Smaller groups OK with fast network
            EnableGradientCompression = false  // No need with high bandwidth
        };
    }

    /// <summary>
    /// Creates a configuration optimized for low-bandwidth networks (like machines connected over ethernet).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Use this when your machines are connected over slower networks like regular
    /// ethernet. It enables compression to reduce the amount of data sent and uses
    /// larger parameter groups to minimize the number of messages.
    /// </para>
    /// </remarks>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <returns>A configuration optimized for low-bandwidth scenarios</returns>
    /// <exception cref="ArgumentNullException">Thrown if communicationBackend is null</exception>
    public static ShardingConfiguration<T> CreateForLowBandwidth(ICommunicationBackend<T> communicationBackend)
    {
        if (communicationBackend == null)
        {
            throw new ArgumentNullException(nameof(communicationBackend));
        }

        return new ShardingConfiguration<T>(communicationBackend)
        {
            AutoSyncGradients = true,
            MinimumParameterGroupSize = 4096,  // Larger groups to reduce messages
            EnableGradientCompression = true   // Compress to save bandwidth
        };
    }
}
