using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides default implementation of distributed training configuration.
/// </summary>
/// <remarks>
/// <para>
/// This class wraps a sharding configuration and provides a simple way to enable/disable
/// distributed training in the PredictionModelBuilder. When enabled, models and optimizers
/// will be automatically wrapped with their distributed counterparts.
/// </para>
/// <para><b>For Beginners:</b> This is the configuration object you pass to PredictionModelBuilder
/// to enable distributed training.
///
/// It's very simple to use:
/// <code>
/// // Step 1: Create a communication backend
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
///
/// // Step 2: Create distributed training config
/// var distributedConfig = new DistributedTrainingConfiguration&lt;double&gt;(backend);
///
/// // Step 3: Use it in your builder
/// var result = builder
///     .ConfigureModel(model)
///     .ConfigureOptimizer(optimizer)
///     .ConfigureDistributedTraining(distributedConfig)
///     .Build(xTrain, yTrain);
/// </code>
///
/// The builder will automatically handle wrapping your model and optimizer for
/// distributed training across 4 processes!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
public class DistributedTrainingConfiguration<T> : IDistributedTrainingConfiguration<T>
{
    /// <inheritdoc/>
    public IShardingConfiguration<T> ShardingConfiguration { get; }

    /// <inheritdoc/>
    public bool IsEnabled { get; }

    /// <summary>
    /// Creates a new distributed training configuration with the specified sharding configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates an enabled distributed training configuration using
    /// the provided sharding configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a configuration that tells the framework to use
    /// distributed training.
    ///
    /// You provide a sharding configuration (which contains the communication backend and settings),
    /// and this wraps it in a way that PredictionModelBuilder understands.
    /// </para>
    /// </remarks>
    /// <param name="shardingConfiguration">The sharding configuration to use</param>
    /// <exception cref="ArgumentNullException">Thrown if shardingConfiguration is null</exception>
    public DistributedTrainingConfiguration(IShardingConfiguration<T> shardingConfiguration)
    {
        ShardingConfiguration = shardingConfiguration ??
            throw new ArgumentNullException(nameof(shardingConfiguration));
        IsEnabled = true;
    }

    /// <summary>
    /// Creates a new distributed training configuration with the specified communication backend.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This convenience constructor creates a default sharding configuration using the provided
    /// communication backend, then wraps it in a distributed training configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This is a shortcut for creating distributed training configuration.
    ///
    /// Instead of creating a sharding configuration first, you can just pass the communication
    /// backend directly:
    /// <code>
    /// var backend = new InMemoryCommunicationBackend&lt;double&gt;(0, 4);
    /// var config = new DistributedTrainingConfiguration&lt;double&gt;(backend);
    /// </code>
    ///
    /// This automatically creates a sharding configuration with good default settings.
    /// </para>
    /// </remarks>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <exception cref="ArgumentNullException">Thrown if communicationBackend is null</exception>
    public DistributedTrainingConfiguration(ICommunicationBackend<T> communicationBackend)
    {
        if (communicationBackend == null)
        {
            throw new ArgumentNullException(nameof(communicationBackend));
        }

        ShardingConfiguration = new ShardingConfiguration<T>(communicationBackend);
        IsEnabled = true;
    }

    /// <summary>
    /// Creates a disabled distributed training configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This private constructor creates a configuration where distributed training is disabled.
    /// It's used by the CreateDisabled factory method.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a configuration that tells the framework NOT to
    /// use distributed training.
    ///
    /// This is used internally when you don't want distributed training. You typically
    /// won't call this directly - just don't configure distributed training in the builder.
    /// </para>
    /// </remarks>
    private DistributedTrainingConfiguration()
    {
        ShardingConfiguration = null!; // Will never be accessed when IsEnabled is false
        IsEnabled = false;
    }

    /// <summary>
    /// Creates a disabled distributed training configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This factory method creates a configuration that indicates distributed training
    /// should not be used. The PredictionModelBuilder will skip distributed wrapping
    /// when this configuration is provided.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a configuration that says "don't use distributed training".
    ///
    /// You typically don't need this - just don't call ConfigureDistributedTraining
    /// on your builder if you don't want distributed training. But this can be useful
    /// if you want to explicitly disable it in certain situations.
    /// </para>
    /// </remarks>
    /// <returns>A disabled distributed training configuration</returns>
    public static DistributedTrainingConfiguration<T> CreateDisabled()
    {
        return new DistributedTrainingConfiguration<T>();
    }
}
