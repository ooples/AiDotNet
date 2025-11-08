using AiDotNet.DistributedTraining;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the configuration for distributed training capabilities.
/// </summary>
/// <remarks>
/// <para>
/// This interface provides access to the sharding configuration that controls
/// how models and optimizers are distributed across multiple processes or GPUs.
/// It's used by PredictionModelBuilder to automatically wrap models and optimizers
/// for distributed training when configured.
/// </para>
/// <para><b>For Beginners:</b> This configuration tells the AI framework how to split
/// your model training across multiple GPUs or computers.
///
/// Think of it like planning how a team will work together:
/// - Who communicates with whom (communication backend)
/// - How work is divided (sharding strategy)
/// - When to share updates (auto-sync settings)
///
/// When you configure this, the framework automatically handles all the complexity
/// of distributed training for you. You just train your model as normal, and it
/// runs across multiple GPUs automatically!
///
/// Example:
/// <code>
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new DistributedTrainingConfiguration&lt;double&gt;(backend);
///
/// var result = new PredictionModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(myModel)
///     .ConfigureOptimizer(myOptimizer)
///     .ConfigureDistributedTraining(config)  // Enable distributed training!
///     .Build(xTrain, yTrain);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
public interface IDistributedTrainingConfiguration<T>
{
    /// <summary>
    /// Gets the sharding configuration for distributed training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The sharding configuration contains all settings related to how parameters
    /// are split across processes, including the communication backend, auto-sync settings,
    /// parameter grouping size, and compression options.
    /// </para>
    /// <para><b>For Beginners:</b> This contains all the detailed settings about how
    /// distributed training works.
    ///
    /// It includes:
    /// - The communication system (how processes talk to each other)
    /// - Whether to automatically sync gradients
    /// - How to group small parameters together for efficiency
    /// - Whether to compress data before sending it
    ///
    /// You typically don't need to access this directly - the framework uses it
    /// automatically when building your model.
    /// </para>
    /// </remarks>
    IShardingConfiguration<T> ShardingConfiguration { get; }

    /// <summary>
    /// Gets whether distributed training is enabled.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates whether the configuration is active and should be used
    /// to wrap models and optimizers for distributed training.
    /// </para>
    /// <para><b>For Beginners:</b> This is a simple on/off switch.
    ///
    /// - True: Distributed training is enabled, models will be split across GPUs
    /// - False: Regular single-GPU training
    ///
    /// This helps the framework know whether to automatically wrap your model
    /// and optimizer for distributed training.
    /// </para>
    /// </remarks>
    bool IsEnabled { get; }
}
