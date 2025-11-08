using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides extension methods for easily enabling distributed training on models and optimizers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Extension methods allow you to add new functionality to existing classes without modifying them.
/// These extensions add a simple .AsDistributed() method to models and optimizers,
/// making it incredibly easy to enable distributed training with just one line of code!
/// </para>
/// <para>
/// Example:
/// <code>
/// // Original model
/// var model = new NeuralNetworkModel&lt;double&gt;(...);
///
/// // Make it distributed with one line!
/// var distributedModel = model.AsDistributed(backend);
///
/// // That's it! Now train as usual
/// distributedModel.Train(inputs, outputs);
/// </code>
/// </para>
/// </remarks>
public static class DistributedExtensions
{
    /// <summary>
    /// Wraps a model with distributed training capabilities using the specified communication backend.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is the "magic" method that makes any model distributed. Just call:
    /// myModel.AsDistributed(backend)
    /// </para>
    /// <para>
    /// What it does:
    /// 1. Creates a configuration for distributed training
    /// 2. Wraps your model in a ShardedModel
    /// 3. Returns the distributed version of your model
    /// </para>
    /// <para>
    /// The distributed model works exactly like the original, but automatically:
    /// - Shards parameters across processes
    /// - Synchronizes gradients during training
    /// - Coordinates with other processes
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <typeparam name="TInput">The input type for the model</typeparam>
    /// <typeparam name="TOutput">The output type for the model</typeparam>
    /// <param name="model">The model to make distributed</param>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <returns>A distributed version of the model</returns>
    /// <exception cref="ArgumentNullException">Thrown if model or backend is null</exception>
    public static IShardedModel<T, TInput, TOutput> AsDistributed<T, TInput, TOutput>(
        this IFullModel<T, TInput, TOutput> model,
        ICommunicationBackend<T> communicationBackend)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (communicationBackend == null)
        {
            throw new ArgumentNullException(nameof(communicationBackend));
        }

        // Create default configuration
        var config = new ShardingConfiguration<T>(communicationBackend);

        return new ShardedModel<T, TInput, TOutput>(model, config);
    }

    /// <summary>
    /// Wraps a model with distributed training capabilities using a custom configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is the same as the basic AsDistributed(), but lets you customize settings.
    /// Use this when you need more control over how distributed training works.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// var config = new ShardingConfiguration&lt;double&gt;(backend)
    /// {
    ///     AutoSyncGradients = true,
    ///     MinimumParameterGroupSize = 2048,
    ///     EnableGradientCompression = true
    /// };
    /// var distributedModel = myModel.AsDistributed(config);
    /// </code>
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <typeparam name="TInput">The input type for the model</typeparam>
    /// <typeparam name="TOutput">The output type for the model</typeparam>
    /// <param name="model">The model to make distributed</param>
    /// <param name="configuration">The sharding configuration to use</param>
    /// <returns>A distributed version of the model</returns>
    /// <exception cref="ArgumentNullException">Thrown if model or configuration is null</exception>
    public static IShardedModel<T, TInput, TOutput> AsDistributed<T, TInput, TOutput>(
        this IFullModel<T, TInput, TOutput> model,
        IShardingConfiguration<T> configuration)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (configuration == null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        return new ShardedModel<T, TInput, TOutput>(model, configuration);
    }

    /// <summary>
    /// Wraps an optimizer with distributed training capabilities.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Just like models, optimizers can be made distributed with one line:
    /// myOptimizer.AsDistributed(backend)
    /// </para>
    /// <para>
    /// What it does:
    /// 1. Creates a configuration for distributed training
    /// 2. Wraps your optimizer in a ShardedOptimizer
    /// 3. Returns the distributed version
    /// </para>
    /// <para>
    /// The distributed optimizer coordinates optimization across all processes,
    /// ensuring they stay synchronized.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <typeparam name="TInput">The input type for the model</typeparam>
    /// <typeparam name="TOutput">The output type for the model</typeparam>
    /// <param name="optimizer">The optimizer to make distributed</param>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <returns>A distributed version of the optimizer</returns>
    /// <exception cref="ArgumentNullException">Thrown if optimizer or backend is null</exception>
    public static IShardedOptimizer<T, TInput, TOutput> AsDistributed<T, TInput, TOutput>(
        this IOptimizer<T, TInput, TOutput> optimizer,
        ICommunicationBackend<T> communicationBackend)
    {
        if (optimizer == null)
        {
            throw new ArgumentNullException(nameof(optimizer));
        }

        if (communicationBackend == null)
        {
            throw new ArgumentNullException(nameof(communicationBackend));
        }

        // Create default configuration
        var config = new ShardingConfiguration<T>(communicationBackend);

        return new ShardedOptimizer<T, TInput, TOutput>(optimizer, config);
    }

    /// <summary>
    /// Wraps an optimizer with distributed training capabilities using a custom configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Same as the basic AsDistributed() for optimizers, but with custom settings.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// var config = ShardingConfiguration&lt;double&gt;.CreateForLowBandwidth(backend);
    /// var distributedOptimizer = myOptimizer.AsDistributed(config);
    /// </code>
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <typeparam name="TInput">The input type for the model</typeparam>
    /// <typeparam name="TOutput">The output type for the model</typeparam>
    /// <param name="optimizer">The optimizer to make distributed</param>
    /// <param name="configuration">The sharding configuration to use</param>
    /// <returns>A distributed version of the optimizer</returns>
    /// <exception cref="ArgumentNullException">Thrown if optimizer or configuration is null</exception>
    public static IShardedOptimizer<T, TInput, TOutput> AsDistributed<T, TInput, TOutput>(
        this IOptimizer<T, TInput, TOutput> optimizer,
        IShardingConfiguration<T> configuration)
    {
        if (optimizer == null)
        {
            throw new ArgumentNullException(nameof(optimizer));
        }

        if (configuration == null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        return new ShardedOptimizer<T, TInput, TOutput>(optimizer, configuration);
    }

    /// <summary>
    /// Creates a distributed model with optimized settings for high-bandwidth networks.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Use this when your GPUs are connected with fast connections (like NVLink).
    /// It automatically configures settings optimized for high-speed networks.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <typeparam name="TInput">The input type for the model</typeparam>
    /// <typeparam name="TOutput">The output type for the model</typeparam>
    /// <param name="model">The model to make distributed</param>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <returns>A distributed model optimized for high-bandwidth networks</returns>
    public static IShardedModel<T, TInput, TOutput> AsDistributedForHighBandwidth<T, TInput, TOutput>(
        this IFullModel<T, TInput, TOutput> model,
        ICommunicationBackend<T> communicationBackend)
    {
        var config = ShardingConfiguration<T>.CreateForHighBandwidth(communicationBackend);
        return model.AsDistributed(config);
    }

    /// <summary>
    /// Creates a distributed model with optimized settings for low-bandwidth networks.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Use this when your machines are connected over slower networks (like ethernet).
    /// It automatically configures settings to minimize network traffic.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <typeparam name="TInput">The input type for the model</typeparam>
    /// <typeparam name="TOutput">The output type for the model</typeparam>
    /// <param name="model">The model to make distributed</param>
    /// <param name="communicationBackend">The communication backend to use</param>
    /// <returns>A distributed model optimized for low-bandwidth networks</returns>
    public static IShardedModel<T, TInput, TOutput> AsDistributedForLowBandwidth<T, TInput, TOutput>(
        this IFullModel<T, TInput, TOutput> model,
        ICommunicationBackend<T> communicationBackend)
    {
        var config = ShardingConfiguration<T>.CreateForLowBandwidth(communicationBackend);
        return model.AsDistributed(config);
    }
}
