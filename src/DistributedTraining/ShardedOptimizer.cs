using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements a distributed optimizer wrapper that coordinates optimization across multiple processes.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// This class wraps any existing optimizer (like Adam, SGD, etc.) and makes it work across
/// multiple GPUs or machines. It automatically handles:
/// - Synchronizing gradients across all processes
/// - Coordinating parameter updates
/// - Ensuring all processes stay in sync
/// </para>
/// <para>
/// Think of it like a team of coaches working together - each has their own expertise
/// (the wrapped optimizer), but they coordinate their efforts to train the team effectively.
/// </para>
/// <para>
/// Example:
/// <code>
/// // Original optimizer
/// var optimizer = new AdamOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, options);
///
/// // Wrap it for distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var distributedOptimizer = new ShardedOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     optimizer, config);
///
/// // Now optimize as usual - distributed magic happens automatically!
/// var result = distributedOptimizer.Optimize(inputData);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ShardedOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a new sharded optimizer wrapping an existing optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This constructor takes your existing optimizer and makes it distributed.
    /// You provide:
    /// 1. The optimizer you want to make distributed
    /// 2. A configuration that tells us how to do the distribution
    /// </para>
    /// <para>
    /// The optimizer will automatically synchronize across all processes during optimization.
    /// </para>
    /// </remarks>
    /// <param name="wrappedOptimizer">The optimizer to wrap with distributed capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if optimizer or config is null</exception>
    public ShardedOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
        : base(wrappedOptimizer, config)
    {
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
        {
            throw new ArgumentNullException(nameof(inputData));
        }

        // Ensure all processes start together
        Config.CommunicationBackend.Barrier();

        // If the model is a sharded model, it will handle gradient synchronization
        // Otherwise, we need to handle it ourselves
        bool modelHandlesSync = inputData.Model is IShardedModel<T, TInput, TOutput>;

        // Perform optimization on the wrapped optimizer
        var result = WrappedOptimizer.Optimize(inputData);

        // If model doesn't handle synchronization and auto-sync is enabled, sync parameters
        if (!modelHandlesSync && Config.AutoSyncGradients)
        {
            SynchronizeParameters(result.BestModel);
        }

        // Synchronize optimizer state if needed
        SynchronizeOptimizerState();

        // Ensure all processes finish together
        Config.CommunicationBackend.Barrier();

        return result;
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // For now, this is a placeholder
        // In a full implementation, we would synchronize optimizer-specific state
        // like momentum buffers, variance estimates (for Adam), etc.

        // Different optimizers have different state to sync:
        // - SGD with momentum: velocity vectors
        // - Adam: first and second moment estimates
        // - RMSprop: squared gradient moving average

        // This would require either:
        // 1. Extending IOptimizer with state access methods
        // 2. Type-specific handling for known optimizer types
        // 3. A generic state serialization mechanism

        // For the MVP, we assume stateless or that the wrapped optimizer handles its own state
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize sharding configuration info
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(Config.AutoSyncGradients);
        writer.Write(Config.MinimumParameterGroupSize);
        writer.Write(Config.EnableGradientCompression);

        // Serialize wrapped optimizer
        var optimizerData = WrappedOptimizer.Serialize();
        writer.Write(optimizerData.Length);
        writer.Write(optimizerData);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read sharding configuration (for validation)
        int savedWorldSize = reader.ReadInt32();
        int savedRank = reader.ReadInt32();
        reader.ReadBoolean(); // AutoSyncGradients
        reader.ReadInt32(); // MinimumParameterGroupSize
        reader.ReadBoolean(); // EnableGradientCompression

        if (savedWorldSize != WorldSize)
        {
            throw new InvalidOperationException(
                $"World size mismatch. Optimizer was saved with {savedWorldSize} processes, " +
                $"but current configuration has {WorldSize} processes.");
        }

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        WrappedOptimizer.Deserialize(optimizerData);
    }
}
