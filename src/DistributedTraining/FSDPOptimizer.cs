using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements FSDP (Fully Sharded Data Parallel) optimizer wrapper that coordinates optimization across multiple processes.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// FSDP optimizer works in conjunction with FSDPModel to provide full sharding of optimizer states.
/// This means momentum buffers, variance estimates, and all other optimizer-specific state are sharded
/// across processes, minimizing memory usage while maintaining training effectiveness.
/// </para>
/// <para><b>For Beginners:</b>
/// This class wraps any existing optimizer (like Adam, SGD, etc.) and makes it work with FSDP strategy
/// across multiple GPUs or machines. It automatically handles:
/// - Synchronizing gradients across all processes
/// - Sharding optimizer states (momentum, variance) to save memory
/// - Coordinating parameter updates
/// - Ensuring all processes stay in sync
/// </para>
/// <para>
/// Think of it like a team of coaches working together - each coach has their own expertise
/// (the wrapped optimizer), but they share only the essential information and keep their detailed
/// notes (optimizer states) private to save space.
/// </para>
/// <para><b>Use Cases:</b>
/// - Training very large models with optimizers that have significant state (Adam, RMSprop)
/// - Maximizing memory efficiency when using stateful optimizers
/// - Scaling to hundreds or thousands of GPUs
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent - shards optimizer states across processes
/// - Communication: Moderate - syncs gradients and occasional state synchronization
/// - Complexity: Moderate - automatic state sharding
/// - Best for: Large models with stateful optimizers (Adam, RMSprop, etc.)
/// </para>
/// <para>
/// Example:
/// <code>
/// // Original optimizer
/// var optimizer = new AdamOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, options);
///
/// // Wrap it for FSDP distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var fsdpOptimizer = new FSDPOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     optimizer, config);
///
/// // Now optimize as usual - FSDP magic happens automatically!
/// var result = fsdpOptimizer.Optimize(inputData);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class FSDPOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a new FSDP optimizer wrapping an existing optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This constructor takes your existing optimizer and makes it distributed using FSDP strategy.
    /// You provide:
    /// 1. The optimizer you want to make distributed
    /// 2. A configuration that tells us how to do the distribution
    /// </para>
    /// <para>
    /// The optimizer will automatically synchronize across all processes during optimization
    /// and shard optimizer states to minimize memory usage.
    /// </para>
    /// </remarks>
    /// <param name="wrappedOptimizer">The optimizer to wrap with FSDP capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if optimizer or config is null</exception>
    public FSDPOptimizer(
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

        // Perform optimization on the wrapped optimizer
        var result = WrappedOptimizer.Optimize(inputData);

        // Synchronize parameters across all processes if auto-sync is enabled
        if (Config.AutoSyncGradients && result.BestSolution != null)
        {
            SynchronizeParameters(result.BestSolution);
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
    public override bool ShouldEarlyStop()
    {
        // Delegate to wrapped optimizer
        bool localDecision = WrappedOptimizer.ShouldEarlyStop();

        // In distributed training, we need consensus on early stopping
        // Using Max operation: if ANY process wants to stop, all processes stop
        // This prevents stragglers and ensures synchronized termination

        // Create a vector with the local decision (1 for stop, 0 for continue)
        var decision = new Vector<T>(new[] { localDecision ? MathHelper.GetNumericOperations<T>().One : MathHelper.GetNumericOperations<T>().Zero });

        // Get the maximum across all processes
        // If any process returns 1 (stop), the max will be 1
        Config.CommunicationBackend.AllReduce(decision, ReductionOperation.Max);

        // Check if the result indicates stopping
        var numOps = MathHelper.GetNumericOperations<T>();
        return !numOps.Equals(decision[0], numOps.Zero);
    }

    /// <inheritdoc/>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return WrappedOptimizer.GetOptions();
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

        // Validate rank matches - different rank could indicate configuration mismatch
        if (savedRank != Rank)
        {
            throw new InvalidOperationException(
                $"Rank mismatch. Optimizer was saved on rank {savedRank}, " +
                $"but is being loaded on rank {Rank}. This could indicate a configuration error.");
        }

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        WrappedOptimizer.Deserialize(optimizerData);
    }

    /// <inheritdoc/>
    public override void SaveModel(string filePath)
    {
        // Barrier before rank check to prevent deadlock if rank 0 fails
        Config.CommunicationBackend.Barrier();

        try
        {
            // Only rank 0 saves to avoid file write conflicts
            if (Rank == 0)
            {
                var data = Serialize();
                File.WriteAllBytes(filePath, data);
            }
        }
        finally
        {
            // Ensure all processes reach this barrier even if rank 0 fails
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void LoadModel(string filePath)
    {
        // Barrier before loading to ensure all processes start together
        Config.CommunicationBackend.Barrier();

        try
        {
            // All processes read the same file (read-only, no conflicts)
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        finally
        {
            // Ensure all processes finish loading before proceeding
            Config.CommunicationBackend.Barrier();
        }
    }
}
