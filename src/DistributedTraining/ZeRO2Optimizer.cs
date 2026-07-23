using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements ZeRO Stage 2 optimizer - shards gradients and optimizer states across ranks.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// True ZeRO-2 implementation using ReduceScatter for gradient sharding. Each rank:
/// 1. Computes local gradients on full parameter set
/// 2. ReduceScatter: reduces gradients AND scatters them (each rank gets a shard)
/// 3. Updates only its shard of parameters using its shard of gradients
/// 4. AllGather: reconstructs full parameters from shards for next forward pass
///
/// This saves memory by distributing gradient storage and parameter updates across ranks.
/// </para>
/// <para><b>For Beginners:</b>
/// ZeRO-2 divides the work of storing and updating parameters across processes. Think of it
/// like a team where each person is responsible for maintaining a specific section of a large
/// document. Everyone reads the full document (forward pass), but each person only stores and
/// updates their assigned section (backward pass). Before the next iteration, they share their
/// sections to reconstruct the full document.
/// </para>
/// <para><b>Use Cases:</b>
/// - Large models where gradient memory is significant (billions of parameters)
/// - Want memory savings beyond DDP
/// - Good network for AllGather operations
/// - Works with ANY gradient-based optimizer (SGD, Adam, RMSprop, etc.)
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Very Good - gradients and optimizer states sharded (1/N of DDP)
/// - Communication: ReduceScatter + AllGather (vs AllReduce for DDP)
/// - Synchronization: Perfect - all ranks reconstruct identical parameters
/// - Complexity: Moderate - requires parameter sharding logic
/// - Best for: Large models with limited GPU memory
/// </para>
/// <para><b>Memory Savings vs DDP:</b>
/// - DDP: Each rank stores full gradients + full optimizer state
/// - ZeRO-2: Each rank stores 1/N gradients + 1/N optimizer state
/// - Savings increase linearly with world size
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ZeRO2Optimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a ZeRO-2 optimizer that shards gradients and optimizer states.
    /// </summary>
    /// <param name="wrappedOptimizer">The base optimizer to wrap (any gradient-based optimizer: SGD, Adam, RMSprop, etc.)</param>
    /// <param name="config">Configuration for distributed training communication</param>
    /// <exception cref="ArgumentException">If wrapped optimizer is not gradient-based</exception>
    public ZeRO2Optimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
        : base(wrappedOptimizer, config)
    {
        // Verify wrapped optimizer supports gradient operations
        if (wrappedOptimizer is not IGradientBasedOptimizer<T, TInput, TOutput>)
        {
            throw new ArgumentException(
                $"ZeRO-2 requires a gradient-based optimizer, but received {wrappedOptimizer.GetType().Name}. " +
                "Use gradient-based optimizers like SGD, Adam, RMSprop, etc.",
                nameof(wrappedOptimizer));
        }
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // CRITICAL: Opening barrier must execute BEFORE any divergent logic to synchronize all workers.
        Config.CommunicationBackend.Barrier();

        try
        {
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));

            // When cross-rank synchronization is off, fall back to a plain local step.
            if (!Config.AutoSyncGradients)
                return RunWrappedOptimizerStep(inputData);

            // ZeRO Stage-2 (Rajbhandari et al. 2020): gradients AND optimizer state are PARTITIONED.
            // RunShardedZeroStep with shardGradients:true computes gradients backward-only (no full
            // update), ReduceScatters them so each rank holds ONLY its averaged gradient shard (the
            // full averaged gradient is never materialized on any rank — the memory win over ZeRO-1),
            // updates just this rank's parameter shard with the wrapped optimizer (whose Adam m/v state
            // is therefore sized to the shard), and AllGathers the updated shards into the full vector.
            // CpuOffloadOptimizer is scoped to the update; CpuOffloadGradients drains the shard to CPU
            // before it is read; CpuOffloadParams drops the GPU param cache after write-back.
            var result = RunShardedZeroStep(inputData, shardGradients: true);
            return result;
        }
        finally
        {
            // CRITICAL: Closing barrier ALWAYS executes to prevent deadlock
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Not supported for ZeRO-2 (same contract as ZeRO-1/FSDP): optimizer state is PARTITIONED. Gradients
    /// are ReduceScattered so each rank holds only its shard, and the wrapped optimizer's UpdateParameters
    /// runs on ONLY that shard, so its Adam m/v state is advanced solely for this rank's parameters. There
    /// is no replicated cross-rank state to synchronize; returning silently would mislead the caller.
    /// </remarks>
    public override void SynchronizeOptimizerState()
        => throw new NotSupportedException(
            "ZeRO-2 optimizer state is partitioned across ranks (each rank owns only its shard's Adam " +
            "state); there is no replicated state to synchronize, so explicit state synchronization is " +
            "neither required nor supported.");

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
}
