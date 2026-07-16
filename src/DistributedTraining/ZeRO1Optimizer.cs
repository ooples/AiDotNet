using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements ZeRO Stage 1 optimizer - shards optimizer states only.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// ZeRO-1 optimizer shards optimizer states (momentum buffers, variance estimates) across processes
/// while keeping parameters and gradients replicated. This reduces memory overhead from optimizer
/// state (which can be 4x the model size for Adam: fp32 params + momentum + variance + gradients).
/// When needed for updates, optimizer states are gathered from their respective owners.
/// </para>
/// <para><b>For Beginners:</b>
/// This optimizer saves memory by splitting the optimizer's internal memory (like momentum in Adam)
/// across processes. The model parameters are still fully replicated, but each process only stores
/// a portion of the optimizer's "memory" or "state". When it's time to update parameters, processes
/// share their pieces of the optimizer state as needed.
/// </para>
/// <para><b>Use Cases:</b>
/// - Using stateful optimizers (Adam, RMSprop) with limited memory
/// - Want memory savings without full ZeRO-3/FSDP complexity
/// - Works well with ZeRO1Model
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Good - saves ~4x memory from optimizer states for Adam
/// - Communication: Low - same as DDP plus occasional state gather
/// - Complexity: Moderate - state sharding adds some complexity
/// - Best for: Memory-constrained scenarios with stateful optimizers
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ZeRO1Optimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    public ZeRO1Optimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
        : base(wrappedOptimizer, config)
    {
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        // Opening barrier: every rank starts the collective sequence together.
        Config.CommunicationBackend.Barrier();
        try
        {
            // When cross-rank synchronization is off, fall back to a plain local step (equivalent
            // to non-distributed training on this rank).
            if (!Config.AutoSyncGradients)
                return RunWrappedOptimizerStep(inputData);

            // ZeRO Stage-1 (Rajbhandari et al. 2020): gradients stay REPLICATED (AllReduce), while the
            // optimizer state (Adam m/v) AND the parameter update are PARTITIONED — each rank updates
            // only its parameter shard with the wrapped optimizer, so that optimizer's persistent state
            // is sized to the shard. RunShardedZeroStep computes gradients backward-only, AllReduces
            // them, applies the sharded update (with CpuOffloadOptimizer scoped to the update), and
            // AllGathers the updated shards back into the full parameter vector.
            return RunShardedZeroStep(inputData, shardGradients: false);
        }
        finally
        {
            // Closing barrier always runs (even on exception) so no rank is left waiting on a peer.
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // ZeRO-1 optimizer-state partitioning is achieved INTRINSICALLY inside Optimize: each rank
        // calls the wrapped optimizer's UpdateParameters with ONLY its parameter shard, so the wrapped
        // optimizer's persistent Adam m/v state is allocated and advanced solely for that shard — no
        // rank ever holds the other ranks' state. Consequently there is no per-step state exchange to
        // perform here: each rank owns and retains its shard's state across steps, and the only
        // cross-rank exchange needed (the updated PARAMETERS) is the AllGather already done in the step.
        // This method therefore intentionally does no work; state ownership is not a placeholder but a
        // structural property of the sharded update. (A checkpoint that needs the full optimizer state
        // would AllGather each rank's shard state; that belongs to serialization, not the hot path.)
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
