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
        // Opening barrier: every rank starts the collective sequence together. Validation happens AFTER
        // it (and inside the try) so a rank that receives null still reaches the closing barrier and
        // cannot strand its peers.
        Config.CommunicationBackend.Barrier();
        try
        {
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));

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
    /// <remarks>
    /// Not supported for ZeRO-1: optimizer state is PARTITIONED, not replicated. Each rank calls the wrapped
    /// optimizer's UpdateParameters with ONLY its parameter shard, so the Adam m/v state is allocated and
    /// advanced solely for that shard — no rank ever holds another rank's state. There is therefore no
    /// replicated cross-rank state to synchronize; returning silently would mislead a caller into believing
    /// a synchronization occurred. (Persisting the full optimizer state for a checkpoint is a serialization
    /// concern — each rank saves its own shard state — not a hot-path synchronization.)
    /// </remarks>
    public override void SynchronizeOptimizerState()
        => throw new NotSupportedException(
            "ZeRO-1 optimizer state is partitioned across ranks (each rank owns only its shard's Adam " +
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
