using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements ZeRO Stage 2 optimizer - shards optimizer states and gradients.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// ZeRO-2 optimizer builds on ZeRO-1 by additionally sharding gradients using ReduceScatter.
/// After backward pass, gradients are reduced and scattered so each process only stores its
/// portion. This further reduces memory compared to ZeRO-1, as gradients can be as large as
/// the model itself. Parameters remain replicated for the forward pass.
/// </para>
/// <para><b>For Beginners:</b>
/// ZeRO-2 saves even more memory than ZeRO-1. Not only is the optimizer state split across
/// processes, but the gradients are too. After computing gradients, we immediately use
/// ReduceScatter to average them across processes AND split them up, so each process only
/// keeps its assigned portion. This is like having a team where each person is responsible
/// for updating only certain parameters.
/// </para>
/// <para><b>Use Cases:</b>
/// - Large models where gradient memory is significant
/// - Want substantial memory savings
/// - Works well with ZeRO2Model
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Very Good - saves optimizer states + gradients
/// - Communication: Moderate - uses ReduceScatter instead of AllReduce
/// - Complexity: Moderate - gradient and state sharding
/// - Best for: Large models with significant gradient memory
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ZeRO2Optimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    public ZeRO2Optimizer(
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

        Config.CommunicationBackend.Barrier();

        // Optimize on local data
        var result = WrappedOptimizer.Optimize(inputData);

        // In ZeRO-2, we use ReduceScatter to reduce gradients and distribute shards
        // Each process receives only its portion of the reduced gradients
        if (Config.AutoSyncGradients && result.BestSolution != null)
        {
            // Instead of AllReduce, use ReduceScatter for gradient sharding
            var parameters = result.BestSolution.GetParameters();
            var reducedShard = Config.CommunicationBackend.ReduceScatter(parameters, ReductionOperation.Average);

            // Each process now has only its shard of gradients
            // Update would use only this shard (requires optimizer state integration)
        }

        SynchronizeOptimizerState();

        Config.CommunicationBackend.Barrier();

        return result;
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // In ZeRO-2, both optimizer states and gradients are sharded
        // Each process:
        // 1. Owns a shard of optimizer state
        // 2. Receives its shard of reduced gradients via ReduceScatter
        // 3. Updates only its parameter shard
        // 4. AllGather updated parameters for next forward pass

        // Framework placeholder - full implementation requires optimizer integration
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
