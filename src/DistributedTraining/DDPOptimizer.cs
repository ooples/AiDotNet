using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements DDP (Distributed Data Parallel) optimizer - standard AllReduce gradient synchronization.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// DDP optimizer is the standard distributed optimizer that works with DDPModel. After each optimization
/// step, gradients are synchronized across all processes using AllReduce with averaging. This ensures
/// all model replicas perform identical parameter updates and stay synchronized. This is PyTorch's
/// default distributed optimizer strategy.
/// </para>
/// <para><b>For Beginners:</b>
/// This optimizer works with DDP (Distributed Data Parallel) models. After computing gradients on
/// local data, it averages them across all processes so everyone updates their model identically.
/// Think of it like a study group where everyone does practice problems independently, then shares
/// and averages their answers before updating their notes.
/// </para>
/// <para><b>Use Cases:</b>
/// - Standard multi-GPU training with full model replicas
/// - Works with any optimizer (Adam, SGD, RMSprop, etc.)
/// - Default choice for distributed training
/// - Fast interconnects (NVLink, InfiniBand)
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Each process stores full optimizer state
/// - Communication: Low - only gradients synchronized (AllReduce)
/// - Complexity: Low - simplest distributed optimizer
/// - Best for: Standard distributed training scenarios
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class DDPOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    public DDPOptimizer(
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

        // Barrier to ensure all processes start together
        Config.CommunicationBackend.Barrier();

        // Each process optimizes on its local data
        var result = WrappedOptimizer.Optimize(inputData);

        // Synchronize parameters (average across all processes)
        if (Config.AutoSyncGradients && result.BestSolution != null)
        {
            SynchronizeParameters(result.BestSolution);
        }

        // Barrier to ensure all processes finish together
        Config.CommunicationBackend.Barrier();

        return result;
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // In DDP, optimizer states are not sharded
        // Each process maintains its own full optimizer state
        // No synchronization needed unless implementing state averaging
        // (which is not standard for DDP)
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
