using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Local SGD distributed training optimizer - parameter averaging after local optimization.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Local SGD allows each worker to perform multiple local optimization steps independently,
/// then synchronizes model parameters (not gradients) across all workers using AllReduce averaging.
/// This reduces communication frequency compared to traditional DDP while maintaining convergence.
/// Based on "Don't Use Large Mini-Batches, Use Local SGD" (Lin et al., 2020).
/// </para>
/// <para><b>For Beginners:</b>
/// Unlike traditional DDP which synchronizes gradients before every parameter update, Local SGD
/// lets each worker train independently for several steps, then averages the final model parameters.
/// Think of it like students studying independently for a week, then meeting to average their
/// understanding, rather than checking answers after every practice problem.
/// </para>
/// <para><b>Key Difference from DDP:</b>
/// - **Local SGD (this class)**: Optimize locally → Average PARAMETERS → Continue training
/// - **True DDP**: Compute gradients → Average GRADIENTS → Apply averaged gradients → Continue training
/// </para>
/// <para><b>Use Cases:</b>
/// - Reducing communication frequency in distributed training
/// - Slower network connections where communication is expensive
/// - Works with any optimizer (Adam, SGD, RMSprop, etc.)
/// - Large models where parameter synchronization dominates training time
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Each process stores full model and optimizer state
/// - Communication: Very low - parameters synchronized less frequently than gradients
/// - Convergence: Slightly different trajectory than DDP but reaches similar final accuracy
/// - Complexity: Low - straightforward parameter averaging
/// - Best for: Communication-constrained distributed training
/// </para>
/// <para><b>Production Note:</b>
/// For true DDP (gradient averaging), use GradientCompressionOptimizer with compression ratio = 1.0,
/// which properly averages gradients before parameter updates.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class LocalSGDOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a Local SGD optimizer that averages parameters across workers.
    /// </summary>
    /// <param name="wrappedOptimizer">The base optimizer to wrap (SGD, Adam, etc.)</param>
    /// <param name="config">Configuration for distributed training communication</param>
    public LocalSGDOptimizer(
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
        // In Local SGD, optimizer states are not sharded
        // Each process maintains its own full optimizer state
        // No synchronization needed unless implementing state averaging
        // (which is not standard for Local SGD)
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
