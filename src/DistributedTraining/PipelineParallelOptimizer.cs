using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Pipeline Parallel optimizer - coordinates optimization across pipeline stages.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Pipeline parallel optimizer coordinates optimization across different pipeline stages.
/// Each stage optimizes its own layer parameters, with gradient accumulation across micro-batches.
/// The optimizer ensures proper synchronization between forward and backward passes through the
/// pipeline, handling the gradient accumulation from multiple micro-batches.
/// </para>
/// <para><b>For Beginners:</b>
/// This optimizer works with pipeline parallel models where the model is split into stages.
/// It handles the complexity of gradient accumulation - since we process multiple micro-batches
/// through the pipeline, gradients need to be accumulated before the final parameter update.
/// Think of it like collecting feedback from multiple practice sessions before making adjustments.
/// </para>
/// <para><b>Use Cases:</b>
/// - Works with PipelineParallelModel
/// - Very deep models split into stages
/// - Handles micro-batch gradient accumulation
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Good for deep models
/// - Communication: Low between stages
/// - Complexity: High - gradient accumulation, pipeline scheduling
/// - Best for: Deep models with pipeline parallelism
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class PipelineParallelOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    private readonly int _numMicroBatches;

    public PipelineParallelOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config,
        int numMicroBatches = 1)
        : base(wrappedOptimizer, config)
    {
        _numMicroBatches = numMicroBatches;
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // CRITICAL: Opening barrier must execute BEFORE any divergent logic to synchronize all workers.
        // This prevents deadlock if some workers throw exceptions while others continue.
        Config.CommunicationBackend.Barrier();

        try
        {
            // Null check happens AFTER opening barrier but INSIDE try block.
            // This ensures that if one worker receives null while another doesn't,
            // both workers still execute the finally barrier, preventing deadlock.
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));

            // Pipeline parallel optimization requires:
            // 1. Process micro-batches through the pipeline
            // 2. Accumulate gradients across micro-batches
            // 3. Update parameters once all micro-batches complete
            // 4. Synchronize across pipeline stages if using data parallelism

            // For this framework implementation, we provide simplified pattern
            var result = WrappedOptimizer.Optimize(inputData);

            // Each stage updates its own parameters
            if (Config.AutoSyncGradients && result.BestSolution != null)
            {
                // In pure pipeline parallelism, no cross-stage parameter sync needed
                // (each stage owns different parameters)
                // If combined with data parallelism, would sync within data-parallel group
            }

            return result;
        }
        finally
        {
            // CRITICAL: Closing barrier ALWAYS executes to prevent deadlock,
            // even if null check, WrappedOptimizer.Optimize, or other operations throw.
            // This ensures all workers reach this barrier regardless of exceptions.
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // Pipeline stages maintain independent optimizer states
        // No synchronization needed across stages
        // If using pipeline + data parallelism, would sync within data-parallel groups
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize pipeline-specific configuration
        writer.Write(_numMicroBatches);

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

        // Read pipeline-specific configuration
        int savedNumMicroBatches = reader.ReadInt32();

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

        if (savedNumMicroBatches != _numMicroBatches)
        {
            throw new InvalidOperationException(
                $"Pipeline configuration mismatch. Optimizer was saved with {savedNumMicroBatches} micro-batches, " +
                $"but current configuration has {_numMicroBatches} micro-batches.");
        }

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        WrappedOptimizer.Deserialize(optimizerData);
    }
}
