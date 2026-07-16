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
        if (numMicroBatches < 1)
            throw new ArgumentOutOfRangeException(nameof(numMicroBatches),
                numMicroBatches, "numMicroBatches must be >= 1.");

        // Micro-batch PIPELINE SCHEDULING (splitting a batch into micro-batches and interleaving
        // their forward/backward across pipeline stages with gradient accumulation) is a MODEL-level
        // concern: it requires per-stage layer execution and inter-stage activation exchange, which is
        // implemented in PipelineParallelModel (GPipe, 1F1B, ZB-H1/H2, ZB-V, Interleaved-1F1B, Looped-BFS
        // — Huang et al. 2019 and follow-ups). This OPTIMIZER only performs the per-stage PARAMETER
        // UPDATE after the model has produced the (accumulated) gradients, so it cannot itself schedule
        // micro-batches. Advertising numMicroBatches > 1 here would be a false claim; reject it and
        // direct the caller to the model that actually implements the schedules.
        if (numMicroBatches > 1)
            throw new NotSupportedException(
                "PipelineParallelOptimizer performs only the per-stage optimizer update; micro-batch " +
                "pipeline scheduling and gradient accumulation are implemented in PipelineParallelModel " +
                $"(configure microBatchCount there and choose a schedule). Got numMicroBatches={numMicroBatches}.");

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

            // The micro-batch pipeline schedule (per-stage forward/backward interleaving + gradient
            // accumulation across micro-batches) runs in PipelineParallelModel; by the time control
            // reaches this optimizer the stage's gradients are already accumulated. This optimizer
            // therefore performs exactly the per-stage PARAMETER UPDATE — the numMicroBatches=1
            // contract enforced in the constructor. RunWrappedOptimizerStep engages
            // IShardingConfiguration.CpuOffloadOptimizer: the Adam m/v state + update run on CpuEngine.
            var result = RunWrappedOptimizerStep(inputData);

            // Each stage updates its own parameters
            if (Config.AutoSyncGradients && result.BestSolution != null)
            {
                // In pure pipeline parallelism, no cross-stage parameter sync needed
                // (each stage owns different parameters)
                // If combined with data parallelism, would sync within data-parallel group
            }

            OffloadParamsToCpu(result.BestSolution);
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
