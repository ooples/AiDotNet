using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements 3D Parallelism optimizer - coordinates across data, tensor, and pipeline dimensions.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// 3D Parallelism optimizer coordinates optimization across all three parallelism dimensions:
/// - Data parallel: synchronizes gradients across data-parallel replicas
/// - Tensor parallel: synchronizes within tensor-parallel groups
/// - Pipeline parallel: handles gradient accumulation across micro-batches
///
/// This requires managing separate communication groups for each dimension and ensuring
/// proper synchronization order to maintain correctness and efficiency.
/// </para>
/// <para><b>For Beginners:</b>
/// This is the most complex optimizer, coordinating all three types of parallelism.
/// It needs to handle:
/// 1. Averaging gradients across data-parallel replicas (GPUs processing different batches)
/// 2. Synchronizing tensor-parallel groups (GPUs sharing layer computations)
/// 3. Accumulating gradients from pipeline micro-batches
///
/// Think of it like coordinating a massive team split into departments (pipeline stages),
/// work groups (tensor parallel), and shifts (data parallel) - all need to sync at the right times.
/// </para>
/// <para><b>Use Cases:</b>
/// - Frontier-scale models (100B+ parameters)
/// - 100s to 1000s of GPUs
/// - Works with HybridShardedModel
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent - exploits all dimensions
/// - Communication: Complex - multiple sync patterns
/// - Complexity: Very High - most complex optimizer
/// - Best for: Largest scale training
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class HybridShardedOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    private readonly int _pipelineParallelSize;
    private readonly int _tensorParallelSize;
    private readonly int _dataParallelSize;

    public HybridShardedOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config,
        int pipelineParallelSize = 1,
        int tensorParallelSize = 1,
        int dataParallelSize = -1)
        : base(wrappedOptimizer, config)
    {
        _pipelineParallelSize = pipelineParallelSize;
        _tensorParallelSize = tensorParallelSize;

        if (dataParallelSize == -1)
        {
            _dataParallelSize = WorldSize / (pipelineParallelSize * tensorParallelSize);
        }
        else
        {
            _dataParallelSize = dataParallelSize;
        }

        if (_pipelineParallelSize * _tensorParallelSize * _dataParallelSize != WorldSize)
        {
            throw new ArgumentException(
                $"Pipeline ({_pipelineParallelSize}) × Tensor ({_tensorParallelSize}) × " +
                $"Data ({_dataParallelSize}) must equal WorldSize ({WorldSize})");
        }
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        Config.CommunicationBackend.Barrier();

        // 3D parallel optimization requires careful coordination:
        // 1. Pipeline: gradient accumulation across micro-batches
        // 2. Tensor: synchronization within tensor-parallel group
        // 3. Data: synchronization across data-parallel replicas

        try
        {
            var result = WrappedOptimizer.Optimize(inputData);

            if (Config.AutoSyncGradients && result.BestSolution != null)
            {
                // CRITICAL: HybridShardedOptimizer requires subgroup-aware gradient synchronization
                // which is not yet implemented. The base class SynchronizeParameters() performs
                // a full-world AllReduce that incorrectly averages parameters across ALL ranks,
                // destroying the tensor/pipeline shard structure.
                //
                // Correct implementation requires:
                // 1. First sync within tensor-parallel group (AllReduce for sum partial results)
                // 2. Then sync across data-parallel replicas (AllReduce for average gradients)
                // 3. Pipeline stages handle their own gradient accumulation
                //
                // This needs:
                // - Subgroup communicators for each parallelism dimension (tensor/data/pipeline groups)
                // - Gradient-specific synchronization (not parameter synchronization)
                // - Proper handling of optimizer states per dimension
                //
                // Without proper implementation, gradients remain unsynchronized or parameters
                // get incorrectly averaged, breaking 3D parallel semantics.

                throw new NotSupportedException(
                    "HybridShardedOptimizer with AutoSyncGradients=true requires subgroup-aware " +
                    "gradient synchronization that is not yet implemented. Proper 3D parallelism " +
                    "needs separate communicators for tensor-parallel, data-parallel, and pipeline-parallel " +
                    "groups. Use AutoSyncGradients=false and implement custom gradient synchronization, " +
                    "or use a simpler parallelism strategy (DDP, FSDP, ZeRO-2) for production use.");
            }

            return result;
        }
        finally
        {
            // Ensure barrier always executes to prevent deadlock,
            // even if WrappedOptimizer.Optimize throws an exception
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // In 3D parallelism, optimizer state management is complex:
        // - Pipeline stages: independent states (no sync)
        // - Tensor parallel: states partitioned by layer slice (sync within group)
        // - Data parallel: states replicated (no sync needed)

        // Full implementation would use process groups for each dimension
        // Framework placeholder
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize hybrid-specific configuration
        writer.Write(_pipelineParallelSize);
        writer.Write(_tensorParallelSize);
        writer.Write(_dataParallelSize);

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

        // Read hybrid-specific configuration
        int savedPipelineParallelSize = reader.ReadInt32();
        int savedTensorParallelSize = reader.ReadInt32();
        int savedDataParallelSize = reader.ReadInt32();

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

        if (savedPipelineParallelSize != _pipelineParallelSize ||
            savedTensorParallelSize != _tensorParallelSize ||
            savedDataParallelSize != _dataParallelSize)
        {
            throw new InvalidOperationException(
                $"Hybrid parallelism configuration mismatch. Optimizer was saved with " +
                $"pipeline={savedPipelineParallelSize}, tensor={savedTensorParallelSize}, data={savedDataParallelSize}, " +
                $"but current configuration has pipeline={_pipelineParallelSize}, " +
                $"tensor={_tensorParallelSize}, data={_dataParallelSize}.");
        }

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        WrappedOptimizer.Deserialize(optimizerData);
    }
}
