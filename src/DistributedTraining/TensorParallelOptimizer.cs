using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Tensor Parallel optimizer - coordinates updates for tensor-parallel layers.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Tensor parallel optimizer coordinates optimization for models using tensor parallelism.
/// Different parts of each layer are distributed across processes, requiring careful
/// synchronization. For column-parallel layers, an AllReduce is needed after computation.
/// For row-parallel layers, synchronization happens before computation. The optimizer
/// ensures proper gradient flow and parameter updates across the tensor-parallel group.
/// </para>
/// <para><b>For Beginners:</b>
/// This optimizer works with tensor parallel models where individual layers are split.
/// Since each process only has part of each layer, we need to carefully coordinate
/// gradient synchronization. Different layer types require different synchronization
/// patterns (before or after the computation).
/// </para>
/// <para><b>Use Cases:</b>
/// - Works with TensorParallelModel
/// - Transformer models with large layers
/// - Very wide models
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent for wide layers
/// - Communication: High - frequent synchronization within layers
/// - Complexity: Very High - layer-specific patterns
/// - Best for: Large transformers, fast interconnects
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class TensorParallelOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    public TensorParallelOptimizer(
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

        // Optimize on local tensor-parallel shard
        var result = WrappedOptimizer.Optimize(inputData);

        // Synchronize across tensor-parallel group
        if (Config.AutoSyncGradients && result.BestSolution != null)
        {
            SynchronizeParameters(result.BestSolution);
        }

        Config.CommunicationBackend.Barrier();

        return result;
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // In tensor parallelism, each process owns a slice of each layer
        // Optimizer states are partitioned similarly
        // Synchronization happens at the parameter level during forward/backward
        // rather than at optimizer state level
    }
}
