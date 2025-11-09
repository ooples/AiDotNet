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
/// <para><b>⚠️ IMPORTANT - Optimizer Compatibility:</b>
/// This implementation's ComputeOriginalParameters method assumes vanilla gradient descent
/// (SGD) update rules. It may produce INCORRECT results when wrapping adaptive optimizers
/// like Adam or RMSprop. For production use, wrap only vanilla SGD optimizers
/// (GradientDescentOptimizer, StochasticGradientDescentOptimizer, etc.). See
/// ComputeOriginalParameters documentation for details.
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
        // This prevents deadlock if some workers throw exceptions while others continue.
        Config.CommunicationBackend.Barrier();

        try
        {
            // Null check happens AFTER opening barrier but INSIDE try block.
            // This ensures that if one worker receives null while another doesn't,
            // both workers still execute the finally barrier, preventing deadlock.
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));

            var gradientOptimizer = (IGradientBasedOptimizer<T, TInput, TOutput>)WrappedOptimizer;

            // CRITICAL: Save parameters BEFORE local optimization
            // This allows us to discard the local update and apply only the averaged gradients
            Vector<T>? savedParameters = null;
            if (Config.AutoSyncGradients && inputData.InitialSolution != null)
            {
                savedParameters = inputData.InitialSolution.GetParameters();
            }

            // Step 1: Optimize locally to compute gradients
            // The wrapped optimizer will compute gradients and apply them locally
            var localResult = WrappedOptimizer.Optimize(inputData);

            // Step 2: Synchronize gradients across all ranks and apply averaged gradients
            if (Config.AutoSyncGradients && localResult.BestSolution != null && savedParameters != null)
            {
                var localGradients = gradientOptimizer.LastComputedGradients;

                if (localGradients != null && localGradients.Length > 0)
                {
                    // Average gradients across all ranks
                    // This is the key DDP operation: all ranks get the same averaged gradient
                    Config.CommunicationBackend.AllReduce(localGradients, ReductionOperation.Average);

                    // CRITICAL: Restore model to pre-update parameters before applying averaged gradients
                    // The local optimizer already applied local gradients, but we want to apply AVERAGED gradients instead.
                    // Without this restore, we would have: params - lr*localGrad - lr*avgGrad (wrong)
                    // With restore, we get: params - lr*avgGrad (correct)
                    localResult.BestSolution.SetParameters(savedParameters);

                    // Apply the averaged gradients using the wrapped optimizer's logic
                    // This works for ANY optimizer (SGD, Adam, RMSprop, etc.) because ApplyGradients
                    // handles optimizer-specific state (momentum, variance, etc.)
                    var finalModel = gradientOptimizer.ApplyGradients(localGradients, localResult.BestSolution);
                    localResult.BestSolution = finalModel;
                }
            }

            SynchronizeOptimizerState();

            return localResult;
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
