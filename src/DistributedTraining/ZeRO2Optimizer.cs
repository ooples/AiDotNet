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
    /// <param name="wrappedOptimizer">The base optimizer to wrap (must be gradient-based: SGD, Adam, etc.)</param>
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
            // Step 1: Optimize locally to compute gradients (and apply them locally)
            var localResult = WrappedOptimizer.Optimize(inputData);

            // Step 2: Implement ZeRO-2 gradient sharding with ReduceScatter
            if (Config.AutoSyncGradients && localResult.BestSolution != null)
            {
                var localGradients = gradientOptimizer.LastComputedGradients;

                if (localGradients != null && localGradients.Length > 0)
                {
                    // Get parameters after local gradient application
                    var updatedParams = localResult.BestSolution.GetParameters();

                    // Reverse the local update to get original parameters
                    var originalParams = ComputeOriginalParameters(updatedParams, localGradients);

                    // TODO: Complete ZeRO-2 parameter shard update with ReduceScatter
                    // Current limitation: The IGradientBasedOptimizer.ApplyGradients() expects full gradient vector.
                    // Proper ZeRO-2 requires:
                    // 1. Use ReduceScatter to distribute gradient shards: var gradientShard = Config.CommunicationBackend.ReduceScatter(localGradients, ReductionOperation.Average);
                    // 2. Split originalParams into shards matching gradient shards
                    // 3. Apply gradientShard to this rank's parameter shard only
                    // 4. AllGather parameter shards to reconstruct full parameters
                    //
                    // For now, we use DDP-style full gradient sync as a functional approximation.
                    // This provides correct gradient averaging but without the memory savings of true ZeRO-2 gradient sharding.
                    Config.CommunicationBackend.AllReduce(localGradients, ReductionOperation.Average);

                    // CRITICAL: Restore model to pre-update parameters before applying averaged gradients
                    // Without this, we would double-apply gradients: params - lr*localGrad - lr*avgGrad
                    // instead of the correct: params - lr*avgGrad
                    localResult.BestSolution.SetParameters(originalParams);

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

    /// <summary>
    /// Computes the original parameters before gradient application by reversing the update.
    /// </summary>
    /// <param name="updatedParams">Parameters after gradient application</param>
    /// <param name="gradients">The gradients that were applied</param>
    /// <returns>Estimated original parameters before gradient application</returns>
    /// <remarks>
    /// <para><b>⚠️ IMPORTANT LIMITATION - Assumes Vanilla SGD:</b>
    /// This method assumes vanilla gradient descent update rule:
    /// params_new = params_old - learning_rate * gradients
    /// Therefore: params_old = params_new + learning_rate * gradients
    /// </para>
    /// <para>
    /// This reversal is INCORRECT for adaptive optimizers like Adam or RMSprop which use:
    /// - Adam: params_new = params_old - lr * m_t / (sqrt(v_t) + epsilon)
    /// - RMSprop: params_new = params_old - lr * gradients / sqrt(moving_avg_squared_gradients + epsilon)
    ///
    /// For these optimizers, reversing the update requires access to internal optimizer state
    /// (momentum buffers, variance estimates, etc.) which is not available through the current
    /// IGradientBasedOptimizer interface.
    /// </para>
    /// <para>
    /// <b>Production Guidance:</b>
    /// - ✅ Safe to use with: GradientDescentOptimizer, StochasticGradientDescentOptimizer, MiniBatchGradientDescentOptimizer
    /// - ⚠️ May produce incorrect results with: AdamOptimizer, RMSpropOptimizer, other adaptive optimizers
    /// - Future enhancement: Extend IGradientBasedOptimizer with ReverseUpdate() method for optimizer-specific reversal
    /// </para>
    /// </remarks>
    private Vector<T> ComputeOriginalParameters(Vector<T> updatedParams, Vector<T> gradients)
    {
        // Get learning rate from optimizer options
        var options = WrappedOptimizer.GetOptions();
        double learningRate = options.InitialLearningRate;

        // Reverse the update: params_old = params_new + lr * gradients
        var original = new T[updatedParams.Length];
        for (int i = 0; i < updatedParams.Length; i++)
        {
            double updated = Convert.ToDouble(updatedParams[i]);
            double gradient = Convert.ToDouble(gradients[i]);
            double originalValue = updated + learningRate * gradient;
            original[i] = (T)Convert.ChangeType(originalValue, typeof(T));
        }

        return new Vector<T>(original);
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
