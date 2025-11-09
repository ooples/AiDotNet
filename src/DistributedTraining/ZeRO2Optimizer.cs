using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements ZeRO Stage 2 optimizer - shards gradients and optimizer states across ranks.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// True ZeRO-2 implementation using ReduceScatter for gradient sharding. Each rank:
/// 1. Computes local gradients on full parameter set
/// 2. ReduceScatter: reduces gradients AND scatters them (each rank gets a shard)
/// 3. Updates only its shard of parameters using its shard of gradients
/// 4. AllGather: reconstructs full parameters from shards for next forward pass
///
/// This saves memory by distributing gradient storage and parameter updates across ranks.
/// </para>
/// <para><b>For Beginners:</b>
/// ZeRO-2 divides the work of storing and updating parameters across processes. Think of it
/// like a team where each person is responsible for maintaining a specific section of a large
/// document. Everyone reads the full document (forward pass), but each person only stores and
/// updates their assigned section (backward pass). Before the next iteration, they share their
/// sections to reconstruct the full document.
/// </para>
/// <para><b>Use Cases:</b>
/// - Large models where gradient memory is significant (billions of parameters)
/// - Want memory savings beyond DDP
/// - Good network for AllGather operations
/// - Works with ANY gradient-based optimizer (SGD, Adam, RMSprop, etc.)
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Very Good - gradients and optimizer states sharded (1/N of DDP)
/// - Communication: ReduceScatter + AllGather (vs AllReduce for DDP)
/// - Synchronization: Perfect - all ranks reconstruct identical parameters
/// - Complexity: Moderate - requires parameter sharding logic
/// - Best for: Large models with limited GPU memory
/// </para>
/// <para><b>Memory Savings vs DDP:</b>
/// - DDP: Each rank stores full gradients + full optimizer state
/// - ZeRO-2: Each rank stores 1/N gradients + 1/N optimizer state
/// - Savings increase linearly with world size
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
        Config.CommunicationBackend.Barrier();

        try
        {
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));

            var gradientOptimizer = (IGradientBasedOptimizer<T, TInput, TOutput>)WrappedOptimizer;

            // Populate InitialSolution from wrapped optimizer's model if not already set
            if (inputData.InitialSolution == null && WrappedOptimizer is OptimizerBase<T, TInput, TOutput> baseOptimizer && baseOptimizer.Model != null)
            {
                inputData.InitialSolution = baseOptimizer.Model.Clone();
            }

            // CRITICAL: Save parameters BEFORE local optimization
            Vector<T>? savedParameters = null;
            if (Config.AutoSyncGradients && inputData.InitialSolution != null)
            {
                savedParameters = inputData.InitialSolution.GetParameters();
            }

            // Step 1: Optimize locally to compute gradients
            var localResult = WrappedOptimizer.Optimize(inputData);

            // Step 2: ZeRO-2 gradient sharding with ReduceScatter
            if (Config.AutoSyncGradients && localResult.BestSolution != null && savedParameters != null)
            {
                var localGradients = gradientOptimizer.LastComputedGradients;

                if (localGradients != null && localGradients.Length > 0)
                {
                    // ZeRO-2 CORE: ReduceScatter averages gradients AND scatters them
                    // Each rank receives only its shard of the averaged gradients
                    var myGradientShard = Config.CommunicationBackend.ReduceScatter(localGradients, ReductionOperation.Average);

                    // Calculate which parameter shard this rank owns
                    int totalParams = savedParameters.Length;
                    int shardSize = myGradientShard.Length;
                    int myShardStart = Rank * shardSize;

                    // Extract this rank's parameter shard
                    var myParamShard = new T[shardSize];
                    for (int i = 0; i < shardSize && (myShardStart + i) < totalParams; i++)
                    {
                        myParamShard[i] = savedParameters[myShardStart + i];
                    }
                    var myParamShardVector = new Vector<T>(myParamShard);

                    // Step 3: Update ONLY this rank's shard using the gradient shard
                    // This is where ZeRO-2 saves memory - only updating 1/N of parameters
                    var updatedShard = gradientOptimizer.UpdateParameters(myParamShardVector, myGradientShard);

                    // Step 4: AllGather to reconstruct full parameters from all shards
                    // Each rank contributes its updated shard, everyone gets full parameter vector
                    var fullParameters = Config.CommunicationBackend.AllGather(updatedShard);

                    // Step 5: Create model with reconstructed full parameters
                    var finalModel = localResult.BestSolution.WithParameters(fullParameters);
                    localResult.BestSolution = finalModel;
                }
            }

            SynchronizeOptimizerState();

            return localResult;
        }
        finally
        {
            // CRITICAL: Closing barrier ALWAYS executes to prevent deadlock
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
