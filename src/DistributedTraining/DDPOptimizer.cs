using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements true DDP (Distributed Data Parallel) optimizer - industry-standard gradient averaging.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// True DDP is the industry-standard distributed training approach used by PyTorch, TensorFlow, and JAX.
/// After computing gradients on local data, gradients are averaged across all workers using AllReduce,
/// then the averaged gradients are applied to update model parameters. This ensures all workers
/// stay perfectly synchronized with identical parameter updates at every step.
/// </para>
/// <para><b>For Beginners:</b>
/// DDP works by having each worker compute gradients on their local batch of data, then averaging
/// those gradients across all workers before updating the model. It's like a study group where everyone
/// works on different practice problems, shares their solutions, averages the feedback, and everyone
/// applies the same averaged correction to their understanding.
/// </para>
/// <para><b>Key Difference from Local SGD:</b>
/// - **True DDP (this class)**: Compute gradients → Average GRADIENTS → Apply averaged gradients
/// - **Local SGD**: Optimize locally → Average PARAMETERS after multiple steps
///
/// DDP maintains tighter synchronization but requires more frequent communication.
/// </para>
/// <para><b>How It Works:</b>
/// 1. Each worker computes gradients on local data batch
/// 2. Gradients are synchronized via AllReduce (averaging across all workers)
/// 3. Each worker applies the same averaged gradients to their model
/// 4. All workers now have identical parameters
/// 5. Repeat for next iteration
/// </para>
/// <para><b>Use Cases:</b>
/// - Standard multi-GPU distributed training (PyTorch DDP, TensorFlow MirroredStrategy)
/// - Fast interconnects (NVLink, InfiniBand) where communication is cheap
/// - Training where tight synchronization is critical
/// - Works with any optimizer (SGD, Adam, RMSprop, etc.)
/// - Default choice for distributed training with good network
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Each process stores full model and optimizer state
/// - Communication: Moderate - gradients synchronized every step (can use gradient compression)
/// - Synchronization: Perfect - all workers always have identical parameters
/// - Convergence: Identical to single-GPU training (mathematically equivalent)
/// - Complexity: Low - straightforward gradient averaging
/// - Best for: Fast networks, standard distributed training scenarios
/// </para>
/// <para><b>Production Implementation:</b>
/// This implementation uses the gradient access infrastructure (LastComputedGradients, ApplyGradients)
/// to properly average gradients before parameter updates. It reverses local gradient applications
/// to recover original parameters, applies averaged gradients, ensuring true DDP semantics.
/// </para>
/// <para><b>Industry Standard:</b>
/// This implementation matches PyTorch's DistributedDataParallel, TensorFlow's MirroredStrategy,
/// and JAX's pmap with gradient averaging. It is the gold standard for distributed training.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class DDPOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a true DDP optimizer that averages gradients across workers.
    /// </summary>
    /// <param name="wrappedOptimizer">The base optimizer to wrap (must be gradient-based: SGD, Adam, etc.)</param>
    /// <param name="config">Configuration for distributed training communication</param>
    /// <exception cref="ArgumentException">If wrapped optimizer is not gradient-based</exception>
    public DDPOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
        : base(wrappedOptimizer, config)
    {
        // Verify wrapped optimizer supports gradient operations
        if (wrappedOptimizer is not IGradientBasedOptimizer<T, TInput, TOutput>)
        {
            throw new ArgumentException(
                $"DDP requires a gradient-based optimizer, but received {wrappedOptimizer.GetType().Name}. " +
                "Use gradient-based optimizers like SGD, Adam, RMSprop, etc.",
                nameof(wrappedOptimizer));
        }
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        var gradientOptimizer = (IGradientBasedOptimizer<T, TInput, TOutput>)WrappedOptimizer;

        // Populate InitialSolution from wrapped optimizer's model if not already set
        if (inputData.InitialSolution == null && WrappedOptimizer is OptimizerBase<T, TInput, TOutput> baseOptimizer && baseOptimizer.Model != null)
        {
            inputData.InitialSolution = baseOptimizer.Model.Clone();
        }

        // Barrier to ensure all processes start together
        Config.CommunicationBackend.Barrier();

        try
        {
            // CRITICAL: Save parameters BEFORE local optimization
            // This allows us to apply averaged gradients from the correct starting point
            Vector<T>? savedParameters = null;
            if (Config.AutoSyncGradients && inputData.InitialSolution != null)
            {
                savedParameters = inputData.InitialSolution.GetParameters();
            }

            // Step 1: Optimize locally to compute gradients (and apply them locally)
            var localResult = WrappedOptimizer.Optimize(inputData);

            // Step 2: Synchronize gradients across all workers
            if (Config.AutoSyncGradients && localResult.BestSolution != null && savedParameters != null)
            {
                var localGradients = gradientOptimizer.LastComputedGradients;

                if (localGradients != null && localGradients.Length > 0)
                {
                    // Average gradients across all workers (true DDP)
                    Config.CommunicationBackend.AllReduce(localGradients, ReductionOperation.Average);

                    // Apply averaged gradients using the safe 3-parameter overload
                    // This explicitly passes savedParameters (pre-update state) to prevent double-stepping
                    // Works correctly with ANY optimizer (SGD, Adam, RMSprop, etc.) because we're starting
                    // from the original parameters, not the locally-updated ones
                    var finalModel = gradientOptimizer.ApplyGradients(savedParameters, localGradients, localResult.BestSolution);

                    // Update result with model using averaged gradients
                    localResult.BestSolution = finalModel;
                }
            }

            return localResult;
        }
        finally
        {
            // Barrier to ensure all processes finish together (always runs even if exception thrown)
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // In DDP, optimizer states are not sharded
        // Each process maintains its own full optimizer state
        // No synchronization needed for standard DDP
        // (Advanced: some implementations sync optimizer momentum/variance for better convergence)
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
