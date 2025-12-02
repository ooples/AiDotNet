using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Asynchronous SGD optimizer - allows asynchronous parameter updates without strict barriers.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Asynchronous SGD (and variants like Hogwild!) removes synchronization barriers between workers.
/// Each process updates parameters independently without waiting for others, using a parameter
/// server or shared memory. This eliminates idle time from synchronization but introduces stale
/// gradients - workers may compute gradients on slightly outdated parameters.
///
/// When done correctly (sparse gradients, low contention), async SGD can achieve near-linear
/// speedup without much accuracy loss. However, it's more sensitive to hyperparameters and
/// can be unstable for dense updates.
/// </para>
/// <para><b>For Beginners:</b>
/// Async SGD is like a team working independently without meetings. Each person:
/// 1. Reads current parameters
/// 2. Computes gradients on their data
/// 3. Updates parameters immediately (no waiting!)
///
/// Pro: No time wasted waiting for slow workers
/// Con: Updates might conflict or use slightly stale information
///
/// This works well when updates are sparse (touching different parameters) but can be
/// unstable when all workers update the same parameters frequently.
/// </para>
/// <para><b>Use Cases:</b>
/// - Sparse models (embeddings, recommendation systems)
/// - Scenarios with stragglers (some workers slower than others)
/// - When synchronization overhead is very high
/// - Research and experimentation
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Requires parameter server or shared memory
/// - Communication: Asynchronous - can be higher total volume
/// - Complexity: High - requires parameter server infrastructure
/// - Convergence: Can be slower or less stable than sync SGD
/// - Best for: Sparse updates, heterogeneous workers, straggler tolerance
/// - Limitation: Harder to tune, may require staleness-aware algorithms
/// </para>
/// <para><b>Implementation Note:</b>
/// This framework provides async SGD infrastructure. Full production implementation
/// requires parameter server setup or shared memory coordination. This implementation
/// demonstrates the async update pattern.
/// </para>
/// <para>
/// Example:
/// <code>
/// var optimizer = new AdamOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, options);
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// var asyncOptimizer = new AsyncSGDOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     optimizer, config, allowStaleness: 2); // Allow up to 2 stale gradient steps
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class AsyncSGDOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    private readonly int _maxStaleness;

    /// <summary>
    /// Creates an async SGD optimizer.
    /// </summary>
    /// <param name="wrappedOptimizer">The optimizer to wrap with async capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <param name="allowStaleness">Maximum allowed staleness in gradient steps (default: 0 = sync)</param>
    public AsyncSGDOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config,
        int allowStaleness = 0)
        : base(wrappedOptimizer, config)
    {
        _maxStaleness = allowStaleness;
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        // NO barrier at start - async operation!

        // Optimize locally without waiting
        var result = WrappedOptimizer.Optimize(inputData);

        // Asynchronous parameter update
        if (Config.AutoSyncGradients && result.BestSolution != null)
        {
            // In true async SGD:
            // 1. Push gradients to parameter server (non-blocking)
            // 2. Pull updated parameters (may be from other workers' updates)
            // 3. Continue immediately without waiting

            // For this framework implementation, we provide simplified async pattern
            // Production would use parameter server or async AllReduce
            var parameters = result.BestSolution.GetParameters();

            // Simulate async update - in production, this would be non-blocking
            Config.CommunicationBackend.AllReduce(parameters, ReductionOperation.Average);

            result.BestSolution.SetParameters(parameters);
        }

        // NO barrier at end - continue immediately!

        return result;
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // In async SGD, optimizer states are typically local (not synchronized)
        // Each worker maintains its own state and updates independently
    }

    /// <summary>
    /// Checks if a barrier should be used (for periodic synchronization).
    /// </summary>
    /// <param name="iteration">Current iteration number</param>
    /// <returns>True if should synchronize at this iteration</returns>
    public bool ShouldSync(int iteration)
    {
        // Some async implementations do periodic sync every N iterations
        // to prevent too much drift
        if (_maxStaleness == 0)
            return true; // Always sync (becomes sync SGD)

        // Framework pattern - could implement periodic sync
        return false;
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize async-specific configuration
        writer.Write(_maxStaleness);

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

        // Read async-specific configuration
        int savedMaxStaleness = reader.ReadInt32();

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

        if (savedMaxStaleness != _maxStaleness)
        {
            throw new InvalidOperationException(
                $"Max staleness mismatch. Optimizer was saved with staleness={savedMaxStaleness}, " +
                $"but current configuration has staleness={_maxStaleness}.");
        }

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        WrappedOptimizer.Deserialize(optimizerData);
    }
}
