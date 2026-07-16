using AiDotNet.Helpers;
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
    /// Maximum number of steps a worker may run ahead of the slowest worker before it must synchronize
    /// (the Stale-Synchronous Parallel staleness bound, Ho et al. 2013). 0 means synchronize every step,
    /// which reduces async SGD exactly to synchronous gradient SGD.
    /// </summary>
    public int MaxStaleness => _maxStaleness;

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
        if (allowStaleness < 0)
            throw new ArgumentOutOfRangeException(nameof(allowStaleness), "Staleness bound must be non-negative.");

        _maxStaleness = allowStaleness;
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        // NO barrier — async SGD deliberately does not synchronize workers (its defining property).

        // When cross-rank synchronization is off, fall back to a plain local step.
        if (!Config.AutoSyncGradients)
            return RunWrappedOptimizerStep(inputData);

        // Downpour / parameter-server async SGD (Dean et al. 2012; Stale-Synchronous Parallel, Ho et al.
        // 2013): each worker computes a GRADIENT and pushes it to the server, which applies the combined
        // update — gradient-based, NOT parameter averaging (parameter averaging is a different algorithm:
        // EASGD / model averaging). RunDataParallelStep does exactly the per-step version: backward-only
        // gradient, combine, single update from the original parameters. The asynchrony and the up-to-
        // MaxStaleness stale reads are a property of the RUNTIME; the synchronous InMemory backend enforces
        // staleness 0, under which async SGD reduces exactly to synchronous gradient SGD — the limit
        // implemented here (see ShouldSync for the bounded-staleness barrier a real async runtime applies).
        return RunDataParallelStep(inputData, ReductionOperation.Average);
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
        if (iteration < 0)
            throw new ArgumentOutOfRangeException(nameof(iteration), "Iteration must be non-negative.");

        // Stale-Synchronous Parallel bounded-staleness barrier (Ho et al. 2013): a worker may run ahead by
        // at most _maxStaleness steps before it MUST synchronize, so no replica ever reads parameters more
        // than _maxStaleness steps stale. staleness 0 => synchronize every step (reduces to synchronous
        // SGD). staleness s>0 => synchronize once every (s+1) steps, which bounds the maximum drift to s.
        if (_maxStaleness == 0)
            return true;

        return (iteration % (_maxStaleness + 1)) == 0;
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
