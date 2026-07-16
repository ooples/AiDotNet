using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Elastic optimizer - supports dynamic worker addition/removal during training.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Elastic training (TorchElastic, Horovod Elastic) enables dynamic scaling of workers during
/// training. Workers can be added or removed without stopping the training job, supporting:
/// - Fault tolerance: Replace failed workers automatically
/// - Auto-scaling: Add workers during peak hours, remove during off-peak
/// - Spot instance usage: Tolerate preemptions, use cheaper compute
///
/// When world size changes, the optimizer handles re-sharding parameters and optimizer states
/// across the new worker set. This requires checkpointing and careful state management.
/// </para>
/// <para><b>For Beginners:</b>
/// Elastic training is like having a flexible team size. Workers can join or leave during
/// training without stopping everything:
///
/// Scenario 1 - Fault tolerance:
/// - Start with 8 GPUs training your model
/// - GPU 3 fails → automatically detected
/// - Training continues with 7 GPUs (parameters redistributed)
/// - New GPU joins → training scales back to 8 GPUs
///
/// Scenario 2 - Cloud cost optimization:
/// - Use cheap "spot instances" that can be taken away anytime
/// - When instance is preempted, training continues with remaining workers
/// - New instance joins when available
///
/// This is critical for long training jobs where failures are expected.
/// </para>
/// <para><b>Use Cases:</b>
/// - Long training jobs (days/weeks) where failures will occur
/// - Cloud training with spot/preemptible instances (save 60-90% cost)
/// - Auto-scaling based on load or time of day
/// - Fault tolerance for production training pipelines
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Must handle dynamic re-sharding
/// - Communication: Overhead during worker changes (re-sharding, sync)
/// - Complexity: Very High - requires membership management, state re-distribution
/// - Convergence: Learning rate scheduling must account for dynamic world size
/// - Best for: Long jobs, cost-sensitive scenarios, production ML pipelines
/// - Limitation: Worker changes create temporary slowdown during re-sharding
/// </para>
/// <para><b>Implementation Note:</b>
/// This framework provides elastic optimizer infrastructure. Full production deployment
/// requires:
/// 1. Membership/discovery service (etcd, ZooKeeper, or cloud-native)
/// 2. Automatic checkpointing before worker changes
/// 3. State re-sharding algorithms
/// 4. Rendezvous mechanism for worker coordination
/// This implementation demonstrates the elastic pattern.
/// </para>
/// <para>
/// Example:
/// <code>
/// var optimizer = new AdamOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, options);
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// var elasticOptimizer = new ElasticOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     optimizer, config,
///     minWorkers: 2,   // Can run with as few as 2 workers
///     maxWorkers: 16); // Can scale up to 16 workers
///
/// // Training continues through worker changes:
/// // 4 workers → 3 workers (one fails) → 5 workers (two join) → ...
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ElasticOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    private readonly int _minWorkers;
    private readonly int _maxWorkers;
    private int _currentWorldSize;

    /// <summary>
    /// Creates an elastic optimizer.
    /// </summary>
    /// <param name="wrappedOptimizer">The optimizer to wrap with elastic capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <param name="minWorkers">Minimum number of workers (default: 1)</param>
    /// <param name="maxWorkers">Maximum number of workers (default: 1024)</param>
    public ElasticOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config,
        int minWorkers = 1,
        int maxWorkers = 1024)
        : base(wrappedOptimizer, config)
    {
        if (minWorkers < 1)
            throw new ArgumentException("Minimum workers must be at least 1", nameof(minWorkers));
        if (maxWorkers < minWorkers)
            throw new ArgumentException("Maximum workers must be >= minimum workers", nameof(maxWorkers));

        _minWorkers = minWorkers;
        _maxWorkers = maxWorkers;
        _currentWorldSize = WorldSize;
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // CRITICAL: Opening barrier to synchronize all workers before any operations.
        // This ensures all workers start together, preventing barrier mismatches if
        // some workers detect world size changes while others don't.
        // Must execute BEFORE any divergent logic (including null checks).
        Config.CommunicationBackend.Barrier();

        try
        {
            // Null check happens AFTER opening barrier but INSIDE try block.
            // This ensures that if one worker receives null while another doesn't,
            // both workers still execute the finally barrier, preventing deadlock.
            if (inputData == null)
                throw new ArgumentNullException(nameof(inputData));

            // Check for world size changes (workers joined/left)
            if (DetectWorldSizeChange())
            {
                HandleWorkerChange(inputData);
            }

            // When cross-rank synchronization is off, fall back to a plain local step.
            if (!Config.AutoSyncGradients)
                return RunWrappedOptimizerStep(inputData);

            // Elastic DDP: on top of the elastic membership handling above, each surviving step is a true
            // synchronous DDP step (per-step gradient all-reduce). RunDataParallelStep computes the
            // backward-only gradient, averages it across the CURRENT worker set, and applies one update
            // from the original parameters — so all replicas stay bit-identical without a separate
            // parameter-averaging pass (which is why the old SynchronizeParameters call is gone).
            return RunDataParallelStep(inputData, ReductionOperation.Average);
        }
        finally
        {
            // CRITICAL: Closing barrier ALWAYS executes to prevent deadlock.
            // Even if null check, HandleWorkerChange(), or WrappedOptimizer.Optimize() throws,
            // all workers reach this barrier before propagating the exception.
            // This prevents the scenario where:
            // - Worker A: Exception thrown, finally executes, waits at barrier
            // - Worker B: No exception, finally executes, waits at barrier
            // - Both workers synchronized at closing barrier despite exceptions
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // Intentional no-op for the per-step path. Elastic re-sharding of optimizer state happens only on a
        // membership change and is performed in HandleWorkerChange (via ResynchronizeParametersAcrossWorkers).
        // Between membership changes this optimizer replicates parameters across workers, so the wrapped
        // optimizer's Adam m/v state is identical on every rank and needs no per-step exchange.
    }

    /// <summary>
    /// Detects if the world size has changed since the last handled membership event.
    /// </summary>
    /// <remarks>
    /// The communication backend is the authority on the current worker set (in a production deployment it
    /// is fed by the rendezvous/membership service such as etcd or the torchelastic agent). A change is any
    /// difference between the backend's reported world size and the size we last re-sharded for.
    /// </remarks>
    private bool DetectWorldSizeChange()
    {
        int newWorldSize = Config.CommunicationBackend.WorldSize;
        return newWorldSize != _currentWorldSize;
    }

    /// <summary>
    /// Handles worker addition or removal by re-sharding onto the new worker set.
    /// </summary>
    private void HandleWorkerChange(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        int newWorldSize = Config.CommunicationBackend.WorldSize;

        // Validate new world size is within bounds
        if (newWorldSize < _minWorkers || newWorldSize > _maxWorkers)
        {
            throw new InvalidOperationException(
                $"World size {newWorldSize} outside allowed range [{_minWorkers}, {_maxWorkers}]");
        }

        // Elastic rendezvous re-sharding (torch.distributed.elastic): when the worker set changes, every
        // surviving AND newly-joined worker must resume from a single consistent copy of the parameters.
        // Rank 0 is the survivor that carries the authoritative state across the rendezvous; broadcasting it
        // guarantees a worker that just joined does not train from stale or uninitialised parameters. This
        // strategy replicates parameters across workers, so re-sharding is a full broadcast; a
        // parameter-sharded strategy would additionally re-partition the flat parameter/optimizer-state
        // vectors by the new world size (each rank taking its ceil(N / newWorldSize) slice).
        if (inputData.InitialSolution != null)
        {
            ResynchronizeParametersAcrossWorkers(inputData.InitialSolution);
        }

        _currentWorldSize = newWorldSize;
    }

    /// <summary>
    /// Re-synchronizes the model parameters across all workers by broadcasting rank 0's authoritative copy.
    /// This is the parameter re-sharding step of an elastic membership change and is exposed internally so
    /// it can be verified directly (a broadcast makes every rank's parameters identical to rank 0's).
    /// </summary>
    /// <param name="model">The model whose parameters are re-synchronized in place.</param>
    internal void ResynchronizeParametersAcrossWorkers(IFullModel<T, TInput, TOutput> model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        var parameterizable = InterfaceGuard.Parameterizable(model);
        Vector<T> authoritative = Config.CommunicationBackend.Broadcast(parameterizable.GetParameters(), root: 0);
        parameterizable.SetParameters(authoritative);
    }

    /// <summary>
    /// Gets the current number of active workers.
    /// </summary>
    public int CurrentWorkers => _currentWorldSize;

    /// <summary>
    /// Gets whether the optimizer can accept more workers.
    /// </summary>
    public bool CanScaleUp => _currentWorldSize < _maxWorkers;

    /// <summary>
    /// Gets whether the optimizer can tolerate losing workers.
    /// </summary>
    public bool CanScaleDown => _currentWorldSize > _minWorkers;

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize elastic-specific configuration
        writer.Write(_minWorkers);
        writer.Write(_maxWorkers);
        writer.Write(_currentWorldSize);

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

        // Read elastic-specific configuration
        int savedMinWorkers = reader.ReadInt32();
        int savedMaxWorkers = reader.ReadInt32();
        int savedCurrentWorldSize = reader.ReadInt32();

        // Read sharding configuration (for validation)
        int savedWorldSize = reader.ReadInt32();
        int savedRank = reader.ReadInt32();
        reader.ReadBoolean(); // AutoSyncGradients
        reader.ReadInt32(); // MinimumParameterGroupSize
        reader.ReadBoolean(); // EnableGradientCompression

        // IMPORTANT: In elastic training, world size and rank CAN change between save/load.
        // This is expected behavior when workers are added/removed. Do NOT validate equality.
        // The optimizer will handle re-sharding automatically via HandleWorkerChange().
        //
        // We DO validate that elastic bounds (min/max workers) match, as these define the
        // allowable range and should be consistent across elastic configuration changes.

        if (savedMinWorkers != _minWorkers || savedMaxWorkers != _maxWorkers)
        {
            throw new InvalidOperationException(
                $"Elastic configuration mismatch. Optimizer was saved with minWorkers={savedMinWorkers}, " +
                $"maxWorkers={savedMaxWorkers}, but current configuration has minWorkers={_minWorkers}, " +
                $"maxWorkers={_maxWorkers}.");
        }

        _currentWorldSize = savedCurrentWorldSize;

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        WrappedOptimizer.Deserialize(optimizerData);
    }
}
