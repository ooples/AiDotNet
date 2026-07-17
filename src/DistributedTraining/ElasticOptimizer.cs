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

            // Check for world size changes (workers joined/left). CRITICAL: the decision must be COLLECTIVE
            // — a rank-local DetectWorldSizeChange() would let some ranks enter HandleWorkerChange's
            // collectives while others go straight to the gradient reduce, producing a divergent collective
            // sequence (deadlock/corruption). AllReduce(Max) over a 1/0 flag makes every rank agree that a
            // membership change occurred before any rank branches into the rendezvous.
            var changed = new Vector<T>(new[] { DetectWorldSizeChange() ? NumOps.One : NumOps.Zero });
            Config.CommunicationBackend.AllReduce(changed, ReductionOperation.Max);
            if (!NumOps.Equals(changed[0], NumOps.Zero))
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
    /// <remarks>
    /// Not supported as a standalone call. Between membership changes this optimizer replicates parameters
    /// and the wrapped optimizer's Adam m/v state is identical on every rank (no per-step exchange needed);
    /// the only state movement is the parameter broadcast in HandleWorkerChange on a membership change.
    /// A general "synchronize optimizer state" — including transferring serialized Adam state to a newly
    /// joined worker — is a full elastic state-transfer protocol that this optimizer does not implement, so
    /// it fails fast here rather than silently reporting success.
    /// </remarks>
    public override void SynchronizeOptimizerState()
        => throw new NotSupportedException(
            "ElasticOptimizer does not support standalone optimizer-state synchronization. Parameter " +
            "re-sync happens in HandleWorkerChange on a membership change; transferring serialized optimizer " +
            "(Adam m/v) state to newly joined workers is not implemented.");

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
        //
        // CRITICAL: the broadcast is a COLLECTIVE — it must NOT be gated on rank-local InitialSolution, or a
        // rank that happened to receive a model would enter Broadcast while a rank that did not would skip it
        // and race ahead to the closing barrier, deadlocking the worker set. Resolve a model on every rank
        // and require ALL ranks to have one (an AllReduce(Min) over a 1/0 availability flag) BEFORE any rank
        // broadcasts. A missing model on any rank makes every rank throw identically (no divergence).
        var model = inputData.InitialSolution
            ?? (WrappedOptimizer as OptimizerBase<T, TInput, TOutput>)?.Model;
        var available = new Vector<T>(new[] { model is null ? NumOps.Zero : NumOps.One });
        Config.CommunicationBackend.AllReduce(available, ReductionOperation.Min);
        if (model is null || NumOps.Equals(available[0], NumOps.Zero))
        {
            throw new InvalidOperationException(
                "Elastic rendezvous requires every worker to provide a model (OptimizationInputData.InitialSolution " +
                "or a wrapped OptimizerBase whose Model is set) so the authoritative parameters can be broadcast.");
        }

        ResynchronizeParametersAcrossWorkers(model);
        ResynchronizeOptimizerStateAcrossWorkers();

        _currentWorldSize = newWorldSize;
    }

    /// <summary>
    /// Transfers the wrapped optimizer's serialized state (e.g. Adam's m/v moment buffers) from rank 0 (the
    /// designated survivor) to every worker, so a newly-joined worker resumes with the authoritative
    /// optimizer state and not just the parameters. The serialized bytes are carried over the Vector-typed
    /// communication backend as one element per byte (a length header is broadcast first). This is a
    /// COLLECTIVE — every rank must call it — and is invoked from HandleWorkerChange after the parameter
    /// re-sync. (Full survivor election across an arbitrary surviving set is a separate torchelastic-scale
    /// feature; rank 0 is the designated survivor here.)
    /// </summary>
    internal void ResynchronizeOptimizerStateAcrossWorkers()
    {
        var backend = Config.CommunicationBackend;

        // Rank 0 (survivor) provides the authoritative serialized optimizer state; other ranks contribute a
        // placeholder that Broadcast overwrites. Broadcast the length first so non-root ranks size the buffer.
        byte[] rootBytes = Rank == 0 ? WrappedOptimizer.Serialize() : System.Array.Empty<byte>();
        var header = backend.Broadcast(new Vector<T>(new[] { NumOps.FromDouble(rootBytes.Length) }), root: 0);
        int length = NumOps.ToInt32(header[0]);
        if (length <= 0)
            return; // optimizer has no serialized state to transfer

        var payload = new T[length];
        if (Rank == 0)
            for (int i = 0; i < length; i++) payload[i] = NumOps.FromDouble(rootBytes[i]);
        var received = backend.Broadcast(new Vector<T>(payload), root: 0);

        var stateBytes = new byte[length];
        for (int i = 0; i < length; i++) stateBytes[i] = (byte)(NumOps.ToInt32(received[i]) & 0xFF);
        WrappedOptimizer.Deserialize(stateBytes);
    }

    /// <summary>Test hook: returns the wrapped optimizer's serialized state so an invariant can verify the
    /// elastic optimizer-state transfer made every rank's state identical to rank 0's.</summary>
    internal byte[] SerializeWrappedOptimizerForTest() => WrappedOptimizer.Serialize();

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
