using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements FSDP (Fully Sharded Data Parallel) optimizer wrapper that coordinates optimization across multiple processes.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// FSDP optimizer works in conjunction with FSDPModel to provide full sharding of optimizer states.
/// This means momentum buffers, variance estimates, and all other optimizer-specific state are sharded
/// across processes, minimizing memory usage while maintaining training effectiveness.
/// </para>
/// <para><b>For Beginners:</b>
/// This class wraps any existing optimizer (like Adam, SGD, etc.) and makes it work with FSDP strategy
/// across multiple GPUs or machines. It automatically handles:
/// - Synchronizing gradients across all processes
/// - Sharding optimizer states (momentum, variance) to save memory
/// - Coordinating parameter updates
/// - Ensuring all processes stay in sync
/// </para>
/// <para>
/// Think of it like a team of coaches working together - each coach has their own expertise
/// (the wrapped optimizer), but they share only the essential information and keep their detailed
/// notes (optimizer states) private to save space.
/// </para>
/// <para><b>Use Cases:</b>
/// - Training very large models with optimizers that have significant state (Adam, RMSprop)
/// - Maximizing memory efficiency when using stateful optimizers
/// - Scaling to hundreds or thousands of GPUs
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent - shards optimizer states across processes
/// - Communication: Moderate - syncs gradients and occasional state synchronization
/// - Complexity: Moderate - automatic state sharding
/// - Best for: Large models with stateful optimizers (Adam, RMSprop, etc.)
/// </para>
/// <para>
/// Example:
/// <code>
/// // Original optimizer
/// var optimizer = new AdamOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, options);
///
/// // Wrap it for FSDP distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var fsdpOptimizer = new FSDPOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     optimizer, config);
///
/// // Now optimize as usual - FSDP magic happens automatically!
/// var result = fsdpOptimizer.Optimize(inputData);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class FSDPOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a new FSDP optimizer wrapping an existing optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This constructor takes your existing optimizer and makes it distributed using FSDP strategy.
    /// You provide:
    /// 1. The optimizer you want to make distributed
    /// 2. A configuration that tells us how to do the distribution
    /// </para>
    /// <para>
    /// The optimizer will automatically synchronize across all processes during optimization
    /// and shard optimizer states to minimize memory usage.
    /// </para>
    /// </remarks>
    /// <param name="wrappedOptimizer">The optimizer to wrap with FSDP capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if optimizer or config is null</exception>
    public FSDPOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
        : base(wrappedOptimizer, config)
    {
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Validate BEFORE touching any collective: a null input is a programming error that must fail this
        // rank immediately, not after it has entered a barrier/collective that peers are blocked on.
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        // Local-step mode runs ENTIRELY OUTSIDE the collective sequence (no barrier, no ReduceScatter/
        // AllGather), so a rank configured for local steps can never deadlock against a rank in the sharded
        // path. This must precede the opening barrier.
        if (!Config.AutoSyncGradients)
            return RunWrappedOptimizerStep(inputData);

        // Synchronized FSDP == ZeRO Stage-3 (Zhao et al. 2023; Rajbhandari et al. 2020): gradients AND
        // optimizer state are partitioned. RunShardedZeroStep(shardGradients:true) ReduceScatters the
        // gradients (each rank holds only its averaged shard), updates ONLY this rank's parameter shard with
        // the wrapped optimizer — so its Adam m/v state is sized to the shard (real state partitioning) —
        // and AllGathers the updated shards. Stage-3 PARAMETER residency is provided by the Stage-3 sharded
        // layer; FSDPModel handles the model-side param sharding. CpuOffload flags are honored in the step.
        //
        // This is SPMD: every rank MUST reach the same collective sequence. The opening barrier lines the
        // ranks up; the closing barrier is a clean STEP-BOUNDARY synchronization so no rank races ahead to
        // the next step (or, in a shared-runtime harness, into another rank's teardown) while a peer is
        // still finishing this step's AllGather.
        //
        // What the closing barrier is NOT: fault tolerance. It is deliberately placed on the SUCCESS path
        // (not in a try/finally) because a finally-barrier would be FALSE SAFETY — if one rank fails inside
        // ReduceScatter/AllGather while a peer is blocked in that same collective, a trailing Barrier() (a
        // DIFFERENT collective) cannot rescue the mismatched sequence. Recovering from a divergent
        // mid-collective failure requires a backend-level abort/cancellation, which ICommunicationBackend
        // does not expose; a symmetric failure (all ranks throw) needs no barrier and the next step's
        // opening barrier re-synchronizes survivors.
        Config.CommunicationBackend.Barrier();
        var result = RunShardedZeroStep(inputData, shardGradients: true);
        Config.CommunicationBackend.Barrier();
        return result;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Not supported for FSDP: optimizer state is PARTITIONED, not replicated. State partitioning is
    /// achieved intrinsically inside <c>RunShardedZeroStep</c> — the wrapped optimizer's
    /// <c>UpdateParameters</c> is called with ONLY this rank's gradient/parameter shard, so its Adam m/v
    /// state is allocated and advanced solely for this rank's parameters. No rank holds another rank's
    /// state, so there is no cross-rank state to synchronize; calling this method would be a logic error,
    /// and returning silently could mislead a caller into believing a synchronization occurred.
    /// </remarks>
    public override void SynchronizeOptimizerState()
        => throw new NotSupportedException(
            "FSDP optimizer state is partitioned across ranks during Optimize (each rank owns only its " +
            "shard's Adam state); there is no replicated state to synchronize, so explicit state " +
            "synchronization is neither required nor supported.");

    /// <inheritdoc/>
    public override bool ShouldEarlyStop()
    {
        // Delegate to wrapped optimizer
        bool localDecision = WrappedOptimizer.ShouldEarlyStop();

        // In distributed training, we need consensus on early stopping
        // Using Max operation: if ANY process wants to stop, all processes stop
        // This prevents stragglers and ensures synchronized termination

        // Create a vector with the local decision (1 for stop, 0 for continue)
        var decision = new Vector<T>(new[] { localDecision ? MathHelper.GetNumericOperations<T>().One : MathHelper.GetNumericOperations<T>().Zero });

        // Get the maximum across all processes
        // If any process returns 1 (stop), the max will be 1
        Config.CommunicationBackend.AllReduce(decision, ReductionOperation.Max);

        // Check if the result indicates stopping
        var numOps = MathHelper.GetNumericOperations<T>();
        return !numOps.Equals(decision[0], numOps.Zero);
    }

    /// <inheritdoc/>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return WrappedOptimizer.GetOptions();
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

        // Validate rank matches - different rank could indicate configuration mismatch
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

    // SaveModel/LoadModel are inherited from ShardedOptimizerBase, which now writes one .rank{Rank}
    // checkpoint file per rank — exactly the per-shard round-trip FSDP requires (this override used to
    // provide it, but the behavior is now the shared base default for all sharded strategies).
}