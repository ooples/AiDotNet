using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Validation;


namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides base implementation for distributed optimizers with parameter sharding.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for all sharded optimizers,
/// including optimizer wrapping, parameter synchronization, consensus-based early stopping,
/// and serialization. Derived classes can customize the optimization strategy, implement
/// different sharding approaches (FSDP, ZeRO, etc.), or add optimizer-specific features.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all distributed optimizers build upon.
///
/// Think of this as a template for coordinating optimization across multiple computers or GPUs.
/// It handles common tasks like:
/// - Wrapping regular optimizers to work in distributed mode
/// - Syncing parameters across all processes after updates
/// - Making sure all processes agree on when to stop training
/// - Saving and loading distributed optimizer state
///
/// Specific types of distributed optimizers (like data-parallel or ZeRO) inherit from
/// this and add their own strategies. This prevents code duplication and ensures all
/// distributed optimizers work consistently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public abstract class ShardedOptimizerBase<T, TInput, TOutput> : IShardedOptimizer<T, TInput, TOutput>, IModelShape
{
    /// <summary>
    /// Provides numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The wrapped optimizer that this sharded optimizer delegates to.
    /// </summary>
    private readonly IOptimizer<T, TInput, TOutput> _wrappedOptimizer;

    /// <summary>
    /// The sharding configuration containing communication backend and settings.
    /// </summary>
    protected readonly IShardingConfiguration<T> Config;

    /// <inheritdoc/>
    public IOptimizer<T, TInput, TOutput> WrappedOptimizer => _wrappedOptimizer;

    /// <summary>
    /// Protected access to wrapped optimizer for derived classes.
    /// </summary>
    protected IOptimizer<T, TInput, TOutput> WrappedOptimizerInternal => _wrappedOptimizer;

    /// <inheritdoc/>
    public int Rank => Config.CommunicationBackend.Rank;

    /// <inheritdoc/>
    public int WorldSize => Config.CommunicationBackend.WorldSize;

    /// <inheritdoc/>
    public IShardingConfiguration<T> ShardingConfiguration => Config;

    /// <summary>
    /// Initializes a new instance of the ShardedOptimizerBase class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor wraps an existing optimizer with distributed training capabilities.
    /// It initializes the communication backend if needed and prepares for distributed optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor takes your regular optimizer and makes it distributed.
    ///
    /// You provide:
    /// 1. The optimizer you want to distribute (like Adam, SGD, etc.)
    /// 2. Configuration that tells us how to distribute it
    ///
    /// The constructor automatically:
    /// - Sets up communication if not already done
    /// - Prepares the optimizer for coordinated training
    /// - Ensures all processes can work together
    /// </para>
    /// </remarks>
    /// <param name="wrappedOptimizer">The optimizer to wrap with distributed capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if optimizer or config is null</exception>
    protected ShardedOptimizerBase(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
    {
        Guard.NotNull(wrappedOptimizer);
        _wrappedOptimizer = wrappedOptimizer;
        Guard.NotNull(config);
        Config = config;
        NumOps = MathHelper.GetNumericOperations<T>();

        // Initialize communication backend if needed
        if (!Config.CommunicationBackend.IsInitialized)
        {
            Config.CommunicationBackend.Initialize();
        }
    }

    /// <inheritdoc/>
    public abstract OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData);

    /// <inheritdoc/>
    public abstract void SynchronizeOptimizerState();

    /// <summary>
    /// Runs the wrapped optimizer's Optimize step with the CPU-offload contract
    /// applied from <see cref="IShardingConfiguration{T}.CpuOffloadOptimizer"/>. When
    /// the flag is set AND the process is currently running on a non-CPU engine
    /// (typically <c>DirectGpuTensorEngine</c>), the engine is temporarily swapped
    /// to <see cref="CpuEngine"/> for the duration of the wrapped step so that
    /// every op inside — Adam's <c>m_t = β1·m_{t-1} + (1-β1)·g</c>, the bias
    /// correction, the parameter update — runs on the CPU and the m/v state
    /// tensors are never uploaded to GPU VRAM. This is the standard
    /// DeepSpeed ZeRO-Offload contract for Stage-1: the forward + backward
    /// stay on GPU (their tensors are the caller's problem), but the update
    /// step and the persistent optimizer state live in CPU RAM.
    ///
    /// <para>Also, when <see cref="IShardingConfiguration{T}.CpuOffloadGradients"/>
    /// is set, we materialize (download) the current gradient tensors before
    /// entering the wrapped step so the CPU-side Adam step reads real values
    /// rather than empty deferred arrays.</para>
    ///
    /// <para>Callers: every derived optimizer's <c>Optimize</c> override should
    /// invoke <c>WrappedOptimizer.Optimize(inputData)</c> through this helper
    /// (instead of directly) so the offload contract applies uniformly across
    /// FSDP, ZeRO1/2/3, DDP, and PipelineParallel strategies.</para>
    /// </summary>
    /// <param name="inputData">The input data forwarded to the wrapped optimizer.</param>
    /// <returns>The wrapped optimizer's OptimizationResult.</returns>
    protected OptimizationResult<T, TInput, TOutput> RunWrappedOptimizerStep(
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        using (BeginCpuOffloadScope())
        {
            return WrappedOptimizer.Optimize(inputData);
        }
    }

    /// <summary>
    /// Enters the CPU-offload engine scope described in
    /// <see cref="RunWrappedOptimizerStep"/>. Exposed as a separate helper so
    /// specialized derived optimizers (e.g. asynchronous SGD, gradient
    /// compression) that don't call <c>WrappedOptimizer.Optimize</c> directly
    /// can still opt into the same contract by wrapping their own step
    /// inline: <c>using var _ = BeginCpuOffloadScope();</c>. The returned
    /// disposable is a lightweight no-op when the flag is off or the current
    /// engine is already CPU-based.
    /// </summary>
    protected IDisposable BeginCpuOffloadScope()
    {
        return CpuOffloadScope.Enter(Config);
    }

    /// <summary>
    /// Executes one paper-faithful ZeRO sharded optimizer step (Rajbhandari et al. 2020, "ZeRO:
    /// Memory Optimizations Toward Training Trillion Parameter Models"; CPU offload per Ren et al.
    /// 2021, "ZeRO-Offload: Democratizing Billion-Scale Model Training"):
    /// <list type="number">
    /// <item><b>Backward only</b> — compute local gradients via
    /// <see cref="IGradientComputable{T,TInput,TOutput}.ComputeGradients"/>, WITHOUT running the wrapped
    /// optimizer's full <c>Optimize</c>, so its Adam m/v state is never advanced on the full parameter
    /// vector (that would defeat state partitioning and double-step the update).</item>
    /// <item><b>Reduce</b> the gradients across ranks — <c>AllReduce</c> for ZeRO-1 (gradients stay
    /// replicated), or <c>ReduceScatter</c> for ZeRO-2/3 (each rank holds only its averaged shard).</item>
    /// <item><b>Sharded update</b> — update ONLY this rank's parameter shard with the wrapped optimizer.
    /// Because the wrapped optimizer only ever sees this shard, its persistent state (Adam's m/v) is
    /// sized to the shard: this IS optimizer-state partitioning. CPU offload (the engine swap) is scoped
    /// to THIS update alone — forward/backward already ran on the accelerator; only the optimizer state
    /// and parameter update move to CPU.</item>
    /// <item><b>AllGather</b> the updated shards to reconstruct the full parameter vector, write it back
    /// to the model, then drop any GPU-cached parameters for the next forward.</item>
    /// </list>
    /// The wrapped optimizer's <c>UpdateParameters(shard, gradShard)</c> is the SINGLE optimizer step —
    /// no full-vector update precedes it — so there is no double-stepping and the m/v state stays sized
    /// to the shard across steps.
    /// </summary>
    /// <param name="inputData">Training batch (<c>XTrain</c>/<c>YTrain</c>) plus the model to shard
    /// (<c>InitialSolution</c>, or the wrapped <see cref="OptimizerBase{T,TInput,TOutput}"/>'s Model).</param>
    /// <param name="shardGradients"><c>false</c> = ZeRO-1 (AllReduce full gradients, then slice this
    /// rank's shard: gradients replicated, only optimizer state + parameters sharded). <c>true</c> =
    /// ZeRO-2/3 (ReduceScatter: gradients are sharded too, so the full averaged gradient is never
    /// materialized on any single rank).</param>
    protected OptimizationResult<T, TInput, TOutput> RunShardedZeroStep(
        OptimizationInputData<T, TInput, TOutput> inputData,
        bool shardGradients)
    {
        if (inputData is null) throw new ArgumentNullException(nameof(inputData));

        var model = inputData.InitialSolution
            ?? (WrappedOptimizer as OptimizerBase<T, TInput, TOutput>)?.Model?.Clone()
            ?? throw new InvalidOperationException(
                "ZeRO sharded step requires a model to compute gradients from: set " +
                "OptimizationInputData.InitialSolution, or wrap an OptimizerBase whose Model is set.");

        if (WrappedOptimizer is not IGradientBasedOptimizer<T, TInput, TOutput> gradOpt)
            throw new InvalidOperationException(
                $"ZeRO requires a gradient-based wrapped optimizer; received {WrappedOptimizer.GetType().Name}.");

        var paramModel = InterfaceGuard.Parameterizable(model);
        var gradModel = InterfaceGuard.GradientComputable(model);

        var originalParams = paramModel.GetParameters();
        if (originalParams is null || originalParams.Length == 0)
            return new OptimizationResult<T, TInput, TOutput> { BestSolution = model };
        int totalParams = originalParams.Length;

        int worldSize = WorldSize < 1 ? 1 : WorldSize;
        // Ceil split so the shards tile the (possibly zero-padded) parameter vector evenly — required
        // by ReduceScatter (length divisible by worldSize) and AllGather (equal-length concatenation).
        int shardSize = (totalParams + worldSize - 1) / worldSize;
        int paddedLen = shardSize * worldSize;
        int myStart = Rank * shardSize;

        // 1. Backward only — gradients WITHOUT advancing optimizer state.
        var gradients = gradModel.ComputeGradients(inputData.XTrain, inputData.YTrain);
        if (gradients is null || gradients.Length == 0)
            return new OptimizationResult<T, TInput, TOutput> { BestSolution = model };
        if (gradients.Length != totalParams)
            throw new InvalidOperationException(
                $"ZeRO: gradient length {gradients.Length} does not match parameter count {totalParams}.");

        // 2. Reduce across ranks; select this rank's gradient shard. CpuOffloadGradients must make the
        //    buffer that is actually COMMUNICATED CPU-resident, so offload BEFORE the collective.
        Vector<T> myGradShard;
        if (shardGradients)
        {
            // ZeRO-2/3: ReduceScatter averages AND scatters — pad to a multiple of worldSize first.
            var paddedGrads = PadTo(gradients, paddedLen);
            OffloadGradientsToCpu(paddedGrads);
            myGradShard = Config.CommunicationBackend.ReduceScatter(paddedGrads, ReductionOperation.Average);
        }
        else
        {
            // ZeRO-1: AllReduce averages the replicated full gradients in place; slice this shard.
            OffloadGradientsToCpu(gradients);
            Config.CommunicationBackend.AllReduce(gradients, ReductionOperation.Average);
            myGradShard = SliceShard(gradients, myStart, shardSize, totalParams);
        }

        // 3. Sharded update — the SINGLE optimizer step; Adam m/v state sized to the shard; the
        //    CPU-offload engine swap is scoped to this update only.
        var myParamShard = SliceShard(originalParams, myStart, shardSize, totalParams);
        Vector<T> updatedShard;
        using (BeginCpuOffloadScope())
        {
            updatedShard = gradOpt.UpdateParameters(myParamShard, myGradShard);
        }

        // 4. AllGather the shards -> full (padded) params -> trim padding -> write back.
        var gathered = worldSize > 1
            ? Config.CommunicationBackend.AllGather(updatedShard)
            : updatedShard;
        var finalParams = TrimTo(gathered, totalParams);
        paramModel.SetParameters(finalParams);

        // Stage-3 parameter offload hook (no-op unless CpuOffloadParams is set).
        OffloadParamsToCpu(model);

        return new OptimizationResult<T, TInput, TOutput> { BestSolution = model };
    }

    /// <summary>
    /// Executes one paper-faithful DATA-PARALLEL optimizer step for the pure-replication strategies —
    /// synchronous DDP (Li et al. 2020, "PyTorch Distributed"), Downpour / parameter-server async SGD
    /// (Dean et al. 2012), elastic DDP, and gradient-compressed DDP. Unlike <see cref="RunShardedZeroStep"/>
    /// the parameters are NOT sharded (every rank holds the full vector), so the step is:
    /// <list type="number">
    /// <item><b>Backward only</b> — full-vector gradient via
    /// <see cref="IGradientComputable{T,TInput,TOutput}.ComputeGradients"/>, WITHOUT running the wrapped
    /// optimizer's full <c>Optimize</c> loop. This is the crux: the wrapped optimizer's Adam m/v state is
    /// advanced EXACTLY once per global step (in step 3), not once per local inner iteration — so every
    /// rank stays in lock-step and the reduce is over a single, comparable gradient (true per-step DDP, as
    /// PyTorch performs in its backward gradient hook, rather than local-SGD-style multi-step drift).</item>
    /// <item><b>Reduce</b> the full gradient across ranks with <paramref name="reduction"/> (Average = the
    /// mean gradient of synchronous DDP). Identity at world size 1. CpuOffloadGradients makes the
    /// communicated buffer CPU-resident before the collective.</item>
    /// <item><b>Single update</b> — one
    /// <see cref="IGradientBasedOptimizer{T,TInput,TOutput}.ApplyGradients"/> from the ORIGINAL (pre-step)
    /// parameters, which is double-step-safe and optimizer-agnostic; the CPU-offload engine swap is scoped
    /// to this update alone.</item>
    /// </list>
    /// Because every rank applies the SAME reduced gradient from the SAME original parameters with a
    /// fresh-per-step update, all replicas end bit-identical — which is exactly what the two-rank
    /// data-parallel invariants assert.
    /// </summary>
    /// <param name="inputData">Training batch (<c>XTrain</c>/<c>YTrain</c>) plus the model
    /// (<c>InitialSolution</c>, or the wrapped <see cref="OptimizerBase{T,TInput,TOutput}"/>'s Model).</param>
    /// <param name="reduction">The cross-rank gradient reduction (<see cref="ReductionOperation.Average"/>
    /// for the DDP mean gradient).</param>
    /// <param name="transformGradientBeforeReduce">Optional per-rank transform applied to the gradient
    /// buffer that is actually COMMUNICATED (e.g. gradient compression / quantization); identity when null.
    /// Must return a vector matching the parameter count.</param>
    protected OptimizationResult<T, TInput, TOutput> RunDataParallelStep(
        OptimizationInputData<T, TInput, TOutput> inputData,
        ReductionOperation reduction,
        Func<Vector<T>, Vector<T>>? transformGradientBeforeReduce = null)
    {
        if (inputData is null) throw new ArgumentNullException(nameof(inputData));

        var model = inputData.InitialSolution
            ?? (WrappedOptimizer as OptimizerBase<T, TInput, TOutput>)?.Model?.Clone()
            ?? throw new InvalidOperationException(
                "Data-parallel step requires a model to compute gradients from: set " +
                "OptimizationInputData.InitialSolution, or wrap an OptimizerBase whose Model is set.");

        if (WrappedOptimizer is not IGradientBasedOptimizer<T, TInput, TOutput> gradOpt)
            throw new InvalidOperationException(
                $"Data-parallel training requires a gradient-based wrapped optimizer; received {WrappedOptimizer.GetType().Name}.");

        var paramModel = InterfaceGuard.Parameterizable(model);
        var gradModel = InterfaceGuard.GradientComputable(model);

        var originalParams = paramModel.GetParameters();
        if (originalParams is null || originalParams.Length == 0)
            return new OptimizationResult<T, TInput, TOutput> { BestSolution = model };

        // 1. Backward only — full gradient WITHOUT advancing optimizer state through a local loop.
        var gradients = gradModel.ComputeGradients(inputData.XTrain, inputData.YTrain);
        if (gradients is null || gradients.Length == 0)
            return new OptimizationResult<T, TInput, TOutput> { BestSolution = model };
        if (gradients.Length != originalParams.Length)
            throw new InvalidOperationException(
                $"Data-parallel: gradient length {gradients.Length} does not match parameter count {originalParams.Length}.");

        // Optional per-rank transform on the communicated buffer (e.g. gradient compression).
        if (transformGradientBeforeReduce is not null)
        {
            gradients = transformGradientBeforeReduce(gradients);
            if (gradients is null || gradients.Length != originalParams.Length)
                throw new InvalidOperationException(
                    "Data-parallel: gradient transform must return a non-null vector matching the parameter count.");
        }

        // 2. Reduce the full gradient across ranks (identity at world size 1). Offload BEFORE the
        //    collective so CpuOffloadGradients makes the COMMUNICATED buffer CPU-resident.
        OffloadGradientsToCpu(gradients);
        if (WorldSize > 1)
            Config.CommunicationBackend.AllReduce(gradients, reduction);

        // 3. SINGLE update from the ORIGINAL params (no double-step); optimizer engine-swap scoped here.
        IFullModel<T, TInput, TOutput> updated;
        using (BeginCpuOffloadScope())
        {
            updated = gradOpt.ApplyGradients(originalParams, gradients, model);
        }

        OffloadParamsToCpu(updated);
        return new OptimizationResult<T, TInput, TOutput> { BestSolution = updated };
    }

    /// <summary>Returns a length-<paramref name="length"/> copy of <paramref name="source"/>, zero-padding any tail.</summary>
    private Vector<T> PadTo(Vector<T> source, int length)
    {
        if (source.Length == length) return source;
        var padded = new T[length];
        int copy = Math.Min(source.Length, length);
        for (int i = 0; i < copy; i++) padded[i] = source[i];
        for (int i = copy; i < length; i++) padded[i] = NumOps.Zero;
        return new Vector<T>(padded);
    }

    /// <summary>Extracts <paramref name="length"/> elements from <paramref name="start"/>, zero-padding
    /// indices at or beyond <paramref name="validEnd"/> so every rank's shard is the same length (the
    /// collectives require equal-length shards; the padding is trimmed after the final AllGather).</summary>
    private Vector<T> SliceShard(Vector<T> source, int start, int length, int validEnd)
    {
        var shard = new T[length];
        for (int i = 0; i < length; i++)
        {
            int idx = start + i;
            shard[i] = idx < validEnd ? source[idx] : NumOps.Zero;
        }
        return new Vector<T>(shard);
    }

    /// <summary>Returns the first <paramref name="length"/> elements (drops the shard padding introduced for the collectives).</summary>
    private Vector<T> TrimTo(Vector<T> source, int length)
    {
        if (source.Length == length) return source;
        var trimmed = new T[length];
        int copy = Math.Min(source.Length, length);
        for (int i = 0; i < copy; i++) trimmed[i] = source[i];
        for (int i = copy; i < length; i++) trimmed[i] = NumOps.Zero;
        return new Vector<T>(trimmed);
    }

    /// <summary>
    /// Force-materializes a gradient vector's backing array (drains any pending
    /// deferred GPU download) and drops the GPU-cached buffer for that array,
    /// implementing the runtime side of <see cref="IShardingConfiguration{T}.CpuOffloadGradients"/>.
    ///
    /// <para>Called by derived optimizers immediately before the sharded
    /// reduction (<see cref="ICommunicationBackend{T}.AllReduce"/> for DDP,
    /// <see cref="ICommunicationBackend{T}.ReduceScatter"/> for ZeRO2/3 and
    /// FSDP). The reduction operates on the managed <c>Vector&lt;T&gt;</c>
    /// backing array directly — after this call, that array is guaranteed
    /// to hold the current gradient values (not stale zeros from a not-yet-
    /// materialized deferred download) and no GPU cache entry pins a copy
    /// of the pre-reduction bytes.</para>
    ///
    /// <para>DeepSpeed's ZeRO Stage-2 contract: gradients are freed from GPU
    /// VRAM as soon as the backward produces them, reduced across ranks in
    /// CPU RAM, then discarded (each rank keeps only its own shard of the
    /// averaged gradient). This helper does the "freed from GPU" half; the
    /// reduce-scatter that follows in each derived optimizer produces the
    /// per-rank shard.</para>
    /// </summary>
    protected void OffloadGradientsToCpu(Vector<T>? gradients)
    {
        if (!Config.CpuOffloadGradients || gradients is null || gradients.Length == 0) return;
        var array = gradients.GetDataArray();
        if (array is null) return;
        // Force any pending deferred GPU download to drain into `array` NOW so
        // the follow-up AllReduce/ReduceScatter sees the just-produced gradient
        // values instead of the uninitialized bytes GC.AllocateUninitializedArray
        // left in place (see AiDotNet.Tensors PR #604 FinishGpuOp).
        AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.TryMaterialize(array);
        // Drop the GPU cache entry so the next op that touches this array
        // re-uploads from CPU (which now holds the post-reduce shard). Without
        // this the GPU engine's persistent-weight cache would keep serving the
        // pre-reduce bytes.
        if (AiDotNetEngine.Current is DirectGpuTensorEngine gpu)
        {
            gpu.InvalidateWeightCache(array);
        }
    }

    /// <summary>
    /// After the sharded optimizer step, drops any GPU-cached parameter
    /// buffers for the model's parameters, implementing the runtime side of
    /// <see cref="IShardingConfiguration{T}.CpuOffloadParams"/>. The next
    /// forward pass then re-uploads from the CPU-resident (updated) params,
    /// so the parameter tensors don't accumulate a resident GPU shadow copy
    /// between steps.
    ///
    /// <para>Contract note: this is only meaningful for sharded strategies
    /// (FSDP, ZeRO2, ZeRO3) — DDP replicates the full parameter set on every
    /// rank and has no shard to page. DDP callers that reach this helper
    /// short-circuit at the flag check via the sanity guard the facade
    /// applies; the guard here is a fallback so the helper itself is safe
    /// to call from any strategy.</para>
    /// </summary>
    protected void OffloadParamsToCpu(IFullModel<T, TInput, TOutput>? model)
    {
        if (!Config.CpuOffloadParams || model is null) return;
        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu) return;
        Vector<T>? parameters;
        try { parameters = InterfaceGuard.Parameterizable(model).GetParameters(); }
        catch (System.Exception ex)
        {
            // Do NOT silently disable offload on failure — the caller asked for CpuOffloadParams, so a
            // model whose parameters cannot be extracted is a real misconfiguration that must surface.
            throw new System.InvalidOperationException(
                "CpuOffloadParams is enabled but the model's parameters could not be extracted for " +
                "offload. Wrap a parameterizable model, or disable CpuOffloadParams.", ex);
        }
        if (parameters is null || parameters.Length == 0) return;
        var array = parameters.GetDataArray();
        if (array is null) return;
        // Force any pending download so the CPU array holds the updated
        // values, THEN invalidate the GPU cache so the next forward
        // re-uploads (the point of CpuOffloadParams is that params live on
        // CPU between steps; the GPU sees them only during the next forward).
        AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.TryMaterialize(array);
        gpu.InvalidateWeightCache(array);
    }

    // RAII holder for the engine swap. AiDotNetEngine.Current is a PROCESS-GLOBAL static
    // (Volatile.Read/Write over one `_current` field) that every tensor op reads. That has two
    // consequences for concurrent offload scopes:
    //   1. Correctness: the CPU-offload contract requires every op inside the wrapped step to run
    //      on CpuEngine. If another thread reassigns the global Current to a GPU engine mid-step,
    //      this step's ops silently run on GPU — defeating the offload.
    //   2. Safety: two threads interleaving save→swap→restore around a shared global can capture
    //      each other's temporary CpuEngine as "prior" and restore the wrong engine.
    // Both are inherent to a single global Current, so the coherent fix is to serialize the whole
    // swap+step+restore through a process-wide gate — the equivalent, for one shared Current, of the
    // AsyncLocal engine stack an execution-context-local Current would provide. The gate is a Monitor
    // (reentrant), so nested offload scopes on the same thread — an offloaded optimizer wrapping
    // another — still work. The fast path (offload off, or already on CPU) takes no lock and returns
    // the shared NoOp singleton, so the common case stays allocation- and contention-free.
    private static readonly object EngineSwapGate = new();

    private static class CpuOffloadScope
    {
        internal static IDisposable Enter(IShardingConfiguration<T> config)
        {
            if (!config.CpuOffloadOptimizer)
                return NoOp.Instance;

            bool taken = false;
            System.Threading.Monitor.Enter(EngineSwapGate, ref taken);
            try
            {
                var prior = AiDotNetEngine.Current;
                // Already on CPU (no GPU engine active) → nothing to swap; release the gate now.
                if (prior is CpuEngine)
                {
                    if (taken) { System.Threading.Monitor.Exit(EngineSwapGate); taken = false; }
                    return NoOp.Instance;
                }

                AiDotNetEngine.Current = new CpuEngine();
                // Ownership of the held gate transfers to EngineRestore, which releases it on Dispose
                // AFTER restoring the prior engine.
                var restore = new EngineRestore(prior);
                taken = false;
                return restore;
            }
            catch
            {
                if (taken) System.Threading.Monitor.Exit(EngineSwapGate);
                throw;
            }
        }

        private sealed class NoOp : IDisposable
        {
            internal static readonly NoOp Instance = new();
            public void Dispose() { }
        }

        private sealed class EngineRestore : IDisposable
        {
            private IEngine? _prior;
            internal EngineRestore(IEngine prior) { _prior = prior; }
            public void Dispose()
            {
                var prior = System.Threading.Interlocked.Exchange(ref _prior, null);
                if (prior is not null)
                {
                    AiDotNetEngine.Current = prior;
                    System.Threading.Monitor.Exit(EngineSwapGate);
                }
            }
        }
    }

    /// <summary>
    /// Synchronizes model parameters across all processes using AllReduce with averaging.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method averages parameters across all processes, ensuring consistency.
    /// It's called after optimization steps to keep all processes synchronized.
    /// </para>
    /// <para><b>For Beginners:</b> After each process updates its model, we need to
    /// make sure everyone has the same parameters.
    ///
    /// This method averages the parameters from all processes. For example, if GPU 0
    /// calculated parameter value 1.0 and GPU 1 calculated 1.2, after sync both will
    /// have 1.1 (the average).
    /// </para>
    /// </remarks>
    /// <param name="model">The model whose parameters to synchronize</param>
    protected virtual void SynchronizeParameters(IFullModel<T, TInput, TOutput>? model)
    {
        if (model == null)
        {
            return;
        }

        // Don't sync if it's already a sharded model (handles its own sync)
        if (model is IShardedModel<T, TInput, TOutput>)
        {
            return;
        }

        // Get current parameters
        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();

        // Average parameters across all processes
        Config.CommunicationBackend.AllReduce(parameters, ReductionOperation.Average);

        // Update model with averaged parameters
        InterfaceGuard.Parameterizable(model).SetParameters(parameters);
    }

    /// <inheritdoc/>
    public virtual bool ShouldEarlyStop()
    {
        // Delegate to wrapped optimizer
        bool localDecision = WrappedOptimizer.ShouldEarlyStop();

        // In distributed training, we need consensus on early stopping
        // All processes should agree to stop, otherwise some might continue while others stop
        // For now, we'll use a simple approach: if any process wants to stop, all stop

        // Create a vector with the local decision (1 for stop, 0 for continue)
        var decision = new Vector<T>(new[] { localDecision ? NumOps.One : NumOps.Zero });

        // Get the maximum across all processes
        // If any process returns 1 (stop), the max will be 1
        Config.CommunicationBackend.AllReduce(decision, ReductionOperation.Max);

        // Check if the result indicates stopping
        return !NumOps.Equals(decision[0], NumOps.Zero);
    }

    /// <inheritdoc/>
    public virtual OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return WrappedOptimizer.GetOptions();
    }

    /// <summary>
    /// Gets the gradients computed during the last optimization step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sharded optimizers delegate gradient access to the wrapped optimizer.
    /// If the wrapped optimizer is gradient-based, this will return the actual computed gradients.
    /// Otherwise, it returns an empty vector.
    /// </para>
    /// </remarks>
    public virtual Vector<T> LastComputedGradients
    {
        get
        {
            var gradientOptimizer = WrappedOptimizer as IGradientBasedOptimizer<T, TInput, TOutput>;
            return gradientOptimizer?.LastComputedGradients ?? Vector<T>.Empty();
        }
    }

    /// <summary>
    /// Applies pre-computed gradients to a model's parameters.
    /// </summary>
    /// <param name="gradients">The gradients to apply</param>
    /// <param name="model">The model to update</param>
    /// <returns>The updated model</returns>
    /// <remarks>
    /// <para>
    /// Sharded optimizers delegate gradient application to the wrapped optimizer.
    /// If the wrapped optimizer is gradient-based, this will apply the gradients.
    /// Otherwise, throws NotSupportedException.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">If the wrapped optimizer is not gradient-based</exception>
    public virtual IFullModel<T, TInput, TOutput> ApplyGradients(Vector<T> gradients, IFullModel<T, TInput, TOutput> model)
    {
        var gradientOptimizer = WrappedOptimizer as IGradientBasedOptimizer<T, TInput, TOutput>;
        if (gradientOptimizer == null)
        {
            throw new NotSupportedException(
                $"ApplyGradients requires a gradient-based optimizer, but wrapped optimizer {WrappedOptimizer.GetType().Name} does not implement IGradientBasedOptimizer.");
        }

        return gradientOptimizer.ApplyGradients(gradients, model);
    }

    /// <inheritdoc/>
    public virtual void Reset()
    {
        // Delegate reset to wrapped optimizer
        WrappedOptimizer.Reset();
    }

    /// <inheritdoc/>
    public virtual void SetModel(IFullModel<T, TInput, TOutput> model)
    {
        // Delegate to wrapped optimizer
        WrappedOptimizer.SetModel(model);
    }

    /// <inheritdoc/>
    public abstract byte[] Serialize();

    /// <inheritdoc/>
    public abstract void Deserialize(byte[] data);

    /// <inheritdoc/>
    public virtual int[] GetInputShape()
    {
        // Delegate to the wrapped optimizer's model if it exposes shape
        if (WrappedOptimizer is IModelShape shapeProvider)
        {
            return shapeProvider.GetInputShape();
        }

        return Array.Empty<int>();
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        if (WrappedOptimizer is IModelShape shapeProvider)
        {
            return shapeProvider.GetOutputShape();
        }

        return Array.Empty<int>();
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc/>
    public virtual void SaveModel(string filePath)
    {
        // Only rank 0 saves to avoid conflicts
        if (Rank == 0)
        {
            Helpers.ModelPersistenceGuard.EnforceBeforeSave();
            using (Helpers.ModelPersistenceGuard.InternalOperation())
            {
                var data = Serialize();
                byte[] envelopedData = ModelFileHeader.WrapWithHeader(
                    data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary);
                File.WriteAllBytes(filePath, envelopedData);
            }
        }

        // Wait for rank 0 to finish writing
        Config.CommunicationBackend.Barrier();
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();

        // All processes read the same file
        var data = File.ReadAllBytes(filePath);

        // Extract payload from AIMF envelope
        data = ModelFileHeader.ExtractPayload(data);

        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            Deserialize(data);
        }

        // Ensure all processes finish loading
        Config.CommunicationBackend.Barrier();
    }
}
