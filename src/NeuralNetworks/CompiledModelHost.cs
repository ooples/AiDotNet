using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Composable inference-compilation helper that traces a model's forward pass
/// on the first call at a given shape and replays the compiled plan on subsequent
/// calls. Falls back to eager execution when compilation isn't beneficial or
/// fails at trace time.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
/// <remarks>
/// <para>
/// Designed as a <b>component</b>, not a base class. Consumers own the layer list
/// and Predict API; they compose one <see cref="CompiledModelHost{T}"/> instance
/// to get compile + replay + invalidation + Dispose cleanup with a tiny surface:
/// </para>
/// <code>
/// public class MyNetwork {
///     private readonly CompiledModelHost&lt;T&gt; _host = new();
///     private int _structureVersion;
///
///     public Tensor&lt;T&gt; Predict(Tensor&lt;T&gt; input) =&gt;
///         _host.Predict(input, _structureVersion, () =&gt; ForwardEager(input));
///
///     public void Dispose() { _host.Dispose(); /* cascade layers */ }
/// }
/// </code>
/// <para>
/// This replaces the ad-hoc <c>_compiledInferenceCache</c> fields scattered across
/// model families (<see cref="NeuralNetworkBase{T}"/>, <c>NoisePredictorBase</c>,
/// etc.) with a single implementation — one attachment point for every future
/// compilation feature (AOT plan serialization, CUDA Graph capture, symbolic
/// shape plans, persistent autotune).
/// </para>
/// <para>
/// <b>Invalidation contract:</b> callers pass a monotonic
/// <paramref name="structureVersion"/> on every <c>Predict</c>. When the caller's
/// layer graph mutates (lazy-init resize, layer add/remove, weight swap), bump
/// the version — the host detects the change and drops the stale plan before
/// compiling a fresh one. This prevents replay against a tensor graph that no
/// longer matches the ops the plan captured.
/// </para>
/// </remarks>
internal sealed class CompiledModelHost<T> : IDisposable
{
    /// <summary>
    /// Lazily-allocated Tensors-package compile cache. Keyed internally by input
    /// shape — recompiles automatically when shape changes. We wrap it with our
    /// own structure-version check on top because the cache has no way to know
    /// when the CALLER's layer graph changed.
    /// </summary>
    private CompiledModelCache<T>? _cache;

    /// <summary>Structure version of the last successful compile. Mismatch → invalidate.</summary>
    private int _lastCompiledVersion = -1;

    private bool _disposed;

    /// <summary>
    /// Symbolic-shape strategy that governs which input dims the compile cache
    /// treats as variable. BatchDynamic (default) lets a single compiled plan
    /// serve batch-size 1, 4, 32, 128 etc. without recompiling each time —
    /// matching PyTorch's default <c>torch.compile(dynamic=True)</c> posture.
    /// </summary>
    private readonly SymbolicShapeMode _shapeMode;

    /// <summary>
    /// Stable identity string for this model (typically the full type name plus any
    /// architecture hash). Used as the filename prefix for disk-cached compiled plans
    /// so multiple model types can share the same <see cref="PlanCache"/> directory
    /// without collision. Null = disk caching disabled for this host.
    /// </summary>
    private readonly string? _modelIdentity;

    /// <summary>
    /// Set of shape keys we've already attempted to load from disk for a given
    /// structure version. Prevents repeated failed-load IO on every Predict call.
    /// Keyed as "v{structureVersion}_s{shapeHash}".
    /// </summary>
    private HashSet<string>? _diskCheckedShapes;

    /// <summary>
    /// Plans loaded from disk for this host. Preempt <see cref="_cache"/> so we skip
    /// the compile pass entirely on cold-start replay.
    /// </summary>
    private Dictionary<string, ICompiledPlan<T>>? _preloadedPlans;

    /// <summary>
    /// <b>Hot-path cache for steady-state replay.</b> After every successful
    /// <see cref="Predict"/> call we capture the plan + shape-hash + version
    /// here. The next call, if it sees matching shape and version, takes a
    /// short-circuit straight to <c>SetInputs</c>+<c>Execute</c>, skipping
    /// the full <see cref="Predict"/> path with its lock-acquire, shape
    /// clone, disk-cache poll, symbolic-shape build, fallback-cache lookup,
    /// and fire-and-forget disk save. Those amortise once across the lifetime
    /// of the (shape, version) pair — re-paying them on every call doubles
    /// per-call wall time on small/medium Transformer inference. The hot-path
    /// preserves all the correctness guarantees: still goes through
    /// <c>SetInputs</c> for value-stable replay, still type-checks via the
    /// <c>ICompiledPlan&lt;T&gt;</c> interface, still safely yields to the
    /// slow path when shape or version changes. Volatile reads avoid the
    /// <c>_sync</c> acquisition on every call; the slow path re-validates
    /// under the lock.
    /// </summary>
    private volatile ICompiledPlan<T>? _hotPlan;
    private long _hotShapeKey;
    private int _hotVersion = -1;

    /// <summary>
    /// "Eager-only" shapes — (shape, version) pairs that have FAILED to
    /// compile (trace pass couldn't resolve a leaf, plan execute threw, etc.).
    /// On the next call with the same shape we skip the retry-and-throw cycle
    /// entirely and go straight to eager. Without this, every Predict call
    /// pays the full trace overhead followed by the catch + cache-invalidate
    /// path — the dominant cost on models where the captured graph never
    /// references the user's input tensor (the canonical failure mode is
    /// EmbeddingLayer reading <c>int</c> token IDs from the input that the
    /// GraphMode tracer doesn't recognise as a graph leaf, so the captured
    /// trace has no consumer of the input shape and CompileInference
    /// throws). Cleared on version swap so a future layer-graph change can
    /// re-attempt compile if the new shape works.
    /// </summary>
    private HashSet<(long ShapeKey, int Version)>? _eagerOnlyShapeKeys;

    /// <summary>
    /// Diagnostic counter: number of times the hot-path was hit (cache-only
    /// SetInputs+Execute) vs the slow path. Used by tests / profiling to
    /// verify the steady-state replay short-circuit is actually engaged.
    /// </summary>
    private long _hotPathHits;
    private long _slowPathCalls;

    /// <summary>Hot-path replay count since this host was created. Diagnostic-only.</summary>
    public long HotPathHits => System.Threading.Interlocked.Read(ref _hotPathHits);

    /// <summary>Slow-path call count since this host was created. Diagnostic-only.</summary>
    public long SlowPathCalls => System.Threading.Interlocked.Read(ref _slowPathCalls);

    public CompiledModelHost(
        SymbolicShapeMode shapeMode = SymbolicShapeMode.BatchDynamic,
        string? modelIdentity = null)
    {
        _shapeMode = shapeMode;
        _modelIdentity = modelIdentity;
    }

    /// <summary>
    /// Synchronizes lifecycle mutations on <c>_cache</c>, <c>_lastCompiledVersion</c>,
    /// and <c>_disposed</c>. Without it a concurrent <c>Dispose</c>/<c>Invalidate</c>
    /// racing with <c>Predict</c> on the same model instance can tear down the
    /// cache mid-call (model serving in a request pool is the typical scenario).
    /// </summary>
    private readonly object _sync = new();

    /// <summary>
    /// Count of in-flight <see cref="Predict"/> calls. Incremented inside the
    /// lock right before we release it with a captured cache reference,
    /// decremented in the <c>finally</c> after compile/replay completes.
    /// <see cref="Dispose"/> and the version-swap path use this to defer
    /// actual cache disposal until no thread still holds a reference —
    /// without this, tearing down <c>_cache</c> mid-call could crash or
    /// corrupt in-flight compile/execute. Used only under <c>_sync</c> in
    /// lifecycle checks; the in-flight body increments/decrements via
    /// Interlocked since the calling thread has already released the lock.
    /// </summary>
    private int _activeCalls;

    /// <summary>
    /// Old caches that have been detached (by version swap) but are still
    /// potentially referenced by in-flight Predict calls. The last
    /// Predict to return (transitioning <see cref="_activeCalls"/> to 0)
    /// drains and disposes them.
    /// </summary>
    private List<CompiledModelCache<T>>? _pendingDisposeCaches;

    /// <summary>
    /// True once <see cref="Dispose"/> has been called. If any
    /// <see cref="_activeCalls"/> remain at that moment, the owning cache
    /// is deferred onto <see cref="_pendingDisposeCaches"/> and the actual
    /// cleanup happens when the last in-flight call finishes.
    /// </summary>
    private bool _disposeRequested;

    /// <summary>
    /// Attempts to compile-and-replay the forward pass. Falls back to
    /// <paramref name="eagerForward"/> when compilation is disabled
    /// (<see cref="TensorCodecOptions.EnableCompilation"/> is false) or
    /// throws at trace time.
    /// </summary>
    /// <param name="input">The input tensor. Its shape keys the compile cache.</param>
    /// <param name="structureVersion">
    /// Monotonic layer-graph version from the caller. When this differs from the
    /// version the current plan was compiled against, the cache is invalidated
    /// and a fresh trace runs.
    /// </param>
    /// <param name="eagerForward">
    /// The eager forward pass. Called (a) to trace during compilation, (b) to
    /// fall back when compilation disabled or failed, (c) as the plan's
    /// regenerate hook on shape change.
    /// </param>
    /// <returns>The forward-pass output.</returns>
    /// <remarks>
    /// The trace lambda is invoked exactly ONCE per {shape, structureVersion}
    /// pair (under GraphMode). Subsequent calls at the same shape+version replay
    /// the compiled plan without re-entering the lambda.
    /// </remarks>
    public Tensor<T> Predict(
        Tensor<T> input,
        int structureVersion,
        Func<Tensor<T>> eagerForward)
    {
        if (eagerForward is null)
            throw new ArgumentNullException(nameof(eagerForward));

        // ---------- HOT-PATH: steady-state replay short-circuit ----------
        // After the first successful Predict for a given (shape, version),
        // every subsequent call with the same (shape, version) skips
        // everything below — no lock, no shape clone, no disk-cache poll,
        // no symbolic-shape build, no fallback cache lookup, no disk save.
        // Just SetInputs + Execute on the cached plan. Volatile read of
        // _hotPlan ensures cross-thread visibility without lock; shape and
        // version equality re-validate the cache identity. If any check
        // fails we drop through to the slow path below which is correct
        // under all conditions (including version swaps, cache invalidation,
        // and Dispose races).
        // ---------- EAGER-ONLY shortcut: bail out before any further work
        // when this (shape, version) has previously failed compile. This
        // skips the trace retry, hot-path bool computations, and slow-path
        // entry — straight to eager. Without it every call for a non-
        // compileable shape pays ~14 ms trace + throw + cache-invalidate
        // overhead plus the hot-path bookkeeping. Volatile snapshot read
        // avoids the lock; writers use copy-on-write so readers never see
        // a torn HashSet.
        long shapeKey = 0;
        bool shapeKeyOk = TryComputeShapeKey(input._shape, out shapeKey);
        var eagerSnap = System.Threading.Volatile.Read(ref _eagerOnlyShapeKeys);
        if (shapeKeyOk
            && eagerSnap is not null
            && eagerSnap.Contains((shapeKey, structureVersion)))
        {
            return eagerForward();
        }

        // ---------- HOT-PATH: steady-state replay short-circuit ----------
        // Cached plan + matching shape + version + compilation still on +
        // host not disposed. Everything below is the slow-path that compiles
        // a new plan on first encounter.
        var hotPlan = _hotPlan;
        bool nullPlan = hotPlan is null;
        bool versionMatches = !nullPlan && _hotVersion == structureVersion;
        bool shapeMatches = !nullPlan && versionMatches && shapeKeyOk && _hotShapeKey == shapeKey;
        bool compilationOn = !nullPlan && versionMatches && shapeMatches
            && TensorCodecOptions.Current.EnableCompilation;
        bool notDisposed = !nullPlan && versionMatches && shapeMatches && compilationOn
            && !_disposed && !_disposeRequested;
        if (notDisposed && hotPlan is not null)
        {
            try
            {
                System.Threading.Interlocked.Increment(ref _hotPathHits);
                hotPlan.SetInputs(new[] { input });
                return hotPlan.Execute();
            }
            catch (ObjectDisposedException)
            {
                // Plan was disposed concurrently (version swap drained the
                // pending-dispose queue). Fall through to the slow path
                // which will lock + re-acquire a fresh cache.
                _hotPlan = null;
            }
            catch (System.Exception ex) when (
                ex is System.ArgumentException
                or System.InvalidOperationException
                or System.IndexOutOfRangeException
                or System.NullReferenceException)
            {
                // Compile-replay should be best-effort: any other recoverable
                // failure inside SetInputs / Execute (shape / arity mismatch
                // after an edge-case cache invalidation, intermittent native
                // backend hiccup, etc.) must fall through to the slow path
                // instead of bubbling out and bypassing the eager fallback
                // CompiledModelHost's contract promises. Clear _hotPlan so
                // the slow path rebuilds against a fresh cache rather than
                // returning to a known-bad plan on the next call.
                _hotPlan = null;
                // OutOfMemoryException / StackOverflowException / other
                // truly unrecoverable exceptions intentionally NOT caught
                // here — they should propagate.
            }
        }
        System.Threading.Interlocked.Increment(ref _slowPathCalls);

        // Quick state checks + cache acquisition under lock; the actual
        // compile/replay AND the eager fallback run OUTSIDE the lock because
        // they can be slow and blocking other threads on them would defeat
        // the throughput win. CompiledModelCache itself is thread-safe so
        // concurrent compiles for distinct shapes are fine.
        //
        // Concurrency contract between Dispose/Invalidate/version-swap and
        // in-flight Predict: we increment _activeCalls inside the lock so
        // Dispose can tell whether it's safe to immediately Dispose the
        // owning cache or must defer. On version swap, the DETACHED old
        // cache is not Disposed inline — it goes onto _pendingDisposeCaches
        // and the last in-flight caller drains the queue.
        CompiledModelCache<T>? cache;
        bool fallToEager;
        lock (_sync)
        {
            fallToEager = _disposed || _disposeRequested
                || !TensorCodecOptions.Current.EnableCompilation;

            if (!fallToEager)
            {
                // Layer graph changed since last compile — stale plans would
                // replay against the old tensor references. Swap in a FRESH
                // CompiledModelCache<T> instance rather than invalidating the
                // existing one in place. The OLD cache goes onto the pending-
                // dispose queue because in-flight callers may still hold
                // references to it; draining happens when _activeCalls hits 0.
                if (_cache is not null && structureVersion != _lastCompiledVersion)
                {
                    (_pendingDisposeCaches ??= new List<CompiledModelCache<T>>()).Add(_cache);
                    _cache = new CompiledModelCache<T>();
                    _lastCompiledVersion = structureVersion;
                    // Hot plan was bound to the old cache; clear so the next
                    // call falls into the slow path which will rebuild against
                    // the fresh cache + capture a new hot plan. Also clear the
                    // eager-only verdict cache so a fresh graph gets to attempt
                    // compile again — the prior failure was for a now-stale
                    // layer structure.
                    _hotPlan = null;
                    _eagerOnlyShapeKeys = null;
                }
                cache = _cache ??= new CompiledModelCache<T>();
                // Increment while still holding the lock — Dispose observes
                // a consistent snapshot of the counter.
                _activeCalls++;
            }
            else
            {
                cache = null;
            }
        }

        if (fallToEager || cache is null)
            return eagerForward();

        try
        {
            try
            {
                var concreteShape = (int[])input._shape.Clone();

                // Disk-cache preload: first time we see a shape at this structure
                // version, try to load a previously-compiled plan from PlanCache.
                // Skips the compile cost entirely on cold-start replay.
                if (TryUseDiskCachedPlan(concreteShape, structureVersion, input, out var preloadedResult))
                {
                    _lastCompiledVersion = structureVersion;
                    return preloadedResult;
                }

                // Preserve the Tensor<T> return from eagerForward so the compile
                // pass can explicitly identify the output tensor. Discarding the
                // return (previously `() => { eagerForward(); }`) can select a
                // different GetOrCompileInference overload / leave the output
                // tensor ambiguous to the tracer, producing wrong outputs on
                // replay. Keep the expression-bodied lambda so the value threads
                // through unchanged.
                var symbolicShape = BuildSymbolicShape(concreteShape, _shapeMode);
                var plan = symbolicShape is null
                    ? cache.GetOrCompileInference(concreteShape, () => eagerForward())
                    : cache.GetOrCompileInference(concreteShape, () => eagerForward(), symbolicShape);

                // Fire-and-forget disk save so subsequent cold starts skip the compile.
                MaybeSavePlanToDisk(plan, concreteShape, structureVersion);

                // Safe to write _lastCompiledVersion outside the lock — int writes
                // are atomic in .NET, and the value is only used as a hint by the
                // next Predict's version-mismatch check (which itself runs under
                // the lock and will recover from any race).
                _lastCompiledVersion = structureVersion;
                // VALUE-STABLE REPLAY: refresh the plan's captured input buffer
                // with the CURRENT call's data before Execute. Without this,
                // every call after the trace pass replayed against the trace-
                // time input bytes — producing silently-stale outputs for the
                // common "same shape, new values" pattern (the bug that kept
                // compilation off-by-default on the Predict path). SetInputs
                // is a span-to-span copy into the plan's captured buffer
                // (~O(input.Length) memcpy, sub-millisecond on typical
                // batches). The trace pass already wrote `input`'s data into
                // the captured buffer via the eager lambda, so this call is
                // a no-op (same source and destination data) on the
                // cache-miss path. If the rebind throws (multi-input plan
                // mismatch, strided captured view, etc.) the outer
                // narrow-fallback catch below invalidates the cache and
                // re-routes the call through eager so correctness is
                // preserved end-to-end.
                if (plan is ICompiledPlan<T> rebindable)
                {
                    rebindable.SetInputs(new[] { input });
                }
                var result = plan.Execute();

                // Capture this (plan, shape, version) as the hot-path target
                // so subsequent calls with matching shape and version skip
                // the entire slow path above. TryComputeShapeKey returns
                // false for ranks > 4 or dims > 65535 — those callers stay
                // on the slow path (correct, just slower). The write order
                // (ICompiledPlan? assignment last) makes the hot-path entry
                // safe under volatile read: a concurrent reader either sees
                // null or a fully-initialised (shape, version) pair.
                if (plan is ICompiledPlan<T> hotCandidate
                    && TryComputeShapeKey(concreteShape, out long hotKey))
                {
                    _hotShapeKey = hotKey;
                    _hotVersion = structureVersion;
                    _hotPlan = hotCandidate;
                }
                return result;
            }
            catch (Exception ex) when (
                // Narrow the fallback so fatal/unrecoverable CLR failures
                // propagate instead of being masked by the eager fallback.
                // Allocating again on a poisoned process (OOM, corrupted
                // managed heap) would just crash a second way. Keep the
                // fallback for compile/replay bugs (trace-time exceptions,
                // TensorShapeException, IndexOutOfRange, etc.) which are
                // the failures this catch was designed to tolerate.
                ex is not OutOfMemoryException &&
                ex is not AccessViolationException &&
                ex is not StackOverflowException &&
                ex is not BadImageFormatException &&
                ex is not InvalidProgramException &&
                ex is not System.Threading.ThreadAbortException &&
                ex is not AppDomainUnloadedException &&
                ex is not CannotUnloadAppDomainException)
            {
                // Compilation or replay threw. Invalidate under lock so a concurrent
                // Predict doesn't see a half-torn-down cache. If a broken plan was
                // already cached (replay path failed mid-execute), this drops it so
                // the next call recompiles instead of re-entering the same crash.
                lock (_sync)
                {
                    _cache?.Invalidate();
                    _lastCompiledVersion = -1;
                }
                // Trace the failure so the regression is observable in production
                // telemetry — silent fallback would let perf surveys be the first
                // signal that compiled inference stopped working. Include
                // ex.ToString() so the stack trace and inner exceptions are
                // preserved for diagnostics.
                System.Diagnostics.Trace.TraceWarning(
                    $"CompiledModelHost falling back to eager after compile/replay failure: " +
                    $"{ex.ToString()}");
                // Mark this (shape, version) as eager-only so the next call
                // skips the trace retry and goes straight to eager. The
                // canonical case is models whose forward path consumes the
                // input through a non-graph-traced operation (e.g.
                // EmbeddingLayer reading int token IDs via a lookup that the
                // GraphMode tracer doesn't record as a leaf consumer) — the
                // trace pass cannot bind the input shape, throws here, and
                // every subsequent call would repeat the same failure. The
                // ratchet is shape+version-keyed so a later layer-graph
                // change can re-attempt compile if the new graph works.
                if (TryComputeShapeKey(input._shape, out long failedShape))
                {
                    // Copy-on-write: build a new set from the existing one +
                    // the failed shape, then atomically swap. Readers using
                    // the snapshot reference see a consistent set without
                    // ever needing the lock on the hot path.
                    lock (_sync)
                    {
                        var current = _eagerOnlyShapeKeys;
                        var next = current is null
                            ? new HashSet<(long, int)>()
                            : new HashSet<(long, int)>(current);
                        next.Add((failedShape, structureVersion));
                        System.Threading.Volatile.Write(ref _eagerOnlyShapeKeys, next);
                    }
                }
                return eagerForward();
            }
        }
        finally
        {
            // Decrement the active-call counter and, if we're the last one out
            // while Dispose is pending OR detached caches are waiting to be
            // released, drain them here. Must happen under the lock so the
            // "last-out" check is atomic with the dispose action.
            DrainPendingDisposals();
        }
    }

    /// <summary>
    /// Async overload of <see cref="Predict(Tensor{T}, int, Func{Tensor{T}})"/>.
    /// Routes through <c>ICompiledPlan&lt;T&gt;.ExecuteAsync(ct)</c> on the
    /// trace-and-replay path so callers in async pipelines (web servers,
    /// diffusion denoising loops, multi-stage VLM forwards) can await
    /// completion without blocking a threadpool worker for the full plan.
    /// </summary>
    /// <remarks>
    /// <para>
    /// On the eager-fallback path (compilation disabled, plan failed to
    /// compile, or process is being torn down) this method runs the eager
    /// forward synchronously and returns a completed <see cref="System.Threading.Tasks.ValueTask{T}"/>.
    /// The async fast-path activates only when a compiled plan exists,
    /// which is the common case after the first call at a given shape.
    /// </para>
    /// <para>
    /// Cancellation: the token is honored at three points — pre-flight
    /// (before any work is queued), post-trace (after the plan compiles),
    /// and inside <c>plan.ExecuteAsync</c> (Tensors-side cooperative
    /// cancellation between steps). Cancelling between steps does not
    /// invalidate the cached plan; subsequent calls reuse it.
    /// </para>
    /// </remarks>
    public async System.Threading.Tasks.ValueTask<Tensor<T>> PredictAsync(
        Tensor<T> input,
        int structureVersion,
        Func<Tensor<T>> eagerForward,
        System.Threading.CancellationToken cancellationToken = default)
    {
        if (eagerForward is null)
            throw new ArgumentNullException(nameof(eagerForward));

        cancellationToken.ThrowIfCancellationRequested();

        CompiledModelCache<T>? cache;
        bool fallToEager;
        lock (_sync)
        {
            fallToEager = _disposed || _disposeRequested
                || !TensorCodecOptions.Current.EnableCompilation;

            if (!fallToEager)
            {
                if (_cache is not null && structureVersion != _lastCompiledVersion)
                {
                    (_pendingDisposeCaches ??= new List<CompiledModelCache<T>>()).Add(_cache);
                    _cache = new CompiledModelCache<T>();
                    _lastCompiledVersion = structureVersion;
                    // Hot plan was bound to the old cache; clear so the next
                    // call falls into the slow path which will rebuild against
                    // the fresh cache + capture a new hot plan. Also clear the
                    // eager-only verdict cache so a fresh graph gets to attempt
                    // compile again — the prior failure was for a now-stale
                    // layer structure.
                    _hotPlan = null;
                    _eagerOnlyShapeKeys = null;
                }
                cache = _cache ??= new CompiledModelCache<T>();
                _activeCalls++;
            }
            else
            {
                cache = null;
            }
        }

        if (fallToEager || cache is null)
            return eagerForward();

        try
        {
            try
            {
                var concreteShape = (int[])input._shape.Clone();

                // Disk-plan hit: route the hit through ExecuteAsync instead
                // of the sync Execute that the matching sync Predict path
                // uses. Otherwise the caller blocks on plan.Execute() and
                // bypasses cooperative cancellation between steps.
                if (TryGetDiskCachedPlan(concreteShape, structureVersion, out var preloadedPlan))
                {
                    _lastCompiledVersion = structureVersion;
                    cancellationToken.ThrowIfCancellationRequested();
                    // VALUE-STABLE REPLAY: rebind the current call's data into
                    // the cached plan's input buffer. See the matching
                    // comment in the sync Predict path.
                    preloadedPlan!.SetInputs(new[] { input });
                    return await preloadedPlan.ExecuteAsync(cancellationToken).ConfigureAwait(false);
                }

                var symbolicShape = BuildSymbolicShape(concreteShape, _shapeMode);
                var plan = symbolicShape is null
                    ? cache.GetOrCompileInference(concreteShape, () => eagerForward())
                    : cache.GetOrCompileInference(concreteShape, () => eagerForward(), symbolicShape);

                MaybeSavePlanToDisk(plan, concreteShape, structureVersion);
                _lastCompiledVersion = structureVersion;

                cancellationToken.ThrowIfCancellationRequested();

                // Real async path — defers to the engine-stream-aware
                // ExecuteAsync added in Tensors PR #298. CPU engines have
                // a fast-path that completes synchronously; GPU engines
                // wrap the CUDA stream / IGpuStream completion event as
                // a polling Task that doesn't block a worker thread.
                // VALUE-STABLE REPLAY: see comment in sync Predict path.
                if (plan is ICompiledPlan<T> rebindable)
                {
                    rebindable.SetInputs(new[] { input });
                }
                return await plan.ExecuteAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (System.OperationCanceledException)
            {
                // Cooperative cancellation propagates without falling back
                // to the eager path — eager would just re-do the whole
                // forward, which the caller didn't ask for.
                throw;
            }
            catch (Exception ex) when (
                ex is not OutOfMemoryException &&
                ex is not AccessViolationException &&
                ex is not StackOverflowException &&
                ex is not BadImageFormatException &&
                ex is not InvalidProgramException &&
                ex is not System.Threading.ThreadAbortException &&
                ex is not AppDomainUnloadedException &&
                ex is not CannotUnloadAppDomainException)
            {
                // Invalidate the cache before falling back so a poisoned
                // plan isn't reused on subsequent calls — mirrors the
                // sync Predict path which invalidates under lock here.
                lock (_sync)
                {
                    _cache?.Invalidate();
                    _lastCompiledVersion = -1;
                }
                System.Diagnostics.Trace.TraceWarning(
                    $"CompiledModelHost.PredictAsync falling back to eager after compile/replay failure: " +
                    $"{ex.ToString()}");
                return eagerForward();
            }
        }
        finally
        {
            DrainPendingDisposals();
        }
    }

    private void DrainPendingDisposals()
    {
        List<CompiledModelCache<T>>? toDispose = null;
        CompiledModelCache<T>? ownedCacheToDispose = null;
        lock (_sync)
        {
            _activeCalls--;
            if (_activeCalls > 0)
                return;

            if (_pendingDisposeCaches is not null)
            {
                toDispose = _pendingDisposeCaches;
                _pendingDisposeCaches = null;
            }

            if (_disposeRequested && !_disposed)
            {
                _disposed = true;
                ownedCacheToDispose = _cache;
                _cache = null;
            }
        }

        // Dispose outside the lock to avoid blocking other threads while
        // pooled-buffer release runs. Swallow disposal exceptions so a
        // misbehaving plan doesn't abort the whole cleanup chain.
        if (toDispose is not null)
        {
            foreach (var c in toDispose)
            {
                try { c.Dispose(); }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.TraceWarning(
                        $"CompiledModelHost: detached cache Dispose failed: {ex.GetType().Name}: {ex.Message}");
                }
            }
        }
        if (ownedCacheToDispose is not null)
        {
            try { ownedCacheToDispose.Dispose(); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"CompiledModelHost: owning cache Dispose failed: {ex.GetType().Name}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Drops all cached plans. Call when the layer graph has changed in a way
    /// not captured by the <c>structureVersion</c> parameter (e.g., ambient
    /// mode changes that affect recorded ops).
    /// </summary>
    public void Invalidate()
    {
        CompiledModelCache<T>? detached = null;
        lock (_sync)
        {
            // Detach the current cache so new Predict calls start with a
            // fresh compile. If nothing is in-flight, dispose the detached
            // cache here; otherwise park it on the pending-dispose queue
            // and let the last in-flight caller release it.
            if (_cache is not null)
            {
                if (_activeCalls > 0)
                {
                    (_pendingDisposeCaches ??= new List<CompiledModelCache<T>>()).Add(_cache);
                }
                else
                {
                    detached = _cache;
                }
                _cache = null;
            }
            _lastCompiledVersion = -1;
        }
        // Dispose outside the lock.
        if (detached is not null)
        {
            try { detached.Dispose(); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"CompiledModelHost.Invalidate: cache Dispose failed: {ex.GetType().Name}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Releases the compile cache. After Dispose, <see cref="Predict"/>
    /// short-circuits directly to the eager path and never allocates a new
    /// cache. If any <see cref="Predict"/> calls are still in flight when
    /// Dispose is called, actual cache disposal is deferred until the last
    /// in-flight call completes so we never tear down a cache out from
    /// under a live compile/replay.
    /// </summary>
    public void Dispose()
    {
        CompiledModelCache<T>? immediate = null;
        List<CompiledModelCache<T>>? pendingImmediate = null;
        lock (_sync)
        {
            if (_disposed || _disposeRequested) return;
            _disposeRequested = true;
            // Hot plan references a cached plan that is about to be disposed
            // (either now if _activeCalls==0, or by the last in-flight caller).
            // Clear immediately so any concurrent fast-path read sees null and
            // drops to the slow path which observes _disposeRequested and
            // falls back to eager.
            _hotPlan = null;

            if (_activeCalls == 0)
            {
                // No in-flight calls — we can dispose the owning cache
                // plus any previously-deferred detached caches right now.
                _disposed = true;
                immediate = _cache;
                _cache = null;
                pendingImmediate = _pendingDisposeCaches;
                _pendingDisposeCaches = null;
            }
            // else: DrainPendingDisposals (called from the last Predict
            // finally block) will perform the actual disposal.
        }

        if (pendingImmediate is not null)
        {
            foreach (var c in pendingImmediate)
            {
                try { c.Dispose(); }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.TraceWarning(
                        $"CompiledModelHost.Dispose: detached cache Dispose failed: {ex.GetType().Name}: {ex.Message}");
                }
            }
        }
        if (immediate is not null)
        {
            try { immediate.Dispose(); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"CompiledModelHost.Dispose: owning cache Dispose failed: {ex.GetType().Name}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Checks <see cref="PlanCache.Current"/> for a previously-saved plan matching
    /// the current shape + structure version, and executes it if found. Returns
    /// true when the disk-cached plan was used (caller should skip compilation).
    /// </summary>
    private bool TryUseDiskCachedPlan(
        int[] concreteShape,
        int structureVersion,
        Tensor<T> currentInput,
        out Tensor<T> result)
    {
        if (TryGetDiskCachedPlan(concreteShape, structureVersion, out var plan))
        {
            // VALUE-STABLE REPLAY: see comment in Predict above.
            plan!.SetInputs(new[] { currentInput });
            result = plan.Execute();
            // Capture this plan as the hot-path target too — the disk-cache
            // early-return path is also amortisable across calls. Without
            // this, every call falls through TryGetDiskCachedPlan's
            // lock+dict-lookup + SetInputs + Execute, ~10× slower than the
            // hot-path's direct SetInputs + Execute on a captured pointer.
            if (TryComputeShapeKey(concreteShape, out long hotKey))
            {
                _hotShapeKey = hotKey;
                _hotVersion = structureVersion;
                _hotPlan = plan;
            }
            return true;
        }
        result = null!;
        return false;
    }

    /// <summary>
    /// Async-friendly counterpart of <see cref="TryUseDiskCachedPlan"/>: returns
    /// the loaded <see cref="ICompiledPlan{T}"/> instead of synchronously executing
    /// it. Used by <see cref="PredictAsync"/> so the disk-plan-hit path can route
    /// through <c>plan.ExecuteAsync(ct)</c> with cooperative cancellation instead
    /// of blocking on a sync <c>Execute</c>.
    /// </summary>
    private bool TryGetDiskCachedPlan(
        int[] concreteShape,
        int structureVersion,
        out ICompiledPlan<T>? plan)
    {
        plan = null;
        var planCache = PlanCache.Current;
        if (planCache is null || _modelIdentity is null)
        {
            return false;
        }

        var shapeKey = ComputeShapeKey(concreteShape, structureVersion);

        // Preloaded-in-memory hit: skip disk entirely.
        Dictionary<string, ICompiledPlan<T>>? preloaded;
        lock (_sync)
        {
            preloaded = _preloadedPlans;
        }
        if (preloaded is not null && preloaded.TryGetValue(shapeKey, out var cachedPlan))
        {
            if (cachedPlan.IsValid(concreteShape))
            {
                plan = cachedPlan;
                return true;
            }
        }

        // First-time-miss: check disk once per (shape, version).
        bool shouldCheckDisk;
        lock (_sync)
        {
            _diskCheckedShapes ??= new HashSet<string>();
            shouldCheckDisk = _diskCheckedShapes.Add(shapeKey);
        }
        if (!shouldCheckDisk)
        {
            return false;
        }

        try
        {
            var loaded = planCache.TryLoadInferenceAsync<T>(
                _modelIdentity, structureVersion, concreteShape, AiDotNetEngine.Current)
                .GetAwaiter().GetResult();

            if (loaded is null || !loaded.IsValid(concreteShape))
            {
                return false;
            }

            lock (_sync)
            {
                (_preloadedPlans ??= new Dictionary<string, ICompiledPlan<T>>())[shapeKey] = loaded;
            }

            plan = loaded;
            return true;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"CompiledModelHost: disk-cached plan load failed: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Fires a fire-and-forget disk save so subsequent process starts skip compile.
    /// Errors are swallowed — disk caching is an optimization, not a correctness
    /// dependency.
    /// </summary>
    private void MaybeSavePlanToDisk(ICompiledPlan<T> plan, int[] concreteShape, int structureVersion)
    {
        var planCache = PlanCache.Current;
        if (planCache is null || _modelIdentity is null)
        {
            return;
        }

        var shapeKey = ComputeShapeKey(concreteShape, structureVersion);
        lock (_sync)
        {
            // Remember the plan in-memory too; avoids re-saving on next call.
            (_preloadedPlans ??= new Dictionary<string, ICompiledPlan<T>>())[shapeKey] = plan;
        }

        var identity = _modelIdentity;
        _ = Task.Run(async () =>
        {
            try
            {
                await planCache.SaveInferenceAsync(plan, identity, structureVersion, concreteShape)
                    .ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"CompiledModelHost: background plan save failed: {ex.GetType().Name}: {ex.Message}");
            }
        });
    }

    private static string ComputeShapeKey(int[] shape, int structureVersion)
    {
        var sb = new System.Text.StringBuilder(16 + shape.Length * 4);
        sb.Append('v').Append(structureVersion).Append('_');
        for (int i = 0; i < shape.Length; i++)
        {
            if (i > 0) sb.Append('x');
            sb.Append(shape[i]);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Allocation-free shape hash for the hot-path replay check. Packs up to
    /// four 16-bit dim values into a long so equality check is a single
    /// compare. Returns false for ranks/dim sizes outside this window, which
    /// drops the caller to the slow path (correct but slower). For typical
    /// Transformer inference (rank 2/3, dims ≤ 4096) the hot-path covers
    /// nearly all calls. Independent of structure version — that's compared
    /// separately via <c>_hotVersion</c>.
    /// </summary>
    private static bool TryComputeShapeKey(int[] shape, out long key)
    {
        key = 0;
        if (shape is null || shape.Length == 0 || shape.Length > 4)
            return false;
        for (int i = 0; i < shape.Length; i++)
        {
            int dim = shape[i];
            if (dim < 0 || dim > 0xFFFF) return false;
            key |= ((long)dim) << (i * 16);
        }
        // Encode rank in the top byte so [1,4,1] never collides with [1,4].
        key |= ((long)shape.Length) << 56;
        return true;
    }

    /// <summary>
    /// Translates <see cref="SymbolicShapeMode"/> + concrete shape into a Tensors
    /// <see cref="SymbolicShape"/> (or null to fall back to the purely concrete
    /// overload when the rank is too small for the requested mode).
    /// </summary>
    private static SymbolicShape? BuildSymbolicShape(int[] shape, SymbolicShapeMode mode)
    {
        switch (mode)
        {
            case SymbolicShapeMode.Static:
                return null;
            case SymbolicShapeMode.BatchDynamic:
                return shape.Length >= 2 ? SymbolicShape.BatchDynamic(shape) : null;
            case SymbolicShapeMode.BatchAndSeqDynamic:
                return shape.Length >= 3 ? SymbolicShape.BatchAndSeqDynamic(shape) : null;
            case SymbolicShapeMode.AllDynamic:
                return SymbolicShape.AllDynamic(shape);
            default:
                return null;
        }
    }
}

/// <summary>
/// Strategy for how <see cref="CompiledModelHost{T}"/> keys the compile cache. Dynamic
/// dims let one compiled plan serve multiple concrete shapes — essential for bursty
/// inference traffic where every request has a different batch size.
/// </summary>
public enum SymbolicShapeMode
{
    /// <summary>Every dim treated as static. Recompile on any shape change.</summary>
    Static,
    /// <summary>Dim 0 (batch) dynamic; all others static. PyTorch-default behavior.</summary>
    BatchDynamic,
    /// <summary>Dims 0 (batch) and 1 (seq-len) dynamic. For transformer-style inputs.</summary>
    BatchAndSeqDynamic,
    /// <summary>Every dim dynamic. Maximum reuse; slight dispatch overhead on replay.</summary>
    AllDynamic,
}
