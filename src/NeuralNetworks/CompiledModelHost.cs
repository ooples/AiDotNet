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
                // Preserve the Tensor<T> return from eagerForward so the compile
                // pass can explicitly identify the output tensor. Discarding the
                // return (previously `() => { eagerForward(); }`) can select a
                // different GetOrCompileInference overload / leave the output
                // tensor ambiguous to the tracer, producing wrong outputs on
                // replay. Keep the expression-bodied lambda so the value threads
                // through unchanged.
                var plan = cache.GetOrCompileInference(
                    (int[])input._shape.Clone(),
                    () => eagerForward());

                // Safe to write _lastCompiledVersion outside the lock — int writes
                // are atomic in .NET, and the value is only used as a hint by the
                // next Predict's version-mismatch check (which itself runs under
                // the lock and will recover from any race).
                _lastCompiledVersion = structureVersion;
                return plan.Execute();
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
}
