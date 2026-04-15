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
        CompiledModelCache<T>? cache;
        bool fallToEager;
        lock (_sync)
        {
            fallToEager = _disposed || !TensorCodecOptions.Current.EnableCompilation;

            if (!fallToEager)
            {
                // Layer graph changed since last compile — stale plans would
                // replay against the old tensor references. Drop them before
                // compiling again.
                if (_cache is not null && structureVersion != _lastCompiledVersion)
                {
                    _cache.Invalidate();
                    _lastCompiledVersion = structureVersion;
                }
                cache = _cache ??= new CompiledModelCache<T>();
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
            var plan = cache.GetOrCompileInference(
                (int[])input._shape.Clone(),
                () => { eagerForward(); });

            // Safe to write _lastCompiledVersion outside the lock — int writes
            // are atomic in .NET, and the value is only used as a hint by the
            // next Predict's version-mismatch check (which itself runs under
            // the lock and will recover from any race).
            _lastCompiledVersion = structureVersion;
            return plan.Execute();
        }
        catch (Exception ex)
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
            // signal that compiled inference stopped working.
            System.Diagnostics.Trace.TraceWarning(
                $"CompiledModelHost falling back to eager after compile/replay failure: " +
                $"{ex.GetType().Name}: {ex.Message}");
            return eagerForward();
        }
    }

    /// <summary>
    /// Drops all cached plans. Call when the layer graph has changed in a way
    /// not captured by the <c>structureVersion</c> parameter (e.g., ambient
    /// mode changes that affect recorded ops).
    /// </summary>
    public void Invalidate()
    {
        lock (_sync)
        {
            _cache?.Invalidate();
            _lastCompiledVersion = -1;
        }
    }

    /// <summary>
    /// Releases the compile cache. After Dispose, <see cref="Predict"/>
    /// short-circuits directly to the eager path and never allocates a new
    /// cache.
    /// </summary>
    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            _cache?.Dispose();
            _cache = null;
        }
    }
}
