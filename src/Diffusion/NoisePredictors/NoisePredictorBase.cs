using System.Linq;
using AiDotNet.Autodiff;
using AiDotNet.Engines;
using AiDotNet.Extensions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Base class for noise prediction networks used in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides common functionality for all noise predictors,
/// including timestep embedding, parameter management, serialization, and gradient computation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation that all noise prediction networks build upon.
/// Noise predictors are the neural networks at the heart of diffusion models that learn to
/// predict what noise was added to a sample. Different architectures (U-Net, DiT, etc.)
/// extend this base class.
/// </para>
/// </remarks>
public abstract class NoisePredictorBase<T> : INoisePredictor<T>, IModelShape, IDisposable
{
    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Composable inference-compilation helper. Concrete predictors route their
    /// <see cref="PredictNoiseWithEmbedding"/> through <see cref="PredictCompiled"/>
    /// to get compiled-plan replay across the 50+ denoising steps in the diffusion
    /// loop. First call traces, subsequent calls replay. Falls back to eager when
    /// compilation is disabled or fails.
    /// </summary>
    private readonly AiDotNet.NeuralNetworks.CompiledModelHost<T> _compileHost;

    /// <summary>
    /// Verify-then-trust gate (#1622 L3b) shared with <c>NeuralNetworkBase</c>: the per-step
    /// <see cref="PredictCompiled"/> call a diffusion denoising loop makes runs eager once per shape to
    /// confirm the compiled plan matches, then replays the trusted plan for the remaining 50+ steps —
    /// the dominant inference cost of a foundation-scale diffusion model. Output stays numerically
    /// identical to eager (rejected/unverified shapes fall back to eager). Process-wide opt-out via
    /// <c>AIDOTNET_DISABLE_AUTO_COMPILE=1</c>.
    /// </summary>
    private readonly AiDotNet.NeuralNetworks.VerifiedInferenceGate<T> _inferenceGate = new();

    private static readonly bool s_autoCompileDisabled =
        string.Equals(System.Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_AUTO_COMPILE"), "1", System.StringComparison.Ordinal);

    /// <summary>
    /// Monotonic layer-graph version. Concrete predictors bump this via
    /// <see cref="InvalidateCompiledPlans"/> after lazy-init expands tensor shapes
    /// or after <see cref="SetParameters"/> swaps weights. The host drops stale
    /// plans automatically when the version changes.
    /// </summary>
    private int _layerStructureVersion;

    private bool _disposed;

    /// <summary>
    /// Throws <see cref="ObjectDisposedException"/> when the predictor has already
    /// been disposed. Public entry points that touch <see cref="_compileHost"/>,
    /// the timestep-embedding cache, or the layer graph must call this first so
    /// post-Dispose use surfaces a predictable error instead of arbitrary downstream
    /// failures from torn-down resources.
    /// </summary>
    protected void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName);
    }

    /// <summary>
    /// Concrete predictors can override to expose their <see cref="ILayer{T}"/>
    /// instances for (a) Dispose cascade — pool-rented weight tensors return to
    /// the allocator, and (b) future compilation features (plan serialization,
    /// CUDA Graph capture) that need visibility into the layer graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default behavior</b>: returns an empty enumeration. Reflection-based discovery
    /// is NOT the default because it would dispose layers the predictor doesn't own
    /// (e.g., injected/shared cross-attention layers from a shared encoder, a VAE
    /// reference passed in by the caller). <see cref="DisposeOnceGuard"/> only protects
    /// against double-dispose; it does not stop the first predictor from tearing down
    /// a dependency another model still needs. Ownership is expressed by what a
    /// predictor explicitly enumerates, not by what reflection happens to find.
    /// </para>
    /// <para>
    /// Concrete predictors that own their layers and want Dispose-time cleanup must
    /// opt in by overriding this method. They have two options:
    /// </para>
    /// <list type="number">
    /// <item>Override and yield specific field references explicitly
    /// (recommended — zero reflection cost, explicit ownership).</item>
    /// <item>Override to call <c>ReflectInstanceLayers(this)</c> when the predictor
    /// owns every reachable layer. <see cref="ReflectInstanceLayers"/> walks fields
    /// plus <see cref="System.Collections.IEnumerable"/> and
    /// <see cref="System.Collections.IDictionary"/> elements that implement
    /// <see cref="ILayer{T}"/>, and recurses into wrapper objects (DiTBlock,
    /// ResidualStage, etc.) with a cycle guard.</item>
    /// </list>
    /// </remarks>
    protected virtual IEnumerable<ILayer<T>> EnumerateLayers() =>
        Enumerable.Empty<ILayer<T>>();

    /// <summary>
    /// Walks an object's instance fields and yields anything that implements
    /// <see cref="ILayer{T}"/>, recursively descending into owned wrapper objects
    /// (e.g. <c>DiTBlock</c>, <c>UNetEncoderStage</c>) so block-heavy predictors get
    /// correct cleanup without each block needing to manually re-implement
    /// <c>EnumerateLayers</c>. The cycle guard via <c>visited</c> prevents infinite
    /// recursion on graphs that share sublayers.
    /// </summary>
    protected static IEnumerable<ILayer<T>> ReflectInstanceLayers(object root)
    {
        var visited = new HashSet<object>(AiDotNet.Helpers.TensorReferenceComparer<object>.Instance);
        var stack = new Stack<object>();
        if (!visited.Add(root)) yield break;
        stack.Push(root);

        while (stack.Count > 0)
        {
            var current = stack.Pop();
            var currentType = current.GetType();
            for (var t = currentType; t != null && t != typeof(object); t = t.BaseType)
            {
                // Cached, pre-filtered reference-type declared fields for this type — see
                // GetDeclaredWalkableFields. Skips the per-field value-type/string probe
                // (RuntimeType.IsPrimitiveImpl) that dominated the forward (#1646).
                foreach (var field in GetDeclaredWalkableFields(t))
                {
                    object? value;
                    try { value = field.GetValue(current); }
                    catch (Exception ex)
                    {
                        // Trace rather than silently skip — without this a private
                        // field whose getter throws would leak its layer's resources
                        // without any diagnostic trail at Dispose time.
                        System.Diagnostics.Trace.TraceWarning(
                            $"NoisePredictorBase.Dispose: skipping field '{field.Name}' " +
                            $"on {t.Name} due to reflection read failure: {ex.GetType().Name}: {ex.Message}");
                        continue;
                    }
                    if (value is null || !visited.Add(value)) continue;

                    if (value is ILayer<T> layer)
                    {
                        yield return layer;
                        // Layers may also own sublayers via reflectable fields
                        // (composite layers, attention with internal projections);
                        // descend into them too.
                        stack.Push(value);
                    }
                    else if (value is System.Collections.IDictionary dictionary)
                    {
                        // Dictionary<K, V>.GetEnumerator yields KeyValuePair<K,V>,
                        // not the values — so the generic IEnumerable branch below
                        // would MISS layers held in the values slot. Handle
                        // IDictionary explicitly so Dictionary<K, ILayer<T>> is
                        // disposed correctly.
                        foreach (System.Collections.DictionaryEntry entry in dictionary)
                        {
                            if (entry.Value is null || !visited.Add(entry.Value)) continue;
                            if (entry.Value is ILayer<T> nestedLayer)
                            {
                                yield return nestedLayer;
                                stack.Push(entry.Value);
                            }
                            else if (IsWalkableWrapper(entry.Value.GetType()))
                            {
                                stack.Push(entry.Value);
                            }
                        }
                    }
                    else if (value is System.Collections.IEnumerable enumerable && value is not string)
                    {
                        // We MUST enumerate the whole sequence: enumerating a streaming-placeholder
                        // weight tensor (Tensor<T> as IEnumerable<float>) rehydrates it, and the
                        // forward relies on that side effect (NoisePredictorWeightStreamingTests).
                        // But value-type elements — the millions of boxed floats in a Tensor<T> /
                        // float[] / double[] buffer — can never be, or own, a layer, so skip the
                        // `visited` bookkeeping for them: hashing every boxed element
                        // (RuntimeHelpers.GetHashCode) was the dominant walk cost (#1646). `continue`
                        // still advances the enumerator, so rehydration is preserved.
                        foreach (var item in enumerable)
                        {
                            if (item is null || item is ValueType) continue;
                            if (!visited.Add(item)) continue;
                            if (item is ILayer<T> nestedLayer)
                            {
                                yield return nestedLayer;
                                stack.Push(item);
                            }
                            else if (IsWalkableWrapper(item.GetType()))
                            {
                                // Recurse into wrapper objects (DiTBlock, ResidualStage, etc.)
                                // that hold layer fields but are not themselves layers.
                                stack.Push(item);
                            }
                        }
                    }
                    else if (IsWalkableWrapper(value.GetType()))
                    {
                        // Recurse into single owned wrapper objects (e.g. an Encoder
                        // composite that holds layers in its own fields).
                        stack.Push(value);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Returns true for reference types that look like AiDotNet-internal wrapper objects
    /// worth descending into during layer enumeration. We exclude system types and
    /// anything explicitly opt-out (e.g. tensor / vector containers) to keep the walk
    /// bounded; wrappers inside this assembly's namespaces are fair game.
    /// </summary>
    private static bool IsWalkableWrapper(Type type)
        => s_walkableWrapperCache.GetOrAdd(type, static t =>
        {
            if (t.IsPrimitive || t.IsEnum) return false;
            var ns = t.Namespace ?? string.Empty;
            if (ns.StartsWith("System", StringComparison.Ordinal)) return false;
            // Avoid recursing into low-level numeric containers — their fields are arrays
            // of T, not layers, and walking them adds noise for no benefit.
            if (ns.StartsWith("AiDotNet.Tensors", StringComparison.Ordinal)) return false;
            if (t.Name == "Vector`1" || t.Name == "Matrix`1" || t.Name == "Tensor`1") return false;
            return true;
        });

    // Per-type reflection-metadata caches. GetFields and "is this a walkable wrapper" are
    // pure functions of the Type, so resolve each once across every instance and every
    // forward instead of re-probing runtime type handles on the hot streaming path (#1646).
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, System.Reflection.FieldInfo[]> s_walkableFieldCache = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, bool> s_walkableWrapperCache = new();

    /// <summary>
    /// Reference-type, non-string instance fields declared directly on <paramref name="declaringType"/>.
    /// Value-type and string fields are filtered out at cache-build time — they can never be,
    /// or own, an <see cref="ILayer{T}"/> — so the walk never re-checks them. The walk calls
    /// this once per type in the base chain (the caller iterates <c>BaseType</c>).
    /// </summary>
    private static System.Reflection.FieldInfo[] GetDeclaredWalkableFields(Type declaringType)
        => s_walkableFieldCache.GetOrAdd(declaringType, static t =>
        {
            const System.Reflection.BindingFlags flags =
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.DeclaredOnly;
            var fields = new List<System.Reflection.FieldInfo>();
            foreach (var f in t.GetFields(flags))
            {
                var ft = f.FieldType;
                if (ft.IsValueType || ft == typeof(string)) continue;
                fields.Add(f);
            }
            return fields.ToArray();
        });

    /// <summary>
    /// Runs <paramref name="eagerFallback"/> under the compile host — traces on
    /// first call at each input shape, replays the compiled plan on subsequent
    /// calls. Concrete predictors call this from hot forward paths (e.g., the
    /// per-step <see cref="Forward"/> during the diffusion denoising loop) to
    /// get near-zero-overhead replay after the first trace.
    /// </summary>
    /// <param name="input">Shape key for the compile cache.</param>
    /// <param name="eagerFallback">The eager forward pass (traced, replayed, or fallback).</param>
    protected Tensor<T> PredictCompiled(Tensor<T> input, Func<Tensor<T>> eagerFallback)
    {
        // Direct compile host when the verify gate is opted out.
        if (s_autoCompileDisabled)
            return _compileHost.Predict(input, _layerStructureVersion, eagerFallback);

        // #1622 L3b: front the compile host with the verify-then-trust gate so a compiled plan is
        // adopted for a shape only after it matches the eager forward — then replayed for the rest of
        // the denoising loop. Output stays numerically identical to eager (rejected shapes stay eager).
        return _inferenceGate.Run(
            input,
            _layerStructureVersion,
            eager: eagerFallback,
            compiled: () => _compileHost.Predict(input, _layerStructureVersion, eagerFallback),
            onDecision: (enabled, reason) => AiDotNet.Helpers.InferenceDiagnostics.RecordDecision(
                area: "NoisePredictorBase", feature: "AutoCompiledInference", enabled: enabled, reason: reason));
    }

    /// <summary>
    /// Async overload of <see cref="PredictCompiled"/> — routes through
    /// <see cref="CompiledModelHost{T}.PredictAsync"/> so the compiled plan's
    /// <c>ExecuteAsync</c> path is taken. CPU engines complete synchronously
    /// on the same thread (no work transfer cost); GPU engines wrap their
    /// stream completion event as a polling task that does not block a
    /// threadpool worker for the GPU's tail kernels. Concrete noise
    /// predictors call this from <see cref="PredictNoiseAsync"/> overrides
    /// to expose the same trace-and-replay benefit on their async surface.
    /// </summary>
    protected System.Threading.Tasks.ValueTask<Tensor<T>> PredictCompiledAsync(
        Tensor<T> input,
        Func<Tensor<T>> eagerFallback,
        System.Threading.CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        return _compileHost.PredictAsync(input, _layerStructureVersion, eagerFallback, cancellationToken);
    }

    /// <summary>
    /// Bump to signal the layer graph has changed — lazy init expanded a tensor,
    /// weights were reassigned, a sub-layer was replaced. The compile host drops
    /// any plan captured against the prior graph on the next <see cref="PredictCompiled"/>.
    /// </summary>
    protected void InvalidateCompiledPlans()
    {
        _layerStructureVersion++;
        // Drop the cache eagerly rather than wait for the next PredictCompiled
        // to detect the version mismatch. This releases captured tensor buffers
        // immediately — important when the caller is invalidating because the
        // old graph holds memory we want to reclaim now.
        _compileHost.Invalidate();
        // The gate's verdicts/memo are version-scoped (stale entries are ignored after the bump), but
        // clear eagerly so the memo's retained input/output clones are released now too.
        _inferenceGate.Clear();
    }

    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator for initialization and stochastic operations.
    /// </summary>
    protected Random RandomGenerator;

    /// <summary>
    /// The loss function used for training (typically MSE for noise prediction).
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Active feature indices used by the model.
    /// </summary>
    private HashSet<int> _activeFeatureIndices = new HashSet<int>();

    /// <inheritdoc />
    public abstract int InputChannels { get; }

    /// <inheritdoc />
    public abstract int OutputChannels { get; }

    /// <inheritdoc />
    public abstract int BaseChannels { get; }

    /// <inheritdoc />
    public abstract int TimeEmbeddingDim { get; }

    /// <inheritdoc />
    public abstract long ParameterCount { get; }

    /// <summary>
    /// Streams the predictor's trainable weight tensors per-tensor without
    /// materialising a flat aggregate, mirroring PyTorch's
    /// <c>nn.Module.parameters()</c> generator pattern. Default
    /// implementation walks <see cref="Layers"/> and yields each layer's
    /// own chunks; subclasses with non-Layer-based weight storage
    /// (registered tensors, embedded sub-models) override to surface
    /// those too. Foundation-scale predictors (DiT-XL/2 with HiddenDim
    /// 3072 and 48 layers) overflow <see cref="int.MaxValue"/> in the
    /// aggregate; callers walking these chunks accumulate length into a
    /// <see cref="long"/>.
    /// </summary>
    public virtual IEnumerable<Tensor<T>> GetParameterChunks()
    {
        // Default: single chunk wrapping GetParameters(). Concrete predictors
        // with tractable per-block weight stores SHOULD override to yield
        // per-tensor chunks so foundation-scale models avoid materialising a
        // flat aggregate that overflows Vector.Length's int contract. The
        // single-chunk default keeps ParameterCount == sum-of-chunk-lengths
        // exact, which is the contract test (#1237's acceptance criterion)
        // depends on.
        var p = GetParameters();
        if (p.Length == 0) yield break;
        yield return new Tensor<T>(new[] { p.Length }, p);
    }

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;
    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;


    /// <inheritdoc />
    public abstract bool SupportsCFG { get; }

    /// <inheritdoc />
    public abstract bool SupportsCrossAttention { get; }

    /// <inheritdoc />
    public abstract int ContextDimension { get; }

    /// <inheritdoc />
    public ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Initializes a new instance of the NoisePredictorBase class.
    /// </summary>
    /// <param name="lossFunction">Optional custom loss function. Defaults to MSE.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected NoisePredictorBase(ILossFunction<T>? lossFunction = null, int? seed = null)
    {
        LossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        RandomGenerator = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        _compileHost = new AiDotNet.NeuralNetworks.CompiledModelHost<T>(
            shapeMode: AiDotNet.NeuralNetworks.SymbolicShapeMode.BatchDynamic,
            modelIdentity: GetType().FullName ?? GetType().Name);
    }

    #region Transparent Weight Streaming (Tensors #602 / issue #430)

    // 0 = not engaged, 1 = engaged. Int (not bool) so the engage transition can be claimed
    // atomically via Interlocked.CompareExchange — concurrent callers on the same instance
    // then can't both reconfigure the process-global WeightRegistry.
    private int _streamingEngaged;

    // 0 = no successful weight registration yet, 1 = resolved lazy weights have
    // been registered with the streaming pool at least once.
    private int _streamingWeightsRegistered;

    /// <summary>
    /// Parameter-count threshold above which <see cref="MaybeEngageWeightStreaming"/>
    /// engages disk-backed weight streaming for the denoising forward loop. 500 M
    /// is the same memory-pressure inflection point <c>NeuralNetworkBase</c> uses.
    /// </summary>
    private const long DefaultStreamingThresholdParams = 500_000_000L;

    /// <summary>
    /// Test/diagnostic override for <see cref="DefaultStreamingThresholdParams"/> so
    /// controlled-scale tests can exercise the streaming path without a
    /// foundation-scale model. <c>null</c> ⇒ use the default. Process-global.
    /// </summary>
    internal static long? StreamingThresholdOverride { get; set; }

    /// <summary>
    /// Test/diagnostic override for the streaming pool's resident-byte cap so a
    /// small model can be forced to page (the auto-cap is sized for foundation
    /// models). <c>null</c> ⇒ use <see cref="ComputeResidentCapBytes"/>.
    /// </summary>
    internal static long? StreamingResidentCapOverride { get; set; }

    /// <summary>
    /// Engages transparent weight streaming for this predictor when it is large
    /// enough to pressure host RAM. Idempotent — runs at most once per instance.
    /// <para>
    /// When engaged it configures the process-wide <see cref="WeightRegistry"/> for
    /// transparent auto-eviction and flags every owned layer to allocate its lazy
    /// weights through the disk-backed streaming pool. Combined with
    /// <see cref="RegisterResolvedStreamingWeights"/> (called after the first
    /// forward resolves weights), the multi-step denoising loop then keeps only a
    /// bounded resident set: each weight auto-rehydrates on access and the
    /// symmetric owner-drop evicts cold ones.
    /// </para>
    /// <para>
    /// <b>Safety guard.</b> The registry is process-global and
    /// <see cref="WeightRegistry.Configure"/> throws on a pool that already has
    /// live handles. So this only engages when the registry is empty — if another
    /// model currently owns a streaming session, this predictor runs resident
    /// rather than clobbering it. The common single-model inference path sees an
    /// empty registry.
    /// </para>
    /// </summary>
    protected void MaybeEngageWeightStreaming()
    {
        if (System.Threading.Volatile.Read(ref _streamingEngaged) != 0) return;

        long threshold = StreamingThresholdOverride ?? DefaultStreamingThresholdParams;
        if (ParameterCount <= threshold) return;

        // Don't reconfigure (or clobber) a registry another model is using.
        if (WeightRegistry.GetStreamingReport().RegisteredEntryCount > 0) return;

        // Atomically claim engagement: if a concurrent call on this same instance already
        // won the race, that thread owns the WeightRegistry.Configure below and this one
        // returns (runs resident) rather than reconfiguring the process-global registry.
        if (System.Threading.Interlocked.CompareExchange(ref _streamingEngaged, 1, 0) != 0) return;

        var offloadOptions = new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = StreamingResidentCapOverride ?? ComputeResidentCapBytes(),
            TransparentAutoEviction = true,
        };
        try
        {
            WeightRegistry.Configure(offloadOptions);
        }
        catch (InvalidOperationException)
        {
            // Configure also throws on leftover ReservedBytes — orphaned streaming
            // reservations from a PRIOR predictor whose first forward reserved weights
            // (AllocateStreaming) but never reached RegisterResolvedStreamingWeights
            // (its forward threw, or the model was disposed mid-stream). The registry
            // is process-global, so those orphans accumulate across models in one
            // process (a test shard runs many) until every subsequent streaming model
            // fails to engage here. We already confirmed RegisteredEntryCount == 0
            // above, so no LIVE model owns the pool — the leftover reservations belong
            // to a dead model and are safe to forcibly drop. Reset and reconfigure.
            WeightRegistry.Reset();
            WeightRegistry.Configure(offloadOptions);
        }

        // Walk owned layer fields recursively (including those inside the DiT block list)
        // and flag each for the streaming allocator. Iterated LAZILY: each layer is flagged
        // as the walk yields it, BEFORE the walk descends into that layer — the layer's own
        // lazy weight allocation (triggered while descending) reads UseStreamingAllocator, so
        // the flag must be set first. Do NOT materialize this into a list and flag afterwards.
        foreach (var layer in ReflectInstanceLayers(this))
        {
            if (layer is LayerBase<T> lb) lb.UseStreamingAllocator = true;
        }
    }

    /// <summary>
    /// Starts a predictor forward that may need transparent weight streaming.
    /// The returned scope registers resolved lazy weights on successful completion
    /// and releases any in-flight streaming reservations if the first forward fails.
    /// </summary>
    protected WeightStreamingForwardScope BeginWeightStreamingForward()
    {
        MaybeEngageWeightStreaming();
        return new WeightStreamingForwardScope(this);
    }

    /// <summary>
    /// Registers this predictor's now-resolved weights with the streaming pool,
    /// dropping their resident in-memory copies to disk-backed storage. No-op
    /// unless <see cref="MaybeEngageWeightStreaming"/> engaged. Called after the
    /// first forward resolves the lazy weights, so from the second forward on the
    /// transparent auto-rehydrate + owner-drop keep the resident set bounded.
    /// </summary>
    protected void RegisterResolvedStreamingWeights()
    {
        if (System.Threading.Volatile.Read(ref _streamingEngaged) == 0) return;

        // The first forward runs every layer (a diffusion predictor exercises its whole graph
        // each denoising step), so it resolves and registers all lazy weights. Skip the full
        // graph re-walk on forwards 2..N — that per-forward walk was the dominant streaming
        // cost on foundation models (#1646). Worst case a weight that only resolves on a later
        // forward stays resident rather than streamed: identical output, slightly more RAM,
        // never wrong.
        if (System.Threading.Volatile.Read(ref _streamingWeightsRegistered) == 1) return;

        foreach (var layer in ReflectInstanceLayers(this))
        {
            if (layer is not LayerBase<T> lb) continue;
            foreach (var tensor in lb.GetTrainableParameters())
            {
                // Skip placeholders (lazy weights not yet resolved) and tensors
                // already registered (idempotent across forwards).
                if (tensor is null || tensor.Length == 0 || tensor.StreamingPoolHandle >= 0) continue;
                tensor.Lifetime = WeightLifetime.Streaming;
                WeightRegistry.RegisterWeight(tensor);
            }
        }

        System.Threading.Volatile.Write(ref _streamingWeightsRegistered, 1);
    }

    private void ReleaseStreamingWeights(bool includeRegistered)
    {
        if (System.Threading.Volatile.Read(ref _streamingEngaged) == 0) return;

        foreach (var layer in ReflectInstanceLayers(this))
        {
            if (layer is not LayerBase<T> lb) continue;
            foreach (var tensor in lb.GetTrainableParameters())
            {
                if (tensor is null || tensor.Lifetime != WeightLifetime.Streaming) continue;
                if (!includeRegistered && tensor.StreamingPoolHandle >= 0) continue;

                try
                {
                    WeightRegistry.UnregisterWeight(tensor);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.TraceWarning(
                        $"NoisePredictorBase: failed to release streaming weight " +
                        $"from {layer.GetType().Name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
        }

        if (includeRegistered)
        {
            System.Threading.Volatile.Write(ref _streamingWeightsRegistered, 0);
        }
    }

    /// <summary>
    /// Per-forward streaming guard used by concrete predictors. Call
    /// <see cref="Complete"/> with the final output immediately before returning.
    /// </summary>
    protected sealed class WeightStreamingForwardScope : IDisposable
    {
        private readonly NoisePredictorBase<T> _owner;
        private bool _completed;

        internal WeightStreamingForwardScope(NoisePredictorBase<T> owner)
        {
            _owner = owner;
        }

        public Tensor<T> Complete(Tensor<T> result)
        {
            _owner.RegisterResolvedStreamingWeights();
            _completed = true;
            return result;
        }

        public void Dispose()
        {
            if (!_completed)
            {
                var firstForwardDidNotRegister =
                    System.Threading.Volatile.Read(ref _owner._streamingWeightsRegistered) == 0;
                _owner.ReleaseStreamingWeights(includeRegistered: firstForwardDidNotRegister);
            }
        }
    }

    /// <summary>
    /// Resident-byte cap for the streaming pool: half the host's available managed
    /// memory, clamped to [512 MiB, 8 GiB]. Big enough for several layers' working
    /// set (so a single op never needs more than the cap), small enough to leave
    /// headroom for activations and runtime on a 16 GB host.
    /// </summary>
    private static long ComputeResidentCapBytes()
    {
        long cap;
#if NET5_0_OR_GREATER
        long avail = 0;
        try { avail = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes; } catch { /* fall through */ }
        cap = avail > 0 ? avail / 2 : 4L * 1024 * 1024 * 1024;
#else
        // GCMemoryInfo isn't available on net471; use a conservative fixed cap.
        cap = 4L * 1024 * 1024 * 1024;
#endif
        const long min = 512L * 1024 * 1024;
        const long max = 8L * 1024 * 1024 * 1024;
        if (cap < min) cap = min;
        if (cap > max) cap = max;
        return cap;
    }

    #endregion

    #region Lazy Layer Factories

    // Large diffusion noise predictors (DiT-XL: 28 layers × 1152 hidden × 4× MLP ratio)
    // allocate ~4 GB of weight tensors at construction time when layers eagerly call
    // TensorAllocator.Rent from their ctors. That crushes CI and masks real test
    // failures behind OOM. These helpers wire every layer with
    // InitializationStrategies<T>.Lazy so weight tensors stay at size 0 until the
    // first Forward() pass actually needs them — construction becomes O(1) and
    // allocation scales with the actual tests that exercise the model.

    /// <summary>
    /// Creates a <see cref="DenseLayer{T}"/> with lazy weight allocation —
    /// weight/bias tensors stay zero-sized until the first Forward() call.
    /// Resolves shape eagerly (without consuming RNG) so <c>ParameterCount</c>,
    /// <c>GetParameters</c>, and <c>SetParameters</c> work before the first forward pass.
    /// </summary>
    protected static DenseLayer<T> LazyDense(
        int inputSize,
        int outputSize,
        IActivationFunction<T>? activation = null)
    {
        var layer = new DenseLayer<T>(outputSize, activation, InitializationStrategies<T>.Lazy);
        layer.ResolveShapesOnly(new[] { inputSize });
        return layer;
    }

    /// <summary>
    /// Creates a <see cref="DenseLayer{T}"/> with a vector activation and lazy weight
    /// allocation. Distinct name from <see cref="LazyDense(int, int, IActivationFunction{T}?)"/>
    /// because the two ctor overloads otherwise collide on overload resolution for
    /// activations that implement both scalar and vector interfaces.
    /// </summary>
    protected static DenseLayer<T> LazyDenseVec(
        int inputSize,
        int outputSize,
        IVectorActivationFunction<T> vectorActivation)
    {
        var layer = new DenseLayer<T>(outputSize, vectorActivation, InitializationStrategies<T>.Lazy);
        layer.ResolveShapesOnly(new[] { inputSize });
        return layer;
    }

    /// <summary>
    /// Creates a <see cref="LayerNormalizationLayer{T}"/> pre-resolved against
    /// <paramref name="featureSize"/> so its gamma/beta tensors are fully allocated
    /// at construction time. Use when callers iterate <c>ParameterCount</c>,
    /// <c>GetParameters</c>, <c>SetParameters</c>, or <c>Clone</c> before the first
    /// forward — a stock lazy LayerNorm would report zero parameters until forward,
    /// leading to wrong parameter vectors during initialization, serialization,
    /// or cloning.
    /// </summary>
    protected static LayerNormalizationLayer<T> EagerLayerNorm(int featureSize)
    {
        if (featureSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(featureSize),
                $"EagerLayerNorm requires a positive feature size; got {featureSize}.");
        }
        var ln = new LayerNormalizationLayer<T>();
        // Resolve from a [1, featureSize] shape — LayerNorm reads input.Shape[^1]
        // as featureSize, allocates gamma + beta, and registers them. The dummy
        // tensor allocated by ResolveFromShape is discarded; only the shape is used.
        ln.ResolveFromShape(new[] { 1, featureSize });
        return ln;
    }

    /// <summary>
    /// Creates a <see cref="ConvolutionalLayer{T}"/> with lazy weight allocation.
    /// </summary>
    protected static ConvolutionalLayer<T> LazyConv2D(
        int inputDepth,
        int inputHeight,
        int inputWidth,
        int outputDepth,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        IActivationFunction<T>? activation = null)
        => new ConvolutionalLayer<T>(
            outputDepth,
            kernelSize, stride, padding, activation, InitializationStrategies<T>.Lazy);

    /// <summary>
    /// Creates a <see cref="MultiHeadAttentionLayer{T}"/> with lazy Q/K/V/O weight
    /// allocation. DiT transformer stacks contain ~112 of these per tower at
    /// default sizes (16 heads × 4 projections × 28 blocks + 4 projections × 28
    /// cross-attention blocks) — each holding a [hidden, hidden] weight tensor.
    /// Lazy init defers the full ~1 GB of attention weights to first Forward().
    /// </summary>
    protected static MultiHeadAttentionLayer<T> LazyMHA(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        IActivationFunction<T>? activation = null)
    {
        if (sequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(sequenceLength), "sequenceLength must be positive.");
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension), "embeddingDimension must be positive.");
        if (headCount <= 0) throw new ArgumentOutOfRangeException(nameof(headCount), "headCount must be positive.");
        if (embeddingDimension % headCount != 0)
        {
            throw new ArgumentException(
                $"embeddingDimension ({embeddingDimension}) must be evenly divisible by " +
                $"headCount ({headCount}); got remainder {embeddingDimension % headCount}. " +
                "Integer division would silently produce a narrower attention layer than requested.",
                nameof(embeddingDimension));
        }

        return new MultiHeadAttentionLayer<T>(headCount, embeddingDimension / headCount,
            activation, InitializationStrategies<T>.Lazy);
    }

    /// <summary>
    /// Creates a <see cref="SelfAttentionLayer{T}"/> with lazy Q/K/V weight
    /// allocation. DiT and UViT predictors construct one of these per transformer
    /// block — 28 per DiT-XL tower, each carrying 3 × [hidden, hidden] weight
    /// tensors (~32 MB per block at hidden=1152 = ~900 MB per tower). Lazy init
    /// defers the full attention-weight budget to first Forward().
    /// </summary>
    protected static SelfAttentionLayer<T> LazySelfAttention(
        int sequenceLength,
        int embeddingDimension,
        int headCount = 8,
        IActivationFunction<T>? activation = null)
        => new SelfAttentionLayer<T>(
            sequenceLength, embeddingDimension, headCount,
            activation, InitializationStrategies<T>.Lazy);

    #endregion

    #region INoisePredictor<T> Implementation

    /// <inheritdoc />
    public abstract Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null);

    /// <summary>
    /// Async overload of <see cref="PredictNoise(Tensor{T}, int, Tensor{T})"/>.
    /// Routes the forward through <see cref="CompiledModelHost{T}.PredictAsync"/>
    /// so the underlying compiled plan's <c>ExecuteAsync</c> path is taken,
    /// allowing per-step noise predictions in a denoising loop to overlap
    /// host-side scheduler work with the backend's tail kernels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Concrete noise predictors are encouraged to override this to expose
    /// their own compile-host-aware async path. The base implementation
    /// falls back to running the sync <see cref="PredictNoise"/> on the
    /// threadpool — which is no worse than the prior all-sync surface — so
    /// that callers in async pipelines never have to add a Task.Run wrapper
    /// at the call site even before each predictor migrates.
    /// </para>
    /// </remarks>
    public virtual System.Threading.Tasks.ValueTask<Tensor<T>> PredictNoiseAsync(
        Tensor<T> noisySample,
        int timestep,
        Tensor<T>? conditioning = null,
        System.Threading.CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        // Route through the compile host's async path. The compile cache
        // is keyed on (shape, structureVersion); after the first call at
        // each shape, replays go through plan.ExecuteAsync, which on CPU
        // completes synchronously on the same thread (zero overhead vs
        // sync) and on GPU returns a ValueTask that polls the stream
        // completion event without blocking a threadpool worker.
        //
        // We close over (timestep, conditioning) for the eager-fallback
        // capture so the trace-time lambda matches what PredictNoise would
        // do for the same args; the host's shape key is just `noisySample`.
        return PredictCompiledAsync(
            noisySample,
            () => PredictNoise(noisySample, timestep, conditioning),
            cancellationToken);
    }

    /// <inheritdoc />
    /// <remarks>
    /// The base class cannot recover an integer timestep from a sinusoidal time
    /// embedding: <see cref="GetTimestepEmbedding"/> emits <c>sin(t * freq)</c> /
    /// <c>cos(t * freq)</c> values, so reading the first slot would just return a
    /// frequency-modulated sample — not the original timestep — and downstream
    /// <see cref="PredictNoise"/> would denoise against the wrong schedule. Concrete
    /// predictors that consume the embedding directly (e.g., DiT-style time-MLP
    /// conditioning) must override this; predictors that work in integer-timestep
    /// space should call <see cref="PredictNoise(Tensor{T}, int, Tensor{T})"/> with
    /// an explicit timestep instead of routing through this method.
    /// </remarks>
    public virtual Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: PredictNoiseWithEmbedding has no meaningful base " +
            "implementation. The sinusoidal time embedding produced by GetTimestepEmbedding " +
            "encodes the timestep as sin/cos features; the original integer timestep cannot " +
            "be recovered from a single embedding slot. Override this method on the concrete " +
            "predictor to consume the embedding directly (DiT-style time-MLP), or call " +
            "PredictNoise(Tensor<T>, int, Tensor<T>?) with an explicit integer timestep.");
    }

    /// <summary>
    /// Cache for timestep embeddings to avoid recomputing sinusoidal embeddings
    /// for the same timestep during the denoising loop.
    /// </summary>
    private readonly Dictionary<int, Tensor<T>> _timestepEmbeddingCache = new();

    /// <inheritdoc />
    public virtual Tensor<T> GetTimestepEmbedding(int timestep)
    {
        ThrowIfDisposed();
        if (_timestepEmbeddingCache.TryGetValue(timestep, out var cached))
            return cached;

        // Sinusoidal timestep embedding emitted as rank-2 [1, TimeEmbeddingDim].
        // DiffusionResBlock requires rank >= 2 to validate the timeEmbed contract
        // before its lazy time-MLP bakes the input feature dim from the last axis.
        var halfDim = TimeEmbeddingDim / 2;
        var embedding = new Tensor<T>(new[] { 1, TimeEmbeddingDim });
        var embSpan = embedding.AsWritableSpan();

        var logScale = Math.Log(10000.0) / (halfDim - 1);

        for (int i = 0; i < halfDim; i++)
        {
            var freq = Math.Exp(-i * logScale);
            var angle = timestep * freq;

            embSpan[i] = NumOps.FromDouble(Math.Sin(angle));
            embSpan[i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
        }

        _timestepEmbeddingCache[timestep] = embedding;
        return embedding;
    }

    #endregion

    #region IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> Implementation

    /// <inheritdoc />
    public virtual void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        // Compute gradients and apply them
        var gradients = ComputeGradients(input, expectedOutput, LossFunction);
        var learningRate = NumOps.FromDouble(1e-4);
        ApplyGradients(gradients, learningRate);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Noise predictors are timestep-conditional: the model's output for the same input
    /// at <c>t = 0</c> is different from the output at <c>t = 999</c>. Picking an
    /// arbitrary default (the previous behavior of <c>t = 500</c>) returns the wrong
    /// function for any scheduler whose noise schedule isn't centered on that timestep.
    /// Callers must use <see cref="PredictNoise"/> directly with an explicit timestep
    /// (and conditioning context, if applicable) instead.
    /// </remarks>
    public virtual Tensor<T> Predict(Tensor<T> input)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: noise predictors are timestep-conditional and have no " +
            "meaningful single-tensor Predict(input) default. Call PredictNoise(input, " +
            "timestep, context?) directly with an explicit timestep, or run inference " +
            "through the parent diffusion model's sampling loop which orchestrates the " +
            "full timestep schedule.");
    }

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = GetType().Name,
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount,
            Description = $"Noise predictor with {ParameterCount} parameters, {InputChannels} input channels, {BaseChannels} base channels."
        };
    }

    #endregion

    #region IParameterizable<T, Tensor<T>, Tensor<T>> Implementation

    /// <inheritdoc />
    public abstract Vector<T> GetParameters();

    /// <inheritdoc />
    public abstract void SetParameters(Vector<T> parameters);

    /// <inheritdoc />
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var clone = (NoisePredictorBase<T>)Clone();
        clone.SetParameters(parameters);
        return clone;
    }

    #endregion

    #region IModelSerializer Implementation

    /// <inheritdoc />
    public virtual byte[] Serialize()
    {
        ThrowIfDisposed();
        ModelPersistenceGuard.EnforceBeforeSerialize();
        using var stream = new MemoryStream();
        SaveState(stream);
        return stream.ToArray();
    }

    /// <inheritdoc />
    public virtual void Deserialize(byte[] data)
    {
        ThrowIfDisposed();
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        using var stream = new MemoryStream(data);
        LoadState(stream);
    }

    /// <inheritdoc/>
    public virtual int[] GetInputShape()
    {
        return new[] { InputChannels };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        return new[] { OutputChannels };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc />
    public virtual void SaveModel(string filePath)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or whitespace.", nameof(filePath));

        var data = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary);
        File.WriteAllBytes(filePath, envelopedData);
    }

    /// <inheritdoc />
    public virtual void LoadModel(string filePath)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or whitespace.", nameof(filePath));

        var data = File.ReadAllBytes(filePath);

        // Extract payload from AIMF envelope
        data = ModelFileHeader.ExtractPayload(data);

        Deserialize(data);
    }

    #endregion

    #region ICheckpointableModel Implementation

    /// <inheritdoc />
    public virtual void SaveState(Stream stream)
    {
        ThrowIfDisposed();
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // Save version for future compatibility
        writer.Write(1); // Version 1

        // Save architecture info
        writer.Write(InputChannels);
        writer.Write(OutputChannels);
        writer.Write(BaseChannels);
        writer.Write(TimeEmbeddingDim);

        // Save model parameters using SerializationHelper
        SerializationHelper<T>.SerializeVector(writer, GetParameters());

        stream.Flush();
    }

    /// <inheritdoc />
    public virtual void LoadState(Stream stream)
    {
        ThrowIfDisposed();
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // Read version
        var version = reader.ReadInt32();
        if (version != 1)
            throw new InvalidOperationException($"Unsupported model version: {version}");

        // Read and validate architecture info
        var savedInputChannels = reader.ReadInt32();
        var savedOutputChannels = reader.ReadInt32();
        var savedBaseChannels = reader.ReadInt32();
        var savedTimeEmbeddingDim = reader.ReadInt32();

        if (savedInputChannels != InputChannels || savedOutputChannels != OutputChannels ||
            savedBaseChannels != BaseChannels || savedTimeEmbeddingDim != TimeEmbeddingDim)
        {
            throw new InvalidOperationException(
                $"Architecture mismatch: saved ({savedInputChannels}, {savedOutputChannels}, {savedBaseChannels}, {savedTimeEmbeddingDim}) " +
                $"vs current ({InputChannels}, {OutputChannels}, {BaseChannels}, {TimeEmbeddingDim}).");
        }

        // Load model parameters
        SetParameters(SerializationHelper<T>.DeserializeVector(reader));
    }

    #endregion

    #region IFeatureAware Implementation

    /// <summary>
    /// Ensures active feature indices are initialized with default values if empty.
    /// </summary>
    private void EnsureActiveFeatureIndicesInitialized()
    {
        if (_activeFeatureIndices.Count == 0 && ParameterCount > 0)
        {
            for (int i = 0; i < ParameterCount; i++)
            {
                _activeFeatureIndices.Add(i);
            }
        }
    }

    /// <inheritdoc />
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        EnsureActiveFeatureIndicesInitialized();
        return _activeFeatureIndices;
    }

    /// <inheritdoc />
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _activeFeatureIndices = new HashSet<int>(featureIndices);
    }

    /// <inheritdoc />
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        EnsureActiveFeatureIndicesInitialized();
        return _activeFeatureIndices.Contains(featureIndex);
    }

    #endregion

    #region IFeatureImportance<T> Implementation

    /// <inheritdoc />
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        var uniformImportance = NumOps.FromDouble(1.0 / Math.Max(1, ParameterCount));

        for (int i = 0; i < ParameterCount; i++)
        {
            importance[$"param_{i}"] = uniformImportance;
        }

        return importance;
    }

    #endregion

    #region ICloneable<IFullModel<T, Tensor<T>, Tensor<T>>> Implementation

    /// <inheritdoc />
    public abstract IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy();

    /// <inheritdoc />
    IFullModel<T, Tensor<T>, Tensor<T>> ICloneable<IFullModel<T, Tensor<T>, Tensor<T>>>.Clone()
    {
        return Clone();
    }

    /// <summary>
    /// Creates a deep copy of the noise predictor.
    /// </summary>
    /// <returns>A new instance with the same parameters.</returns>
    public abstract INoisePredictor<T> Clone();

    #endregion

    #region IGradientComputable<T, Tensor<T>, Tensor<T>> Implementation

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Noise predictors compute gradients via the engine's <see cref="GradientTape{T}"/>:
    /// every Engine op recorded during <see cref="PredictNoise"/> contributes a
    /// <c>GradFn</c> entry, so reverse-mode AD via
    /// <see cref="ComputeGradientsWithTape"/> produces exact per-tensor gradients
    /// without a manual backward pass through layers (<see cref="ILayer{T}"/> has no
    /// <c>Backward</c> in this codebase — autodiff is tape-only).
    /// </para>
    /// <para>
    /// The default implementation throws because the base class cannot enumerate the
    /// concrete predictor's trainable tensors. Concrete predictors must override either
    /// <see cref="ComputeGradients"/> directly or implement a tape-based path on top of
    /// <see cref="ComputeGradientsWithTape"/>.
    /// </para>
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (target == null)
            throw new ArgumentNullException(nameof(target));

        throw new NotSupportedException(
            $"{GetType().Name} does not implement ComputeGradients. " +
            "Override this method on the concrete predictor and route through " +
            "ComputeGradientsWithTape with the predictor's collected trainable tensors. " +
            "AiDotNet has no per-layer Backward; autodiff goes through the GradientTape " +
            "(see DiffusionModelBase.ComputeGradients for the canonical pattern).");
    }

    /// <summary>
    /// Forward pass through the noise predictor's layers, used as the differentiable
    /// path for tape-based gradient computation in <see cref="ComputeGradientsWithTape"/>.
    /// Concrete predictors must override this to call <see cref="PredictNoise"/> with the
    /// correct timestep (and conditioning context) for the training sample.
    /// </summary>
    /// <remarks>
    /// The base class cannot pick a meaningful default timestep — noise predictors are
    /// timestep-conditional, so any hardcoded value (the previous behavior of <c>t = 0</c>)
    /// would only train one branch of the schedule. Failing fast here forces concrete
    /// predictors / training loops to supply the timestep explicitly.
    /// </remarks>
    protected virtual Tensor<T> Forward(Tensor<T> input)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: Forward(Tensor<T>) has no meaningful default for a " +
            "timestep-conditional noise predictor. Override this method to call " +
            "PredictNoise(input, timestep, context?) with the training sample's timestep, " +
            "or invoke ComputeGradientsWithTape with a custom forwardBuilder so the " +
            "differentiable path knows which timestep / conditioning to use.");
    }

    /// <summary>
    /// Computes gradients using the engine's <see cref="GradientTape{T}"/> for automatic
    /// differentiation. This is the preferred training path — gradients are computed by
    /// recording all engine ops during the forward pass and then running reverse-mode AD.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="target">The target tensor for loss computation.</param>
    /// <param name="trainableParams">The trainable parameter tensors to compute gradients for.</param>
    /// <param name="lossBuilder">
    /// Optional callback that builds a scalar loss tensor from the recorded
    /// <c>(predicted, target)</c> pair using engine ops (so the tape can differentiate
    /// through it). When <c>null</c>, an MSE loss is used as a sensible default. Pass a
    /// non-null builder when the caller's training loop uses a different objective —
    /// otherwise gradients would silently optimize the wrong loss.
    /// </param>
    /// <param name="forwardBuilder">
    /// Optional callback that runs the differentiable forward pass against
    /// <paramref name="input"/>, recording the engine ops the tape needs. When
    /// <c>null</c>, falls back to the protected <see cref="Forward"/> hook (which
    /// concrete predictors override with their timestep-aware implementation). Pass a
    /// non-null builder to bind a specific timestep / conditioning per call without
    /// requiring a Forward override.
    /// </param>
    /// <returns>Dictionary mapping each parameter tensor to its gradient.</returns>
    public Dictionary<Tensor<T>, Tensor<T>> ComputeGradientsWithTape(
        Tensor<T> input,
        Tensor<T> target,
        Tensor<T>[] trainableParams,
        Func<Tensor<T>, Tensor<T>, Tensor<T>>? lossBuilder = null,
        Func<Tensor<T>, Tensor<T>>? forwardBuilder = null)
    {
        using var tape = new GradientTape<T>();

        // Forward pass (recorded by the engine).
        var predicted = forwardBuilder is not null
            ? forwardBuilder(input)
            : Forward(input);

        Tensor<T> loss;
        if (lossBuilder is not null)
        {
            loss = lossBuilder(predicted, target);
        }
        else
        {
            // Default: MSE = mean((predicted - target)^2). Tape-recorded so AD works.
            var diff = Engine.TensorSubtract(predicted, target);
            var squared = Engine.TensorMultiply(diff, diff);
            var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
            loss = Engine.ReduceMean(squared, allAxes, keepDims: false);
        }

        // Reverse-mode AD: compute gradients for all trainable parameters
        return tape.ComputeGradients(loss, trainableParams);
    }

    /// <summary>
    /// Extracts accumulated parameter gradients from all layers after backpropagation.
    /// </summary>
    protected virtual Vector<T> GetParameterGradients()
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not implement GetParameterGradients. " +
            "Override this method to extract layer-level gradients.");
    }

    /// <inheritdoc />
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();

        // Vectorized SGD: params = params - lr * gradients
        var scaledGradients = Engine.Multiply(gradients, learningRate);
        var updated = Engine.Subtract(parameters, scaledGradients);

        SetParameters(updated);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Samples random noise from a standard normal distribution.
    /// </summary>
    /// <param name="shape">The shape of the noise tensor.</param>
    /// <param name="rng">Optional random number generator.</param>
    /// <returns>A tensor of random noise values.</returns>
    protected virtual Tensor<T> SampleNoise(int[] shape, Random? rng = null)
    {
        rng = rng ?? RandomGenerator;
        long totalElements = 1;
        foreach (var dim in shape)
            totalElements = checked(totalElements * dim);

        var noise = new Tensor<T>(shape);
        var noiseSpan = noise.AsWritableSpan();

        for (int i = 0; i < noiseSpan.Length; i++)
        {
            noiseSpan[i] = NumOps.FromDouble(rng.NextGaussian());
        }

        return noise;
    }

    #endregion

    #region IDisposable

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases managed resources — compiled plans first (so pooled tensor
    /// buffers the plans captured are freed before layers Dispose and return
    /// their weights), then every <see cref="ILayer{T}"/> exposed by
    /// <see cref="EnumerateLayers"/> that implements <see cref="IDisposable"/>.
    /// </summary>
    /// <remarks>
    /// <see cref="EnumerateLayers"/> defaults to an empty enumeration so injected
    /// or shared layers (cross-attention from a shared encoder, a VAE reference
    /// passed in by the caller) are NOT torn down here. Concrete predictors that
    /// own their layers must opt in to cleanup by overriding
    /// <see cref="EnumerateLayers"/> — either yielding specific owned-field
    /// references or returning <c>ReflectInstanceLayers(this)</c>. The
    /// <see cref="ObjectDisposedException"/> catch additionally prevents a
    /// shared-layer graph — the same <see cref="ILayer{T}"/> instance used by
    /// multiple predictors or networks — from aborting the cascade when a previous
    /// owner already disposed it.
    /// </remarks>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed || !disposing) return;
        _disposed = true;

        ReleaseStreamingWeights(includeRegistered: true);

        _compileHost.Dispose();

        // Release tensor handles cached per integer timestep — these are
        // owned exclusively by this predictor and have no other Dispose path.
        foreach (var embedding in _timestepEmbeddingCache.Values)
        {
            if (embedding is IDisposable d) d.Dispose();
        }
        _timestepEmbeddingCache.Clear();

        // Route layer Dispose through DisposeOnceGuard — shared layers
        // between predictors (ensemble predictors, cross-attention layers
        // reused from a shared encoder, VAE layers injected into multiple
        // wrappers) are common. Relying on ObjectDisposedException is
        // unsafe because many layer Dispose implementations double-return
        // pooled tensor buffers on a second Dispose call without throwing.
        foreach (var layer in EnumerateLayers())
        {
            if (layer is IDisposable disposable)
            {
                AiDotNet.Helpers.DisposeOnceGuard.TryDispose(disposable);
            }
        }
    }

    #endregion
}
