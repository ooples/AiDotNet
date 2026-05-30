using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Compiled training step — auto-compiles the forward + backward pass on the first step,
/// then replays the compiled plan on subsequent steps for near-zero overhead training.
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><b>Step 1 (tracing):</b> Enables GraphMode, traces the forward pass + loss computation
/// through the layer stack, compiles a CompiledTrainingPlan with backward pass, and executes it.</item>
/// <item><b>Steps 2+ (replay):</b> Calls plan.Step() which replays the compiled forward + backward
/// as flat delegate arrays with pre-allocated gradient buffers. Zero allocation, zero dispatch overhead.</item>
/// </list>
///
/// <para><b>Recompilation triggers:</b></para>
/// <list type="bullet">
/// <item>Input shape changes (different batch size, sequence length, etc.)</item>
/// <item>Explicit Invalidate() call (model structure changed)</item>
/// <item>Compilation failure (falls back to eager TapeTrainingStep for that shape)</item>
/// </list>
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public static class CompiledTapeTrainingStep<T>
{
    [ThreadStatic]
    private static CompiledModelCache<T>? _cache;
    [ThreadStatic]
    private static Tensor<T>[]? _cachedParameters;

    /// <summary>
    /// AiDotNet#1406: identity of the trainable-layer set that produced
    /// <see cref="_cachedParameters"/> and the cached compiled plan. The
    /// per-thread cache above keys plans by tensor shape only, so two
    /// distinct models with the same input/target shapes — created in
    /// sequence on the same test thread, for example — would otherwise
    /// share a single compiled plan. The plan's tensor leaves capture the
    /// FIRST model's parameter refs, and a replay then "trains" model A's
    /// (potentially garbage-collected) tensors while model B's params sit
    /// untouched. Surfaced as
    /// <c>ScientificMLTests.{Hamiltonian,Lagrangian}NeuralNetwork_TrainUpdatesParameters</c>
    /// failing only when run after
    /// <c>UniversalDifferentialEquation_TrainUpdatesParameters</c>.
    /// We track the layer-set identity (element-wise <see cref="object.ReferenceEquals"/>)
    /// and force <see cref="Invalidate"/> when it diverges from the cached
    /// one. Per-instance optimizer state is reset as part of Invalidate, so
    /// the next model gets a clean compile.
    /// </summary>
    [ThreadStatic]
    private static object?[]? _cachedLayerSetIdentities;

    /// <summary>
    /// AiDotNet#1331: persistent input tensor reused across <see cref="TryStepWithFusedOptimizer"/>
    /// calls. The compiled plan captures whatever tensor ref the trace lambda saw — if every call
    /// allocated a fresh <c>Tensor&lt;T&gt;</c> for <c>input</c> (the canonical PyTorch-style
    /// <c>model.Train(input_k, target_k)</c> pattern), the plan's captured ref points at the
    /// FIRST tensor forever and <c>plan.Step()</c> replays with stale data — gradients become
    /// disconnected from the actual training data, loss drifts upward, model never converges.
    /// We solve this by tracing through a SINGLE persistent tensor and copying the caller's
    /// fresh data into it on every step. See <c>InputDataMustRefreshAcrossStep_NotFrozenAtCompileTime</c>
    /// in the Tensors test suite for the diagnostic that proves this pattern.
    /// </summary>
    [ThreadStatic]
    private static Tensor<T>? _persistentInput;

    /// <summary>
    /// AiDotNet#1331: persistent target tensor. Same rationale as <see cref="_persistentInput"/> —
    /// the loss-computation lambda captures <c>target</c> by reference, so per-call fresh target
    /// tensors are invisible to <c>plan.Step()</c>. Funnelling every <c>Train</c> call through
    /// this single tensor (with in-place data copy) keeps the captured graph leaf in sync with
    /// the caller's data.
    /// </summary>
    [ThreadStatic]
    private static Tensor<T>? _persistentTarget;

    /// <summary>
    /// The single plan that has been configured with an optimizer on this
    /// thread. <b>Strict single-plan semantics</b>: Adam/AdamW/SGD moment
    /// buffers live INSIDE the compiled plan (via
    /// <c>ICompiledTrainingPlan.ConfigureOptimizer</c>), so allowing multiple
    /// plans to each accumulate their own state would silently fork
    /// optimizer state across variable-shape batches and diverge from the
    /// reference eager semantics (one optimizer, one state vector). Any
    /// attempt to engage a DIFFERENT plan on the same thread returns
    /// <c>false</c> from <see cref="TryStepWithFusedOptimizer"/> — the
    /// caller must decide whether to halt or continue via eager (the
    /// NeuralNetworkBase caller enforces a strict commitment so the
    /// state-loss cannot happen silently).
    /// </summary>
    [ThreadStatic]
    private static object? _configuredPlan;

    /// <summary>
    /// Snapshot of the hyperparameters passed to
    /// <c>ICompiledTrainingPlan.ConfigureOptimizer</c> on
    /// <see cref="_configuredPlan"/>. Used to detect drift on the same
    /// plan (LR change between steps, beta change) — reconfiguring would
    /// reset m/v buffers and silently corrupt training, so on drift we
    /// also return <c>false</c>.
    /// </summary>
    [ThreadStatic]
    private static (int OptType, float Lr, float B1, float B2, float Eps, float Wd)? _configuredOptimizerConfig;

    /// <summary>
    /// Counter of successful fused-step executions on this thread. Exposed
    /// via <see cref="GetFusedStepCount"/>/<see cref="ResetFusedStepCount"/>
    /// so integration tests can assert that the fused compiled path
    /// <i>actually engaged</i> rather than silently falling back to eager
    /// (a test that only checks "finite loss" cannot distinguish the two).
    /// </summary>
    [ThreadStatic]
    private static long _fusedStepCount;

    /// <summary>
    /// Set once on the calling thread when an AMSGrad fused step fails because the
    /// linked Tensors build can't run the AMSGrad kernel. Subsequent AMSGrad steps
    /// then skip the fused attempt outright (returning false straight to the eager
    /// tape) instead of reconfiguring → throwing → catching → warning every step,
    /// which would turn a one-time capability gap into per-step exception + log churn.
    /// </summary>
    [ThreadStatic]
    private static System.Collections.Generic.HashSet<AiDotNet.Tensors.Engines.Compilation.OptimizerType>? _fusedUnavailableTypes;

    /// <summary>Gets the count of successful fused-step executions on the calling thread.</summary>
    public static long GetFusedStepCount() => _fusedStepCount;

    /// <summary>Resets the fused-step counter on the calling thread to zero.</summary>
    public static void ResetFusedStepCount() { _fusedStepCount = 0; }

    /// <summary>
    /// AiDotNet#1395: when <see cref="TryStepWithFusedOptimizer"/> falls back via
    /// the catch path, the underlying exception is stored here so the caller
    /// (NeuralNetworkBase) can surface it in the "fused has committed but step N
    /// can't engage" InvalidOperationException. Previously the catch's exception
    /// was logged to Trace only — users debugging from a failing test never saw
    /// the actual root cause (e.g. "Parameter N non-contiguous CPU layout" from
    /// <see cref="Tensors.Engines.Compilation.CompiledTrainingPlan{T}.ConfigureOptimizer"/>,
    /// a shape mismatch from a backward kernel, a NaN guard trip). Now the
    /// caller can quote the original exception's type + message + stack so the
    /// error is self-diagnosing.
    /// </summary>
    [ThreadStatic]
    private static System.Exception? _lastFallbackException;

    /// <summary>
    /// AiDotNet#1395: read the last exception that caused
    /// <see cref="TryStepWithFusedOptimizer"/> to fall back, or <c>null</c> if
    /// the most recent fallback was due to one of the explicit return-false
    /// paths (plan switch, config drift, EnableCompilation=false, etc.) rather
    /// than a swallowed exception.
    /// </summary>
    public static System.Exception? GetLastFallbackException() => _lastFallbackException;

    // Reflection-cached lookup of ICompiledTrainingPlan<T>.SetMaxGradNorm(double).
    // Populated lazily on first call per process and reused on every subsequent
    // step. Returns null when the underlying Tensors assembly pre-dates the
    // SetMaxGradNorm API addition (AiDotNet.Tensors PR #359 / first release
    // after 0.81.0), at which point we silently skip the plan-side clip and
    // rely on the eager NeuralNetworkBase.TrainWithTape clipping. This shim is
    // intentionally tolerant — it's the single API-presence check we need to
    // keep AiDotNet building against any 0.8x AiDotNet.Tensors NuGet.
    private static System.Reflection.MethodInfo? s_setMaxGradNormMethod;
    private static bool s_setMaxGradNormProbed;

    private static void TrySetPlanMaxGradNorm(ICompiledTrainingPlan<T> plan, double maxGradNorm)
    {
        if (!s_setMaxGradNormProbed)
        {
            s_setMaxGradNormMethod = typeof(ICompiledTrainingPlan<T>).GetMethod(
                "SetMaxGradNorm", new[] { typeof(double) });
            s_setMaxGradNormProbed = true;
        }
        if (s_setMaxGradNormMethod is null) return; // older Tensors NuGet — eager path clips.
        s_setMaxGradNormMethod.Invoke(plan, new object[] { maxGradNorm });
    }

    /// <summary>
    /// Executes a single compiled training step.
    /// First call traces and compiles; subsequent calls replay the compiled plan.
    /// Falls back to eager execution if compilation fails.
    /// </summary>
    public static T Step(
        IReadOnlyList<ITrainableLayer<T>> layers,
        Tensor<T> input,
        Tensor<T> target,
        T learningRate,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> computeLoss)
    {
        if (!TensorCodecOptions.Current.EnableCompilation)
            return TapeTrainingStep<T>.Step(layers, input, target, learningRate, forward, computeLoss);

        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;

        try
        {
            // AiDotNet#1406: drop the cached compiled plan + parameter array
            // when the caller has switched to a different layer set. Without
            // this the next non-matching model on the same thread would
            // replay the previous model's plan against its own (uninvolved)
            // tensors — the plan was compiled with the FIRST model's leaf
            // refs and the optimizer step would update those tensors, not
            // the caller's.
            if (InvalidateIfLayerSetChanged(layers))
            {
                // Caches cleared; cache field rebound below.
            }
            var cache = _cache ??= new CompiledModelCache<T>();

            // Force layer initialization before collecting parameters.
            // DenseLayer.EnsureInitialized() replaces _weights with a new tensor on
            // first Forward — collecting before that captures stale placeholder tensors.
            if (_cachedParameters is null)
                forward(input);

            // Use the dedup-aware collector here too so the cached array is
            // safe to reuse if a subsequent caller routes through
            // TryStepWithFusedOptimizer. Shared/tied weights would otherwise
            // appear twice in the array — wrong for both eager-SGD here AND
            // wrong for the fused kernel's m/v buffers downstream.
            bool firstCollectThisLifecycle = _cachedParameters is null;
            var parameters = _cachedParameters ??= CollectDeduplicatedParameters(layers);
            if (firstCollectThisLifecycle) RememberLayerSet(layers);

            // Zero gradients before forward pass
            foreach (var layer in layers)
                layer.ZeroGrad();

            // Compile key must include BOTH input AND target shapes — the
            // traced lambda captures target via computeLoss. A plan compiled
            // for input=[B,D], target=[B,C] would otherwise silently replay
            // the wrong op graph if the next call arrives with the same
            // input shape but a different target shape (e.g., a regression
            // output scalar vs. a classification class-index). Build a
            // composite synthetic key (input + separator + target) matching
            // the TryStepWithFusedOptimizer convention so distinct
            // {input, target} pairs hit distinct cache entries.
            var inputShape = input._shape;
            var targetShape = target._shape;
            var compositeKey = new int[inputShape.Length + 1 + targetShape.Length];
            Array.Copy(inputShape, 0, compositeKey, 0, inputShape.Length);
            compositeKey[inputShape.Length] = -1; // separator sentinel (no real dim is negative)
            Array.Copy(targetShape, 0, compositeKey, inputShape.Length + 1, targetShape.Length);

            var plan = cache.GetOrCompileTraining(
                compositeKey,
                () =>
                {
                    // Tensors 0.50.1 changed GetOrCompileTraining to take a
                    // Func<Tensor<T>> — the trace lambda must return the
                    // scalar output (loss) so the compile-graph has a single
                    // terminal node to differentiate from.
                    var predicted = forward(input);
                    return computeLoss(predicted, target);
                },
                parameters);

            // Execute compiled forward + backward
            var lossOutput = plan.Step();

            // In-place SGD: param -= lr * grad (zero allocation)
            UpdateParametersSGD(parameters, plan.Gradients, learningRate, numOps);

            return lossOutput.Length > 0 ? lossOutput[0] : numOps.Zero;
        }
        catch
        {
            // Fall back to eager for this step — next step will retry compilation
            return TapeTrainingStep<T>.Step(layers, input, target, learningRate, forward, computeLoss);
        }
    }

    /// <summary>
    /// Invalidates the compiled plan cache. Call when model structure changes.
    /// </summary>
    /// <summary>
    /// AiDotNet#1406: invalidates the per-thread compiled-plan cache when the
    /// supplied trainable-layer set is not the same one that produced the
    /// currently-cached parameters (compared element-wise by reference
    /// identity). Returns <c>true</c> if an invalidation occurred. Cheap on
    /// the steady state (single ref-compare per layer when the set matches);
    /// only allocates on a model switch.
    /// </summary>
    private static bool InvalidateIfLayerSetChanged<TLayer>(IReadOnlyList<TLayer> layers) where TLayer : class
    {
        var cached = _cachedLayerSetIdentities;
        if (cached is null) return false;

        if (cached.Length != layers.Count)
        {
            Invalidate();
            return true;
        }
        for (int i = 0; i < cached.Length; i++)
        {
            if (!ReferenceEquals(cached[i], layers[i]))
            {
                Invalidate();
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Captures the current trainable-layer set's reference identities so a
    /// subsequent call can detect a model switch. Called immediately after
    /// the cache is populated for the first time in a given lifecycle.
    /// </summary>
    private static void RememberLayerSet<TLayer>(IReadOnlyList<TLayer> layers) where TLayer : class
    {
        var ids = new object?[layers.Count];
        for (int i = 0; i < layers.Count; i++) ids[i] = layers[i];
        _cachedLayerSetIdentities = ids;
    }

    public static void Invalidate()
    {
        _cache?.Invalidate();
        _cachedParameters = null;
        _cachedLayerSetIdentities = null;
        _configuredPlan = null;
        _configuredOptimizerConfig = null;
        // AiDotNet#1331: drop the persistent input/target tensors so the next
        // call traces a fresh plan with new captured leaves. Forgetting this
        // would re-use the old tensors with whatever shape they had — a
        // shape-changed call would then hit ValidateShapesMatch in SetInputs
        // and throw, masking what is really a model-structure change.
        _persistentInput = null;
        _persistentTarget = null;
        // Reset the fused-engagement counter — from this point on, any
        // assertion about "fused ran at least N times" should reflect the
        // new lifecycle.
        _fusedStepCount = 0;
        // AiDotNet#1469 review: a fresh lifecycle must be able to re-enable fused execution. The
        // unavailable-type latch records capability gaps seen during the PREVIOUS lifecycle; clear
        // it here so a re-traced plan (e.g. a different model on this thread) retries the fused path
        // instead of inheriting a stale "disabled" verdict.
        _fusedUnavailableTypes?.Clear();
    }

    /// <summary>
    /// Compiled training step with <b>fused optimizer</b> — forward + backward +
    /// parameter update all run in one compiled kernel. No materialized gradient
    /// tensors between backward and the optimizer step. SIMD-accelerated update
    /// via <see cref="FusedOptimizer"/> (float-only).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is faster than <see cref="Step"/> even for SGD because the parameter
    /// update happens inside the compiled plan's replay kernel — zero allocation,
    /// zero dispatch between backward and update, tight SIMD inner loop.
    /// </para>
    /// <para>
    /// <b>Constraints:</b>
    /// <list type="bullet">
    /// <item>Only <c>T = float</c> is supported (Tensors-side limitation — the
    /// fused optimizer kernels operate on <c>float*</c> directly).</item>
    /// <item>Only SGD, Adam, and AdamW are supported by
    /// <see cref="CompiledTrainingPlan{T}.ConfigureOptimizer"/>. Other optimizer
    /// types must use the plain <see cref="Step"/> method or the eager tape path.</item>
    /// <item>Hyperparameters (LR, betas, eps, weight decay) are <b>baked at the
    /// first configure call per compile</b>. Re-calling with different LR would
    /// reset the Adam momentum buffers — so callers that use learning-rate
    /// schedulers or adaptive rates should use <see cref="Step"/> instead.</item>
    /// </list>
    /// </para>
    /// <para>
    /// The method returns <c>false</c> on compilation failure OR when the fused
    /// path isn't applicable, letting the caller fall through to the eager path.
    /// The out <paramref name="lossValue"/> is meaningful only when the return
    /// is <c>true</c>.
    /// </para>
    /// </remarks>
    /// <returns><c>true</c> when the fused compiled step ran successfully.</returns>
    internal static bool TryStepWithFusedOptimizer(
        IReadOnlyList<ITrainableLayer<T>> layers,
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> computeLoss,
        AiDotNet.Tensors.Engines.Compilation.OptimizerType optimizerType,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        out T lossValue,
        double maxGradNorm = 0.0,
        AiDotNet.Tensors.Engines.Compilation.LrSchedule? lrSchedule = null)
    {
        lossValue = MathHelper.GetNumericOperations<T>().Zero;
        // AiDotNet#1395: clear the previous-call's exception buffer so the
        // caller's GetLastFallbackException reflects only the outcome of THIS
        // call. (Cleared on entry, not on success, so a successful step doesn't
        // leak a stale exception from earlier.)
        _lastFallbackException = null;

        if (!TensorCodecOptions.Current.EnableCompilation) return false;
        // Fused optimizer kernels support float and double on the Tensors
        // side (PR #319 / FusedOptimizer.{SGD,Adam,AdamW}UpdateSimd double
        // overloads + CompiledTrainingPlan.ConfigureOptimizerDouble). Other
        // numeric types still fall through to the eager autograd path.
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double)) return false;
        // Allowlist of optimizer kernels wired through ConfigureOptimizer in the
        // linked AiDotNet.Tensors build (0.88.0: ConfigureOptimizerFloat handles
        // all of these on CPU). Only OptimizerTypes an IFusedOptimizerSpec
        // actually emits are reachable here; the allowlist is the belt-and-braces
        // guard so a spec that names a type the linked Tensors build can't run
        // falls back loudly (via the catch below + the per-type latch) rather
        // than throwing per step — never a wrong update.
        if (optimizerType is not (AiDotNet.Tensors.Engines.Compilation.OptimizerType.SGD
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.SGDMomentum
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adam
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.AdamW
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.AMSGrad
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.Nadam
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.RAdam
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.AdaMax
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.AdaDelta
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adagrad
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.RMSprop
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.Lion
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.LARS
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.LAMB
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.FTRL
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.ASGD
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.Rprop
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.HypergradientSGD
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.ScheduleFreeSGD
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.DAdaptationSGD))
            return false;

        // If a prior fused step already proved this thread's Tensors build can't
        // run THIS optimizer kernel, don't retry the fused path — go straight to
        // the eager tape. Otherwise every step would reconfigure, throw, catch and
        // warn, turning a one-time capability gap into per-step exception/log churn.
        if (_fusedUnavailableTypes is not null && _fusedUnavailableTypes.Contains(optimizerType))
            return false;

        try
        {
            // AiDotNet#1406: drop the cached compiled plan + parameter array
            // when the caller has switched to a different layer set. The
            // per-thread cache keys plans by shape only, so two distinct
            // models with the same (input, target) shapes — chained on the
            // same test thread, for example — would otherwise replay the
            // FIRST model's compiled plan against tensors that no longer
            // exist on the live model. Symptom: post-Train params identical
            // to pre-Train, even though LastLoss reports a non-zero loss
            // (the plan ran on the previous model's now-stale tensors).
            InvalidateIfLayerSetChanged(layers);
            var cache = _cache ??= new CompiledModelCache<T>();

            // AiDotNet#1331: ensure the persistent input/target tensors exist
            // with shapes matching the caller's data. If shape changed since
            // the last call (different batch size, sequence length, ...) the
            // single-plan policy below would refuse the new plan anyway — we
            // pre-empt that by invalidating up front so the user gets a clean
            // recompile rather than a confusing rejection.
            if (_persistentInput is null
                || !ShapesEqual(_persistentInput._shape, input._shape)
                || _persistentTarget is null
                || !ShapesEqual(_persistentTarget._shape, target._shape))
            {
                Invalidate();
                _persistentInput = new Tensor<T>(input._shape);
                _persistentTarget = new Tensor<T>(target._shape);
                // Re-acquire the cache reference after Invalidate cleared it.
                cache = _cache ??= new CompiledModelCache<T>();
            }

            // Copy the caller's fresh per-call data into the persistent
            // tensors BEFORE compilation or replay. The compiled plan's
            // lazy graph leaves point at _persistentInput / _persistentTarget;
            // every plan.Step() (including the implicit one inside the
            // compile lambda) reads from these refs, so writing to them
            // here makes the new data visible.
            input.AsSpan().CopyTo(_persistentInput!.AsWritableSpan());
            target.AsSpan().CopyTo(_persistentTarget!.AsWritableSpan());

            // Force layer initialization before collecting parameters — DenseLayer
            // replaces _weights with a new tensor on first Forward.
            //
            // Issue #350 v3 (compile-vs-eager parity): the pre-trace forward
            // call here MUST NOT consume the same per-step random state that
            // the trace lambda's forward will consume. The trace lambda is
            // called immediately after, runs ForwardForTraining in TRAINING
            // MODE, and any Dropout / sampling op pulls values from the
            // shared ThreadSafeRandom counter at that point. If THIS pre-init
            // forward also runs in training mode, it consumes the SAME
            // counter values FIRST — leaving the trace forward to consume
            // the NEXT batch of values. EAGER mode (TapeTrainingStep) only
            // calls forward once per Step, so its dropout masks come from
            // the counter values that compile-mode's pre-init swallowed.
            // Net effect: every Dropout layer's mask diverges between the
            // two execution paths, and on a deep network like the 53-layer
            // GraFPrint pyramid the cumulative activation drift makes
            // every downstream gradient differ in sign and magnitude.
            //
            // Fix: drop into inference mode for the pre-init forward so
            // Dropout returns input unchanged (no mask, no RNG consumed).
            // Other lazy-init code paths (DenseLayer weights, etc.) still
            // run; only stochastic ops short-circuit. Restore training mode
            // before the trace lambda fires so the actual training forward
            // sees the same RNG state the eager path would.
            if (_cachedParameters is null)
            {
                // TryStepWithFusedOptimizer is the training entry — every layer
                // is in training mode here by precondition (NeuralNetworkBase
                // calls SetTrainingMode(true) before invoking this). Drop them
                // into inference mode for the lazy-init forward, then restore
                // to training mode before the trace lambda fires. This avoids
                // burning the deterministic random counter on Dropout masks
                // that are immediately discarded.
                for (int li = 0; li < layers.Count; li++)
                    layers[li].SetTrainingMode(false);
                try
                {
                    forward(_persistentInput);
                }
                finally
                {
                    for (int li = 0; li < layers.Count; li++)
                        layers[li].SetTrainingMode(true);
                }
            }

            // Use the dedup-aware collector for the fused path. Shared/tied
            // weights (same Tensor<T> instance referenced by multiple layers)
            // would otherwise drive the fused kernel's m/v buffers to update
            // the same parameter twice per step, breaking Adam's moment math.
            bool firstCollectThisLifecycle = _cachedParameters is null;
            var parameters = _cachedParameters ??= CollectDeduplicatedParameters(layers);
            if (firstCollectThisLifecycle) RememberLayerSet(layers);

            foreach (var layer in layers)
                layer.ZeroGrad();

            // Compile key must include BOTH input AND target shapes. The traced
            // lambda captures target shape via computeLoss — a plan compiled for
            // input=[B,D], target=[B,C] would silently replay wrong ops if the
            // next call arrives with input=[B,D], target=[1,B,C] (different
            // reshape/loss graph). Concatenate input + separator + target into
            // one synthetic shape key so distinct {input, target} pairs hit
            // distinct cache entries.
            var inputShape = input._shape;
            var targetShape = target._shape;
            var compositeKey = new int[inputShape.Length + 1 + targetShape.Length];
            Array.Copy(inputShape, 0, compositeKey, 0, inputShape.Length);
            compositeKey[inputShape.Length] = -1; // separator sentinel (no real dim is negative)
            Array.Copy(targetShape, 0, compositeKey, inputShape.Length + 1, targetShape.Length);

            var plan = cache.GetOrCompileTraining(
                compositeKey,
                () =>
                {
                    // Trace through the persistent tensors so plan.Step()
                    // reads from the same refs we update each call.
                    var predicted = forward(_persistentInput!);
                    return computeLoss(predicted, _persistentTarget!);
                },
                parameters);

            // STRICT SINGLE-PLAN POLICY: optimizer state lives inside the
            // compiled plan (ConfigureOptimizer attaches m/v buffers directly
            // to the plan object). Letting multiple plans each accumulate
            // their own m/v would fork Adam state across variable-shape
            // batches and silently diverge from eager semantics (where one
            // optimizer holds one state vector across all shapes).
            //
            // So: the FIRST plan we see on this thread is the only plan we'll
            // configure. Any subsequent call with a different plan (e.g., a
            // new {input, target} shape combo producing a new compiled plan)
            // returns false — the caller (NeuralNetworkBase) then enforces
            // its commitment rule so state loss cannot happen silently.
            //
            // Drift on the SAME plan (LR or beta change between steps) also
            // returns false — reconfiguring would reset m/v.
            var currentConfig = ((int)optimizerType, learningRate, beta1, beta2, epsilon, weightDecay);

            if (_configuredPlan is null)
            {
                // First fused call on this thread. Configure the plan and
                // commit to single-plan semantics from here on. When the
                // caller passed an lrSchedule, use the LrSchedule overload
                // so the fused kernel evaluates the per-step learning rate
                // inline (cosine, exponential, etc.) — no perf penalty vs
                // constant LR, but it lets paper-faithful schedulers
                // (cosine annealing, OneCycle, linear-warmup-cosine) run
                // through the fused path instead of falling back to eager.
                if (lrSchedule != null)
                {
                    plan.ConfigureOptimizer(
                        optimizerType,
                        lrSchedule,
                        beta1,
                        beta2,
                        epsilon,
                        weightDecay);
                }
                else
                {
                    plan.ConfigureOptimizer(
                        optimizerType,
                        learningRate,
                        beta1,
                        beta2,
                        epsilon,
                        weightDecay);
                }
                _configuredPlan = plan;
                _configuredOptimizerConfig = currentConfig;
                // Apply the global gradient-norm clip threshold to the plan
                // when the underlying ICompiledTrainingPlan<T> exposes
                // SetMaxGradNorm. The fused-plan-side clip executes between
                // backward and the optimizer update so the optimizer sees
                // clipped gradients — matching the eager path's semantics
                // (NeuralNetworkBase.TrainWithTape clips before calling
                // opt.Step). Reflection-shim so AiDotNet still builds
                // against AiDotNet.Tensors versions that pre-date the
                // SetMaxGradNorm addition (AiDotNet.Tensors PR #359); on
                // those older versions the eager path's clipping is the
                // only line of defense, which is still correct just not
                // fused. Pass 0 to disable.
                if (maxGradNorm > 0.0)
                    TrySetPlanMaxGradNorm(plan, maxGradNorm);
            }
            else if (!ReferenceEquals(_configuredPlan, plan))
            {
                // Plan switch → different shape or structure produced a new
                // compiled plan. Using a fresh plan would fork optimizer
                // state, so refuse and let the caller handle it (the
                // NeuralNetworkBase caller throws once fused has committed).
                return false;
            }
            else if (_configuredOptimizerConfig is null
                || !_configuredOptimizerConfig.Value.Equals(currentConfig))
            {
                // Same plan, drifted hyperparameters between steps. Refuse
                // to re-configure (would reset m/v) and let the caller
                // handle the drift.
                return false;
            }

            // Execute forward + backward + fused parameter update in one replay.
            var lossOutput = plan.Step();
            lossValue = lossOutput.Length > 0 ? lossOutput[0] : MathHelper.GetNumericOperations<T>().Zero;
            // Signal successful fused engagement so tests/diagnostics can
            // assert the compiled path actually ran — distinguishing it from
            // a silent fallback to the eager path.
            _fusedStepCount++;
            return true;
        }
        catch (Exception ex)
        {
            // Trace the failure so fused-path regressions are observable in
            // production telemetry. Include ex.ToString() so stack trace +
            // inner exceptions reach telemetry — without these, diagnosing
            // a fused-path regression from logs requires reproducing the
            // failure locally. Clear the single-slot config state so any
            // next attempt reconfigures fresh.
            //
            // AiDotNet#1395: also stash the exception so the caller's
            // "fused has committed but step cannot engage" InvalidOperationException
            // can quote the underlying cause (Parameter N non-contiguous CPU
            // layout, shape mismatch, NaN guard, etc.). Trace alone wasn't
            // enough — failing tests don't surface Trace output by default.
            _lastFallbackException = ex;
            System.Diagnostics.Trace.TraceWarning(
                $"CompiledTapeTrainingStep.TryStepWithFusedOptimizer failed, falling back to eager: " +
                $"{ex}");
            // Latch THIS optimizer type as fused-unsupported on this thread ONLY for capability-gap
            // exceptions — a missing kernel/method/type/native-entry won't change within this
            // process, so latching avoids reconfigure/throw/warn churn every step. Transient runtime
            // failures (shape mismatch, NaN guard, a one-off non-contiguous CPU layout) must fall
            // back THIS step but NOT permanently disable fused for the type on this thread, since a
            // later unrelated model could engage it fine (AiDotNet#1469 review). Generalized from the
            // original AMSGrad-only latch.
            if (ex is NotSupportedException or MissingMethodException or TypeLoadException
                or EntryPointNotFoundException or DllNotFoundException)
            {
                (_fusedUnavailableTypes ??= new System.Collections.Generic.HashSet<AiDotNet.Tensors.Engines.Compilation.OptimizerType>())
                    .Add(optimizerType);
            }
            _configuredPlan = null;
            _configuredOptimizerConfig = null;
            return false;
        }
    }


    /// <summary>
    /// Deduplicates trainable parameter tensors by reference identity. The eager
    /// <see cref="TapeTrainingStep{T}"/> path also dedupes (shared/tied weights
    /// must be updated once per step, not once per layer that holds the alias).
    /// </summary>
    private static Tensor<T>[] CollectDeduplicatedParameters(IReadOnlyList<ITrainableLayer<T>> layers)
    {
        var seen = new HashSet<Tensor<T>>(AiDotNet.Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        var result = new List<Tensor<T>>();
        foreach (var layer in layers)
        {
            foreach (var p in layer.GetTrainableParameters())
            {
                if (p is not null && seen.Add(p))
                    result.Add(p);
            }
        }
        return result.ToArray();
    }

    /// <summary>
    /// AiDotNet#1331 helper: structural shape equality. We need this on the
    /// hot path where every <c>Train</c> call decides whether the persistent
    /// input/target tensors are still usable — Tensor's default
    /// <c>Equals</c> compares data, not shape, so a per-call data difference
    /// would force an unnecessary recompile.
    /// </summary>
    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static Tensor<T>[] CollectParameterArray(IReadOnlyList<ITrainableLayer<T>> layers)
    {
        var allParams = new List<Tensor<T>>();
        foreach (var layer in layers)
            allParams.AddRange(layer.GetTrainableParameters());
        return allParams.ToArray();
    }

    /// <summary>
    /// In-place SGD: param[i] -= lr * grad[i] for each element.
    /// Zero allocation — operates directly on the parameter backing arrays.
    /// </summary>
    private static void UpdateParametersSGD(
        Tensor<T>[] parameters, Tensor<T>[] gradients,
        T learningRate, INumericOperations<T> numOps)
    {
        if (parameters.Length != gradients.Length)
            throw new InvalidOperationException(
                $"Parameter count ({parameters.Length}) does not match gradient count ({gradients.Length}). " +
                "The compiled plan produced a different number of gradients than expected.");

        for (int i = 0; i < parameters.Length; i++)
        {
            if (gradients[i] is null) continue;

            var paramSpan = parameters[i].Data.Span;
            var gradSpan = gradients[i].AsSpan();
            for (int j = 0; j < paramSpan.Length; j++)
                paramSpan[j] = numOps.Subtract(paramSpan[j], numOps.Multiply(learningRate, gradSpan[j]));
        }
    }
}
