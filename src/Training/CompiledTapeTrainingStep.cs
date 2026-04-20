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

    /// <summary>Gets the count of successful fused-step executions on the calling thread.</summary>
    public static long GetFusedStepCount() => _fusedStepCount;

    /// <summary>Resets the fused-step counter on the calling thread to zero.</summary>
    public static void ResetFusedStepCount() { _fusedStepCount = 0; }

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
            var parameters = _cachedParameters ??= CollectDeduplicatedParameters(layers);

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
    public static void Invalidate()
    {
        _cache?.Invalidate();
        _cachedParameters = null;
        _configuredPlan = null;
        _configuredOptimizerConfig = null;
        // Reset the fused-engagement counter — from this point on, any
        // assertion about "fused ran at least N times" should reflect the
        // new lifecycle.
        _fusedStepCount = 0;
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
        out T lossValue)
    {
        lossValue = MathHelper.GetNumericOperations<T>().Zero;

        if (!TensorCodecOptions.Current.EnableCompilation) return false;
        // Fused optimizer kernels are float-only on the Tensors side.
        if (typeof(T) != typeof(float)) return false;
        // Only SGD, Adam, AdamW are wired through ConfigureOptimizer.
        if (optimizerType is not (AiDotNet.Tensors.Engines.Compilation.OptimizerType.SGD
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adam
            or AiDotNet.Tensors.Engines.Compilation.OptimizerType.AdamW))
            return false;

        try
        {
            var cache = _cache ??= new CompiledModelCache<T>();

            // Force layer initialization before collecting parameters — DenseLayer
            // replaces _weights with a new tensor on first Forward.
            if (_cachedParameters is null)
                forward(input);

            // Use the dedup-aware collector for the fused path. Shared/tied
            // weights (same Tensor<T> instance referenced by multiple layers)
            // would otherwise drive the fused kernel's m/v buffers to update
            // the same parameter twice per step, breaking Adam's moment math.
            var parameters = _cachedParameters ??= CollectDeduplicatedParameters(layers);

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
                    // Tensors 0.50.1 changed GetOrCompileTraining to take a
                    // Func<Tensor<T>> — the trace lambda must return the
                    // scalar output (loss) so the compile-graph has a single
                    // terminal node to differentiate from.
                    var predicted = forward(input);
                    return computeLoss(predicted, target);
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
                // commit to single-plan semantics from here on.
                plan.ConfigureOptimizer(
                    optimizerType,
                    learningRate,
                    beta1,
                    beta2,
                    epsilon,
                    weightDecay);
                _configuredPlan = plan;
                _configuredOptimizerConfig = currentConfig;
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
            System.Diagnostics.Trace.TraceWarning(
                $"CompiledTapeTrainingStep.TryStepWithFusedOptimizer failed, falling back to eager: " +
                $"{ex}");
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
