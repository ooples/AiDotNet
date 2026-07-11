using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using OptimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType;

namespace AiDotNet.Training;

// Local mirror of AiDotNet.Tensors.Engines.Training.MultiSlotFusedStep<T>
// (Tensors PR ooples/AiDotNet.Tensors#763). Same API by design — when that PR
// merges and the AiDotNet.Tensors NuGet publishes, this file gets deleted and
// the `using AiDotNet.Training` in consumers swaps to
// `using AiDotNet.Tensors.Engines.Training`.

/// <summary>
/// Fused-resident training step with N persistent input slots — supersedes the
/// two-slot (input + target) mechanism in higher-level wrappers when a model's
/// training loop needs to refresh more than two tensors per step.
///
/// <para><b>Motivating cases:</b></para>
/// <list type="bullet">
///   <item><b>Batched-per-element diffusion</b> (Ho et al. 2020, HuggingFace
///   diffusers): each step provides <c>[clean_sample, noise, timesteps]</c> where
///   timesteps is a per-batch-element scalar vector. The compiled plan captures
///   references to all three; the trainer refreshes them per step so replays see
///   fresh RNG draws without recompilation.</item>
///   <item><b>Classifier-free-guidance</b> conditional generation: each step
///   provides <c>[latent, text_embedding, class_embedding, timesteps]</c>.</item>
///   <item><b>TFT-style forecasters</b> with static, historical, and future
///   covariates as separate inputs.</item>
///   <item><b>Stochastic training</b> (CSDI, TabDDPM): timestep index and noise
///   sample are per-step data, not compile-time constants — this trainer feeds
///   them as slots.</item>
/// </list>
///
/// <para><b>Contract:</b> caller allocates persistent tensors OR passes fresh
/// data per step; the trainer copies fresh data into stable-reference persistent
/// tensors before every plan.Step() so the compiled graph re-reads current data
/// without needing to recompile. Plan cache is keyed by the composite shapes of
/// ALL slots — a shape change in any slot triggers a recompile.</para>
/// </summary>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public sealed class MultiSlotFusedStep<T> : IDisposable
{
    // Persistent slot tensors — captured by the compiled plan at trace time.
    // Refreshed in place per step by copying fresh caller data into them.
    // Reference stability is essential: the plan replays against these exact
    // tensor instances, so we must never replace them (only mutate contents).
    private Tensor<T>[]? _persistentSlots;

    // Cached compiled plan. Keyed by the composite shape of all slots plus the
    // trainable-layer set identity. A shape change or layer swap invalidates
    // and recompiles on the next Step.
    private ICompiledTrainingPlan<T>? _plan;
    private int[]? _cachedShapeKey;
    private object?[]? _cachedLayerIdentities;
    private Tensor<T>[]? _cachedParameters;

    // Optimizer configuration frozen at first Step. A different optimizer
    // config on a subsequent Step invalidates and reconfigures.
    private (OptimizerType Type, float Lr, float B1, float B2, float Eps, float Wd)? _configuredOptimizer;

    private bool _disposed;

    /// <summary>Whether the fused-resident training step is available on this thread's
    /// current engine (T == float + DirectGpu with GPU available + compilation enabled).</summary>
    public static bool IsAvailable =>
        typeof(T) == typeof(float)
        && AiDotNetEngine.Current is DirectGpuTensorEngine gpu && gpu.SupportsGpu
        && AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;

    /// <summary>
    /// Runs a single training step with N persistent input slots. The trainer
    /// owns the persistent tensors internally; callers pass fresh data on every
    /// call and the trainer copies it into the persistent tensors before running
    /// the compiled plan.
    /// </summary>
    /// <param name="parameters">Trainable tensors that receive optimizer updates.
    /// Must be a de-duplicated list (shared/tied weights collected once).</param>
    /// <param name="zeroGradAction">Optional callback invoked before every step
    /// to clear per-parameter gradient state (e.g. layer-owned .Grad buffers).
    /// The Tensors library doesn't know about the caller's layer abstraction;
    /// callers pass a closure that iterates their layers and calls their own
    /// ZeroGrad. Pass null when no per-step grad reset is required.</param>
    /// <param name="freshSlotData">Fresh tensor data for each slot. Length and
    /// shapes must be stable across calls (a change triggers a recompile).</param>
    /// <param name="forward">Forward closure — receives the persistent slots
    /// (same references every step, refreshed data) and returns the predicted
    /// output tensor.</param>
    /// <param name="computeLoss">Loss closure — receives the forward output and
    /// the same persistent slots (for target/aux access), returns the scalar
    /// loss tensor.</param>
    /// <param name="optimizerType">Fused optimizer kernel to configure.</param>
    /// <param name="learningRate">Optimizer LR.</param>
    /// <param name="beta1">Momentum β₁ for Adam-family.</param>
    /// <param name="beta2">Momentum β₂ for Adam-family.</param>
    /// <param name="epsilon">Adam ε.</param>
    /// <param name="weightDecay">Optimizer weight-decay (AdamW-style).</param>
    /// <param name="lossValue">Scalar loss value from this step.</param>
    /// <returns>True when the fused compiled step ran successfully. Callers must
    /// handle false by falling back to eager tape-based training.</returns>
    public bool TryStep(
        IReadOnlyList<Tensor<T>> parameters,
        Action? zeroGradAction,
        IReadOnlyList<Tensor<T>> freshSlotData,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>> forward,
        Func<Tensor<T>, IReadOnlyList<Tensor<T>>, Tensor<T>> computeLoss,
        OptimizerType optimizerType,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        out T lossValue)
    {
        ThrowIfDisposed();
        lossValue = MathHelper.GetNumericOperations<T>().Zero;

        if (!IsAvailable) return false;
        if (parameters is null || parameters.Count == 0) return false;
        if (freshSlotData is null || freshSlotData.Count == 0) return false;
        if (forward is null) throw new ArgumentNullException(nameof(forward));
        if (computeLoss is null) throw new ArgumentNullException(nameof(computeLoss));

        // Compute composite shape key across all slots. Rebuild persistent
        // tensors and recompile the plan if the key changed OR the parameter
        // set changed identity.
        int[] shapeKey = ComputeCompositeShapeKey(freshSlotData);
        bool shapeChanged = _cachedShapeKey is null || !ShapeKeysEqual(shapeKey, _cachedShapeKey);
        bool paramsChanged = ParameterSetChanged(parameters);
        bool optimizerChanged = OptimizerConfigChanged(optimizerType, learningRate, beta1, beta2, epsilon, weightDecay);

        if (shapeChanged || paramsChanged)
        {
            InvalidateCachedPlan();
            AllocatePersistentSlots(freshSlotData);
            _cachedShapeKey = shapeKey;
            RememberParameterSet(parameters);
        }

        // Copy fresh data into persistent slots BEFORE the plan runs. The plan
        // captured references to these tensors; refreshing their data makes the
        // plan read current-step data without recompilation.
        if (_persistentSlots is null)
            return false;
        for (int i = 0; i < _persistentSlots.Length; i++)
        {
            var slot = _persistentSlots[i];
            var fresh = freshSlotData[i];
            if (fresh.Length != slot.Length)
                return false;
            fresh.AsSpan().CopyTo(slot.AsWritableSpan());
        }

        // Cache the (already-deduplicated) parameter list. The caller
        // guarantees dedup — this class doesn't know about the caller's layer
        // structure so it can't do reference-based dedup on its own.
        if (_cachedParameters is null)
        {
            _cachedParameters = new Tensor<T>[parameters.Count];
            for (int i = 0; i < parameters.Count; i++) _cachedParameters[i] = parameters[i];
            // GPU-residency: mark parameters as GPU-resident so the compiled
            // plan's optimizer branch takes the GPU Adam path (params stay
            // on-device across steps).
            if (typeof(T) == typeof(float)
                && Environment.GetEnvironmentVariable("AIDOTNET_GPU_RESIDENT_PARAMS") != "0"
                && (optimizerType == OptimizerType.Adam
                    || optimizerType == OptimizerType.AdamW
                    || optimizerType == OptimizerType.SGD))
            {
                foreach (var p in _cachedParameters) p.Gpu();
            }
        }

        zeroGradAction?.Invoke();

        try
        {
            // Trace + compile on first call after invalidation.
            if (_plan is null)
            {
                using var arenaSuspend = TensorArena.Suspend();
                using var scope = GraphMode.Enable();
                var pred = forward(_persistentSlots);
                var loss = computeLoss(pred, _persistentSlots);
                _plan = scope.CompileTraining(_cachedParameters, loss);
            }

            // Configure optimizer on first Step OR when config changed.
            if (optimizerChanged || _configuredOptimizer is null)
            {
                _plan.ConfigureOptimizer(optimizerType, learningRate, beta1, beta2, epsilon, weightDecay);
                _configuredOptimizer = (optimizerType, learningRate, beta1, beta2, epsilon, weightDecay);
            }

            var lossTensor = _plan.Step();
            lossValue = lossTensor.Length > 0 ? lossTensor[0] : MathHelper.GetNumericOperations<T>().Zero;
            return true;
        }
        catch (NotSupportedException)
        {
            // The compiled plan can't execute this graph on this engine. Fall
            // back to eager by returning false so the caller runs their eager
            // tape path.
            InvalidateCachedPlan();
            return false;
        }
        catch (InvalidOperationException)
        {
            InvalidateCachedPlan();
            return false;
        }
    }

    /// <summary>Invalidates the cached compiled plan and persistent slots. Call
    /// when the model's layer structure changes. Data-only refreshes don't need
    /// this — just pass fresh data on the next Step.</summary>
    public void Invalidate() => InvalidateCachedPlan();

    private void InvalidateCachedPlan()
    {
        _plan?.Dispose();
        _plan = null;
        _persistentSlots = null;
        _cachedShapeKey = null;
        _cachedLayerIdentities = null;
        _cachedParameters = null;
        _configuredOptimizer = null;
    }

    private void AllocatePersistentSlots(IReadOnlyList<Tensor<T>> freshSlotData)
    {
        _persistentSlots = new Tensor<T>[freshSlotData.Count];
        for (int i = 0; i < freshSlotData.Count; i++)
        {
            _persistentSlots[i] = new Tensor<T>((int[])freshSlotData[i]._shape.Clone());
        }
    }

    private static int[] ComputeCompositeShapeKey(IReadOnlyList<Tensor<T>> slots)
    {
        int total = slots.Count; // one separator per slot
        for (int i = 0; i < slots.Count; i++) total += slots[i]._shape.Length;
        var key = new int[total];
        int idx = 0;
        for (int i = 0; i < slots.Count; i++)
        {
            var s = slots[i]._shape;
            for (int d = 0; d < s.Length; d++) key[idx++] = s[d];
            key[idx++] = -1 - i; // slot separator (distinct per slot index)
        }
        return key;
    }

    private static bool ShapeKeysEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private bool ParameterSetChanged(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_cachedLayerIdentities is null) return true;
        if (_cachedLayerIdentities.Length != parameters.Count) return true;
        for (int i = 0; i < parameters.Count; i++)
            if (!ReferenceEquals(_cachedLayerIdentities[i], parameters[i])) return true;
        return false;
    }

    private void RememberParameterSet(IReadOnlyList<Tensor<T>> parameters)
    {
        _cachedLayerIdentities = new object?[parameters.Count];
        for (int i = 0; i < parameters.Count; i++) _cachedLayerIdentities[i] = parameters[i];
    }

    private bool OptimizerConfigChanged(OptimizerType type, float lr, float b1, float b2, float eps, float wd)
    {
        if (_configuredOptimizer is null) return true;
        var (cType, cLr, cB1, cB2, cEps, cWd) = _configuredOptimizer.Value;
        return cType != type || cLr != lr || cB1 != b1 || cB2 != b2 || cEps != eps || cWd != wd;
    }

    public void Dispose()
    {
        if (_disposed) return;
        InvalidateCachedPlan();
        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MultiSlotFusedStep<T>));
    }
}
