using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using OptimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType;

namespace AiDotNet.Training;

// Local mirror of AiDotNet.Tensors.Engines.Training.DpSgdFusedStep<T>
// (Tensors PR ooples/AiDotNet.Tensors#763). Same API by design — swap when
// Tensors NuGet publishes.

/// <summary>
/// Fused per-example DP-SGD (Abadi et al. 2016 §3, Algorithm 1). Runs each of K
/// per-example forward+backward passes through the compiled plan (with the
/// optimizer step's learning rate set to zero so weights don't drift between
/// examples), extracts each example's gradients from the parameter .Grad
/// tensors, clips against the GLOBAL parameter-vector L2 norm, sums, adds a
/// single Gaussian noise draw, averages, then applies the final aggregate
/// update to the parameters.
///
/// <para>Fused benefit over an eager per-example loop: each per-example
/// forward+backward runs the compiled plan (with GPU-resident parameters,
/// fused kernels, no per-op host↔device round-trip) instead of a fresh
/// non-persistent <see cref="GradientTape{T}"/>. The clip / aggregate / noise
/// steps run in host code because their control flow (per-example L2 norm
/// then min-clip against C) doesn't fit the current compiled-plan capture
/// model — but those are O(params) scalar operations, not the compute
/// bottleneck. The forward+backward per example IS the expensive part, and
/// that's what gets fused.</para>
///
/// <para>Correctness contract: the clip-BEFORE-aggregate order is enforced by
/// the class's internal structure — callers can't reverse it. This preserves
/// the L2-sensitivity bound the Abadi 2016 privacy proof requires.</para>
/// </summary>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public sealed class DpSgdFusedStep<T> : IDisposable
{
    /// <summary>Ambient engine — matches the codebase convention
    /// (<c>protected IEngine Engine => AiDotNetEngine.Current;</c> on the
    /// activation/optimizer/etc. bases). All tensor arithmetic in this class
    /// goes through <c>Engine</c> so the dispatch is uniform.</summary>
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Cached numeric ops for T. Class-level to match the codebase
    /// pattern instead of threading <c>INumericOperations&lt;T&gt;</c> through
    /// method signatures.</summary>
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    // Persistent per-example slots. Refreshed with each example's data before
    // running plan.Step(). The plan captured these references at trace time.
    private Tensor<T>[]? _persistentSlots;
    private ICompiledTrainingPlan<T>? _plan;
    private int[]? _cachedShapeKey;
    private object?[]? _cachedParamIdentities;
    private Tensor<T>[]? _cachedParameters;
    private bool _disposed;

    /// <summary>Whether the fused-resident DP-SGD path is available on this
    /// thread's current engine.</summary>
    public static bool IsAvailable =>
        typeof(T) == typeof(float)
        && AiDotNetEngine.Current is DirectGpuTensorEngine gpu && gpu.SupportsGpu
        && AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;

    /// <summary>
    /// Runs a full DP-SGD training step over <paramref name="batchSize"/> examples.
    /// </summary>
    /// <param name="parameters">Trainable tensors (de-duplicated by reference).
    /// Each gets its own moment buffer.</param>
    /// <param name="perExampleSlotData">Callback returning the fresh slot data
    /// for the given example index. Slot count and shapes must be stable across
    /// example indices (a change triggers a recompile).</param>
    /// <param name="forward">Forward closure that consumes the persistent slots
    /// and returns the predicted output tensor.</param>
    /// <param name="computeLoss">Loss closure — pred + slots → scalar loss.</param>
    /// <param name="batchSize">Number of per-example passes to run.</param>
    /// <param name="clipNorm">Per-example L2 norm clip (Abadi 2016's C).</param>
    /// <param name="noiseMultiplier">Gaussian noise multiplier (Abadi 2016's σ).</param>
    /// <param name="rng">Retained for API compatibility. Noise is now sampled
    /// on-device via <c>Engine.TensorRandomNormalInto</c> so it dispatches to
    /// the engine's RNG (which supports vectorized / on-device random draws).
    /// The caller's <c>rng</c> parameter is not used by the fused path.</param>
    /// <param name="aggregatedGradients">On success, the per-example-clipped +
    /// noised + averaged gradients keyed by parameter tensor reference. Feed
    /// directly into the caller's configured optimizer (via TapeStepContext,
    /// optimizer.Step, etc.) so the caller's optimizer choice (Adam, SGD,
    /// AdamW, ...) applies the DP-clean update. Empty dictionary when TryStep
    /// returns false.</param>
    /// <returns>True when the fused compiled DP-SGD path ran; false to fall
    /// back to eager.</returns>
    public bool TryStep(
        IReadOnlyList<Tensor<T>> parameters,
        Func<int, IReadOnlyList<Tensor<T>>> perExampleSlotData,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>> forward,
        Func<Tensor<T>, IReadOnlyList<Tensor<T>>, Tensor<T>> computeLoss,
        int batchSize,
        double clipNorm,
        double noiseMultiplier,
        Random rng,
        out Dictionary<Tensor<T>, Tensor<T>> aggregatedGradients)
    {
        ThrowIfDisposed();
        aggregatedGradients = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);

        if (!IsAvailable) return false;
        if (parameters is null || parameters.Count == 0) return false;
        if (perExampleSlotData is null) throw new ArgumentNullException(nameof(perExampleSlotData));
        if (forward is null) throw new ArgumentNullException(nameof(forward));
        if (computeLoss is null) throw new ArgumentNullException(nameof(computeLoss));
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (clipNorm <= 0) throw new ArgumentOutOfRangeException(nameof(clipNorm));
        if (rng is null) throw new ArgumentNullException(nameof(rng));

        try
        {
            // Get the first example to establish slot shapes (before compile).
            var firstExample = perExampleSlotData(0);
            if (firstExample is null || firstExample.Count == 0) return false;

            int[] shapeKey = ComputeCompositeShapeKey(firstExample);
            bool shapeChanged = _cachedShapeKey is null || !ShapeKeysEqual(shapeKey, _cachedShapeKey);
            bool paramsChanged = ParameterSetChanged(parameters);

            if (shapeChanged || paramsChanged)
            {
                InvalidateCachedPlan();
                AllocatePersistentSlots(firstExample);
                _cachedShapeKey = shapeKey;
                RememberParameterSet(parameters);
                _cachedParameters = new Tensor<T>[parameters.Count];
                for (int i = 0; i < parameters.Count; i++) _cachedParameters[i] = parameters[i];

                // GPU-residency for the parameters
                if (typeof(T) == typeof(float)
                    && Environment.GetEnvironmentVariable("AIDOTNET_GPU_RESIDENT_PARAMS") != "0")
                {
                    foreach (var p in _cachedParameters) p.Gpu();
                }
            }

            if (_persistentSlots is null || _cachedParameters is null) return false;

            // Trace + compile on first Step. Plan runs forward+backward and
            // applies a zero-LR SGD step so weights don't drift between
            // per-example replays. The .Grad on each parameter is populated
            // by the backward pass regardless of the optimizer's LR.
            if (_plan is null)
            {
                CopySlotData(firstExample);
                using var arenaSuspend = TensorArena.Suspend();
                using var scope = GraphMode.Enable();
                var pred = forward(_persistentSlots);
                var loss = computeLoss(pred, _persistentSlots);
                _plan = scope.CompileTraining(_cachedParameters, loss);
                _plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f, beta1: 0.9f, beta2: 0.999f, eps: 1e-8f, weightDecay: 0f);
            }

            // Per-example accumulators for clipped gradients — zero-init via
            // vectorized TensorFill (avoids per-element scalar zero writes).
            var clippedSums = new Tensor<T>[_cachedParameters.Length];
            for (int p = 0; p < _cachedParameters.Length; p++)
            {
                clippedSums[p] = new Tensor<T>((int[])_cachedParameters[p]._shape.Clone());
                Engine.TensorFill(clippedSums[p], Ops.Zero);
            }

            // Per-example forward+backward via the compiled plan. Each example
            // populates parameter .Grad; the L2 norm computation and clipped
            // accumulation below use vectorized IEngine ops (TensorMultiply for
            // element-wise square, ReduceSum for the norm reduction,
            // TensorMultiplyScalar for the clip scale, TensorAdd for the
            // accumulation) — no per-element scalar loops on the hot path.
            for (int example = 0; example < batchSize; example++)
            {
                var exampleData = example == 0 ? firstExample : perExampleSlotData(example);
                if (exampleData.Count != _persistentSlots.Length) return false;
                CopySlotData(exampleData);

                // Run compiled forward+backward. LR=0 means weights don't update
                // but .Grad on every parameter gets populated by the backward.
                _plan.Step();

                // GLOBAL L2 norm across ALL parameter gradients concatenated —
                // Abadi 2016 L2-sensitivity contract. Per parameter: sum(g²) via
                // TensorMultiply + ReduceSum (vectorized). Sum-of-scalars across
                // parameters is a small O(paramCount) accumulation (unavoidable
                // since each param has its own gradient tensor).
                T normSquared = Ops.Zero;
                for (int p = 0; p < _cachedParameters.Length; p++)
                {
                    var grad = _cachedParameters[p].Grad;
                    if (grad is null) continue;
                    var sq = Engine.TensorMultiply(grad, grad);
                    // axes=null reduces all axes to a scalar.
                    var perParamSum = Engine.ReduceSum(sq, axes: null, keepDims: false);
                    normSquared = Ops.Add(normSquared, perParamSum.Length > 0 ? perParamSum[0] : Ops.Zero);
                }
                double clipFactor = Math.Min(1.0, clipNorm / Math.Sqrt(Ops.ToDouble(normSquared) + 1e-12));
                var clipFactorT = Ops.FromDouble(clipFactor);

                // Accumulate clipped per-example gradient into sums —
                // vectorized: TensorMultiplyScalar + TensorAdd.
                for (int p = 0; p < _cachedParameters.Length; p++)
                {
                    var grad = _cachedParameters[p].Grad;
                    if (grad is null) continue;
                    var scaled = Engine.TensorMultiplyScalar(grad, clipFactorT);
                    clippedSums[p] = Engine.TensorAdd(clippedSums[p], scaled);
                }
            }

            // Build the DP-SGD-clean aggregated gradient dictionary using
            // vectorized IEngine ops: noise is sampled on-device via
            // TensorRandomNormalInto, aggregate is TensorMultiplyScalar(sum, 1/B)
            // + TensorAdd(scaled, noise). Preserves the caller's optimizer
            // choice (Adam / AdamW / SGD) by returning gradients instead of
            // applying an SGD update here.
            double invBatch = 1.0 / batchSize;
            double noiseStdD = clipNorm * noiseMultiplier * invBatch;
            var invBatchT = Ops.FromDouble(invBatch);
            var noiseStdT = Ops.FromDouble(noiseStdD);
            var zeroMean = Ops.Zero;
            for (int p = 0; p < _cachedParameters.Length; p++)
            {
                var scaledSum = Engine.TensorMultiplyScalar(clippedSums[p], invBatchT);
                Tensor<T> noisyAvg;
                if (noiseStdD > 0)
                {
                    var noise = new Tensor<T>(clippedSums[p]._shape);
                    Engine.TensorRandomNormalInto(noise, zeroMean, noiseStdT);
                    noisyAvg = Engine.TensorAdd(scaledSum, noise);
                }
                else
                {
                    noisyAvg = scaledSum;
                }
                aggregatedGradients[_cachedParameters[p]] = noisyAvg;
            }

            return true;
        }
        catch (NotSupportedException)
        {
            InvalidateCachedPlan();
            return false;
        }
        catch (InvalidOperationException)
        {
            InvalidateCachedPlan();
            return false;
        }
    }

    /// <summary>Invalidates the cached compiled plan and persistent slots.
    /// Call when model structure changes; not needed when only data changes.</summary>
    public void Invalidate() => InvalidateCachedPlan();

    private void InvalidateCachedPlan()
    {
        _plan?.Dispose();
        _plan = null;
        _persistentSlots = null;
        _cachedShapeKey = null;
        _cachedParamIdentities = null;
        _cachedParameters = null;
    }

    private void AllocatePersistentSlots(IReadOnlyList<Tensor<T>> firstExample)
    {
        _persistentSlots = new Tensor<T>[firstExample.Count];
        for (int i = 0; i < firstExample.Count; i++)
        {
            _persistentSlots[i] = new Tensor<T>((int[])firstExample[i]._shape.Clone());
        }
    }

    private void CopySlotData(IReadOnlyList<Tensor<T>> fresh)
    {
        if (_persistentSlots is null) return;
        for (int i = 0; i < _persistentSlots.Length; i++)
        {
            var slot = _persistentSlots[i];
            var src = fresh[i];
            if (src.Length != slot.Length)
                throw new InvalidOperationException(
                    $"DpSgdFusedStep: per-example slot {i} shape changed mid-batch. " +
                    $"Expected {slot.Length} elements, got {src.Length}.");
            src.AsSpan().CopyTo(slot.AsWritableSpan());
        }
    }

    private static int[] ComputeCompositeShapeKey(IReadOnlyList<Tensor<T>> slots)
    {
        int total = slots.Count;
        for (int i = 0; i < slots.Count; i++) total += slots[i]._shape.Length;
        var key = new int[total];
        int idx = 0;
        for (int i = 0; i < slots.Count; i++)
        {
            var s = slots[i]._shape;
            for (int d = 0; d < s.Length; d++) key[idx++] = s[d];
            key[idx++] = -1 - i;
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
        if (_cachedParamIdentities is null) return true;
        if (_cachedParamIdentities.Length != parameters.Count) return true;
        for (int i = 0; i < parameters.Count; i++)
            if (!ReferenceEquals(_cachedParamIdentities[i], parameters[i])) return true;
        return false;
    }

    private void RememberParameterSet(IReadOnlyList<Tensor<T>> parameters)
    {
        _cachedParamIdentities = new object?[parameters.Count];
        for (int i = 0; i < parameters.Count; i++) _cachedParamIdentities[i] = parameters[i];
    }

    public void Dispose()
    {
        if (_disposed) return;
        InvalidateCachedPlan();
        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(DpSgdFusedStep<T>));
    }
}
