using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using OptimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType;

namespace AiDotNet.Training;

// Local mirror of AiDotNet.Tensors.Engines.Training.WganGpFusedStep<T>
// (Tensors PR ooples/AiDotNet.Tensors#763). Same API by design — swap when
// Tensors NuGet publishes.

/// <summary>
/// Fused WGAN-GP critic training step (Gulrajani et al. 2017). Runs the
/// combined <c>E[D(fake)] − E[D(real)] + λ·(‖∇_x̃ D(x̃)‖₂ − 1)²</c> objective
/// through the compiled fused plan, where the gradient-penalty term's inner
/// backward is expressed as a nested-tape computation.
///
/// <para><b>Fused benefit:</b> the outer forward + backward + optimizer step
/// are captured in ONE compiled plan. The inner ∇_x̃ D(x̃) gradient is captured
/// on the outer tape via <c>createGraph=true</c> — this is why the compiled
/// backward's <c>createGraph=true</c> support (Phase 4C on the engine side)
/// is a prerequisite. The plan replays with fresh real/fake batches per step.</para>
///
/// <para><b>Correctness contract:</b> the inner gradient's ops are recorded on
/// the outer tape so the outer backward can differentiate the penalty into disc
/// weights (which the pre-fix AiDotNet code failed to do — see issue #1844).
/// This class's structure enforces the createGraph=true call on the inner tape.</para>
/// </summary>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public sealed class WganGpFusedStep<T> : IDisposable
{
    /// <summary>Ambient engine — matches the codebase convention
    /// (<c>protected IEngine Engine => AiDotNetEngine.Current;</c> on the
    /// activation/optimizer/etc. bases).</summary>
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Cached numeric ops for T. Also matches the codebase pattern:
    /// class-level <c>Ops</c> instead of threading <c>INumericOperations&lt;T&gt;</c>
    /// through method signatures.</summary>
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    // Persistent per-batch slots for (real, fake). Refreshed per step.
    private Tensor<T>[]? _persistentSlots;
    private ICompiledTrainingPlan<T>? _plan;
    private int[]? _cachedShapeKey;
    private object?[]? _cachedParamIdentities;
    private Tensor<T>[]? _cachedParameters;
    private (OptimizerType Type, float Lr, float B1, float B2, float Eps, float Wd)? _configuredOptimizer;
    private bool _disposed;

    public static bool IsAvailable =>
        typeof(T) == typeof(float)
        && AiDotNetEngine.Current is DirectGpuTensorEngine gpu && gpu.SupportsGpu
        && AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;

    /// <summary>
    /// Runs one WGAN-GP critic step with fresh real + fake batches.
    /// </summary>
    /// <param name="discParameters">Critic's trainable tensors (de-duplicated).</param>
    /// <param name="realBatch">Fresh real batch, shape <c>[B, ...]</c>.</param>
    /// <param name="fakeBatch">Fresh fake batch (same shape as realBatch).</param>
    /// <param name="discForward">Critic forward: input tensor → scalar-per-example scores.</param>
    /// <param name="epsilonSampler">Callback returning per-batch interpolation
    /// weights <c>ε ∈ [0, 1]^B</c>, shape <c>[B, 1, ...]</c> broadcasting over the sample.
    /// The interpolated point is <c>x̃ = ε · real + (1−ε) · fake</c>.</param>
    /// <param name="gradientPenaltyWeight">λ from Gulrajani 2017.</param>
    /// <param name="optimizerType">Fused optimizer kernel.</param>
    /// <param name="learningRate">Optimizer LR.</param>
    /// <param name="beta1">Adam β₁.</param>
    /// <param name="beta2">Adam β₂.</param>
    /// <param name="epsilon">Adam ε.</param>
    /// <param name="weightDecay">Optimizer weight decay.</param>
    /// <param name="lossValue">Scalar loss (Wasserstein + λ·GP).</param>
    /// <returns>True on successful fused step, false to fall back to eager.</returns>
    public bool TryStep(
        IReadOnlyList<Tensor<T>> discParameters,
        Tensor<T> realBatch,
        Tensor<T> fakeBatch,
        Func<Tensor<T>, Tensor<T>> discForward,
        Func<int, Tensor<T>> epsilonSampler,
        double gradientPenaltyWeight,
        OptimizerType optimizerType,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        out T lossValue)
    {
        ThrowIfDisposed();
        lossValue = Ops.Zero;

        if (!IsAvailable) return false;
        if (discParameters is null || discParameters.Count == 0) return false;
        if (realBatch is null || fakeBatch is null) return false;
        if (discForward is null) throw new ArgumentNullException(nameof(discForward));
        if (epsilonSampler is null) throw new ArgumentNullException(nameof(epsilonSampler));
        if (!ShapeMatches(realBatch, fakeBatch))
            throw new ArgumentException(
                $"realBatch shape [{string.Join(",", realBatch._shape)}] does not match fakeBatch shape [{string.Join(",", fakeBatch._shape)}].",
                nameof(fakeBatch));

        try
        {
            int batchSize = realBatch.Shape[0];
            int[] shapeKey = ComputeShapeKey(realBatch);
            bool shapeChanged = _cachedShapeKey is null || !ShapeKeysEqual(shapeKey, _cachedShapeKey);
            bool paramsChanged = ParameterSetChanged(discParameters);
            bool optimizerChanged = OptimizerConfigChanged(optimizerType, learningRate, beta1, beta2, epsilon, weightDecay);

            if (shapeChanged || paramsChanged)
            {
                InvalidateCachedPlan();
                AllocatePersistentSlots(realBatch);
                _cachedShapeKey = shapeKey;
                RememberParameterSet(discParameters);
                _cachedParameters = new Tensor<T>[discParameters.Count];
                for (int i = 0; i < discParameters.Count; i++) _cachedParameters[i] = discParameters[i];
                if (typeof(T) == typeof(float)
                    && Environment.GetEnvironmentVariable("AIDOTNET_GPU_RESIDENT_PARAMS") != "0")
                {
                    foreach (var p in _cachedParameters) p.Gpu();
                }
            }

            if (_persistentSlots is null || _cachedParameters is null) return false;

            // Copy fresh real/fake into persistent slots BEFORE running the plan.
            realBatch.AsSpan().CopyTo(_persistentSlots[0].AsWritableSpan());
            fakeBatch.AsSpan().CopyTo(_persistentSlots[1].AsWritableSpan());
            var eps01 = epsilonSampler(batchSize);
            if (_persistentSlots.Length < 3)
            {
                _persistentSlots = ExtendSlots(_persistentSlots, eps01._shape);
            }
            eps01.AsSpan().CopyTo(_persistentSlots[2].AsWritableSpan());

            // Trace + compile on first call.
            if (_plan is null)
            {
                using var arenaSuspend = TensorArena.Suspend();
                using var scope = GraphMode.Enable();
                var loss = BuildWganGpLoss(discForward, gradientPenaltyWeight);
                _plan = scope.CompileTraining(_cachedParameters, loss);
            }

            if (optimizerChanged || _configuredOptimizer is null)
            {
                _plan.ConfigureOptimizer(optimizerType, learningRate, beta1, beta2, epsilon, weightDecay);
                _configuredOptimizer = (optimizerType, learningRate, beta1, beta2, epsilon, weightDecay);
            }

            var lossTensor = _plan.Step();
            lossValue = lossTensor.Length > 0 ? lossTensor[0] : Ops.Zero;
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

    /// <summary>
    /// Builds the tape-recorded WGAN-GP loss <c>E[D(fake)] − E[D(real)] +
    /// λ·(‖∇_x̃ D(x̃)‖₂ − 1)²</c>. The inner gradient uses <c>createGraph=true</c>
    /// so its ops record on the outer (compilation) tape, letting the compiled
    /// backward differentiate the penalty into disc weights.
    /// </summary>
    private Tensor<T> BuildWganGpLoss(
        Func<Tensor<T>, Tensor<T>> discForward,
        double gradientPenaltyWeight)
    {
        if (_persistentSlots is null || _persistentSlots.Length < 3)
            throw new InvalidOperationException("WganGpFusedStep: persistent slots not allocated.");

        var real = _persistentSlots[0];
        var fake = _persistentSlots[1];
        var epsilon01 = _persistentSlots[2];

        // Wasserstein term: E[D(fake)] − E[D(real)].
        var realScores = discForward(real);
        var fakeScores = discForward(fake);
        var scoreAxes = System.Linq.Enumerable.Range(0, realScores.Shape.Length).ToArray();
        var wasserstein = Engine.TensorSubtract(
            Engine.ReduceMean(fakeScores, scoreAxes, keepDims: false),
            Engine.ReduceMean(realScores, scoreAxes, keepDims: false));

        // Interpolated x̃ = ε · real + (1 − ε) · fake, broadcasting ε over the
        // sample. epsilon01 has shape [B, 1, ...] matching the sample.
        var interpolated = Engine.TensorAdd(
            Engine.TensorBroadcastMultiply(epsilon01, real),
            Engine.TensorBroadcastMultiply(
                Engine.TensorSubtract(OnesLike(epsilon01), epsilon01),
                fake));

        // Inner gradient penalty. Run the disc forward on the interpolated point
        // under a nested tape with createGraph=true so the inner backward's ops
        // record on the outer tape — enabling the outer backward (this plan's
        // compiled backward) to differentiate the penalty into disc weights.
        Tensor<T> penalty;
        using (var innerTape = new GradientTape<T>())
        {
            var interpScores = discForward(interpolated);
            var summedScores = Engine.ReduceSum(interpScores,
                System.Linq.Enumerable.Range(0, interpScores.Shape.Length).ToArray(),
                keepDims: false);
            // createGraph=true records the inner backward's ops on the outer tape.
            var innerGrads = innerTape.ComputeGradients(summedScores, new[] { interpolated }, createGraph: true);
            var inputGrad = innerGrads.TryGetValue(interpolated, out var g)
                ? g
                : new Tensor<T>(interpolated._shape);

            int batchSize = interpolated.Shape[0];
            int elementsPer = interpolated.Length / batchSize;
            var gradReshaped = Engine.Reshape(inputGrad, new[] { batchSize, elementsPer });
            var gradSquared = Engine.TensorMultiply(gradReshaped, gradReshaped);
            var normSquared = Engine.ReduceSum(gradSquared, new[] { 1 }, keepDims: false);
            var norm = Engine.TensorSqrt(Engine.TensorAddScalar(normSquared, Ops.FromDouble(1e-12)));
            var deviation = Engine.TensorSubtract(norm, OnesLike(norm));
            var perExPenalty = Engine.TensorMultiply(deviation, deviation);
            var penaltyAxes = System.Linq.Enumerable.Range(0, perExPenalty.Shape.Length).ToArray();
            penalty = Engine.ReduceMean(perExPenalty, penaltyAxes, keepDims: false);
        }

        // Total = wasserstein + λ · penalty.
        var weightedPenalty = Engine.TensorMultiplyScalar(penalty, Ops.FromDouble(gradientPenaltyWeight));
        var alignedPenalty = weightedPenalty._shape.SequenceEqual(wasserstein._shape)
            ? weightedPenalty
            : Engine.Reshape(weightedPenalty, wasserstein._shape);
        return Engine.TensorAdd(wasserstein, alignedPenalty);
    }

    private static Tensor<T> OnesLike(Tensor<T> reference)
    {
        // Vectorized fill through the engine — dispatches to on-device
        // fill (GPU/SIMD) instead of a per-element scalar write loop.
        var ones = new Tensor<T>(reference._shape);
        Engine.TensorFill(ones, Ops.One);
        return ones;
    }

    public void Invalidate() => InvalidateCachedPlan();

    private void InvalidateCachedPlan()
    {
        _plan?.Dispose();
        _plan = null;
        _persistentSlots = null;
        _cachedShapeKey = null;
        _cachedParamIdentities = null;
        _cachedParameters = null;
        _configuredOptimizer = null;
    }

    private void AllocatePersistentSlots(Tensor<T> realBatch)
    {
        // Slot 0: real, Slot 1: fake, Slot 2: epsilon (allocated on first Step
        // when epsilonSampler is called).
        _persistentSlots = new Tensor<T>[3];
        _persistentSlots[0] = new Tensor<T>((int[])realBatch._shape.Clone());
        _persistentSlots[1] = new Tensor<T>((int[])realBatch._shape.Clone());
        // Slot 2 shape depends on epsilon shape; allocated at first refresh.
        _persistentSlots[2] = null!;  // placeholder — replaced on first Refresh.
    }

    private static Tensor<T>[] ExtendSlots(Tensor<T>[] slots, int[] epsShape)
    {
        // Called if slot 2 was allocated as a null placeholder.
        slots[2] = new Tensor<T>((int[])epsShape.Clone());
        return slots;
    }

    private static bool ShapeMatches(Tensor<T> a, Tensor<T> b)
    {
        if (a.Rank != b.Rank) return false;
        for (int i = 0; i < a.Rank; i++) if (a.Shape[i] != b.Shape[i]) return false;
        return true;
    }

    private static int[] ComputeShapeKey(Tensor<T> t)
    {
        var key = new int[t.Rank];
        for (int i = 0; i < t.Rank; i++) key[i] = t.Shape[i];
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
        if (_disposed) throw new ObjectDisposedException(nameof(WganGpFusedStep<T>));
    }
}
