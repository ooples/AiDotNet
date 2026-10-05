using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// The fused training step for classes that train on the gradient tape without a base class owning a
/// <see cref="TapeTrainingStepper{T}"/>: the generators in NeuralNetworks/SyntheticData, the graph task models,
/// CLAP. One compiled plan does forward, backward and the optimizer update, GPU-resident on a GPU engine and a
/// fused CPU kernel otherwise.
/// </summary>
/// <remarks>
/// <para>Use pattern:</para>
/// <code>
/// if (FusedTrainingStep&lt;T&gt;.TryStep(trainableLayers, input, target, Forward, Loss, _optimizer, out T loss, owner: this))
///     return;
/// // eager fallback here
/// </code>
/// <para>
/// This was <c>GpuResidentFusedStep</c>: GPU-only, and without the checks the other bases run (a plan that stops
/// persisting updates, #1822; a committed plan that cannot continue). It now drives the same
/// <see cref="FusedTrainingSession{T}"/> as every base class, one session per owner.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float and double fuse; other types fall back to eager).</typeparam>
internal static class FusedTrainingStep<T>
{
    private sealed class OwnerState
    {
        public OwnerState(object owner) => Session = new FusedTrainingSession<T>(owner);
        public FusedTrainingSession<T> Session { get; }
        public object? LastOptimizer { get; set; }
    }

    private static readonly ConditionalWeakTable<object, OwnerState> States = new();

    /// <summary>
    /// Asks the optimizer itself how it maps onto the fused kernel (<see cref="Optimizers.Fused.IFusedOptimizerSpec"/>).
    /// False - eager fallback - for an optimizer that has no fused equivalent or declines in its current configuration.
    /// </summary>
    public static bool TryResolveOptimizerConfig(object? optimizer, out Optimizers.Fused.FusedOptimizerConfig config)
    {
        config = default;
        return optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> typed
            && FusedTrainingSession<T>.TryMapToFusedOptimizerConfig(typed, out config);
    }

    /// <summary>
    /// True when the fused path can engage at all: a numeric type the fused kernels support and compilation on. The
    /// engine does not matter; the plan is GPU-resident on a GPU engine and a fused CPU kernel otherwise.
    /// </summary>
    public static bool IsAvailable =>
        (typeof(T) == typeof(float) || typeof(T) == typeof(double))
        && AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;

    /// <summary>
    /// Runs one fused training step and returns true, or returns false when the fused path does not apply (the caller
    /// runs its eager fallback). Throws when a plan that already trained this owner cannot continue, unless the cause
    /// is a device out-of-memory or transient fault (then the plan is dropped and false returned).
    /// </summary>
    /// <param name="maxGradNorm">Global-norm clip for the fused step (0 = none); combined with the optimizer's own.</param>
    /// <param name="owner">The model; keys its compiled plan and fused session. Defaults to the first layer.</param>
    public static bool TryStep(
        IReadOnlyList<ITrainableLayer<T>> layers,
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> computeLoss,
        object? optimizer,
        out T lossValue,
        double maxGradNorm = 1.0,
        IReadOnlyList<Tensor<T>>? extraTensors = null,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients = null,
        object? owner = null)
    {
        lossValue = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Zero;
        layers ??= Array.Empty<ITrainableLayer<T>>();
        if (optimizer is not IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> typed)
            return false;
        object? key = owner
            ?? (layers.Count > 0 ? (object)layers[0] : null)
            ?? (extraTensors is { Count: > 0 } ? (object)extraTensors[0] : null);
        if (key is null)
            return false;

        var state = States.GetValue(key, k => new OwnerState(k));
        // A different optimizer instance starts from its own fresh moments, as it would eagerly; the plan holding
        // the previous optimizer's moments is dropped rather than refused as hyperparameter drift.
        if (!ReferenceEquals(state.LastOptimizer, typed))
        {
            if (state.LastOptimizer is not null)
                state.Session.Reset(stickyDisable: false);
            state.LastOptimizer = typed;
        }

        var outcome = state.Session.TryStep(new FusedTrainingStepRequest<T>
        {
            Layers = layers,
            Input = input,
            Target = target,
            Forward = forward,
            ComputeLoss = computeLoss,
            Optimizer = typed,
            ExtraParameters = extraTensors,
            ModelGradientClip = maxGradNorm,
            OnGradients = onGradients,
        }, out lossValue);

        switch (outcome)
        {
            case FusedStepOutcome.Stepped:
                return true;
            case FusedStepOutcome.CommittedFailure:
                var cause = state.Session.LastFallbackException;
                if (!TapeTrainingStepper<T>.IsGpuOutOfMemoryFailure(cause) && !TapeTrainingStepper<T>.IsGpuTransientFailure(cause))
                    throw TapeTrainingStepper<T>.CommittedPlanCannotContinue(cause);
                state.Session.Reset(stickyDisable: true);
                return false;
            default:
                return false;
        }
    }
}