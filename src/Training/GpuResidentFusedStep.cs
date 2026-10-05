using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using OptimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType;

namespace AiDotNet.Training;

/// <summary>
/// Shared entry point for GPU-resident fused training steps in classes that don't
/// inherit from <c>NeuralNetworkBase</c> or <c>TimeSeriesModelBase</c> but still want
/// their <c>Train</c> to route forward + backward + optimizer through a single on-device
/// compiled plan. Wraps <see cref="CompiledTapeTrainingStep{T}.TryStepWithFusedOptimizer"/>
/// with an optimizer-config resolver (converts the model's runtime <c>IGradientBasedOptimizer</c>
/// into the fused-plan's <see cref="OptimizerType"/> + hyperparameters).
///
/// <para>Use pattern (mirrors <c>NeuralNetworkBase.TrainWithFusedStep</c>):</para>
/// <code>
/// if (CanTrainOnGpu && trainableLayers.Count > 0
///     &amp;&amp; GpuResidentFusedStep&lt;T&gt;.TryStep(
///         trainableLayers, input, target, Forward, loss.ComputeTapeLoss, _optimizer, out T fusedLoss))
/// {
///     LastLoss = fusedLoss;
///     return;
/// }
/// // eager fallback here
/// </code>
/// </summary>
/// <typeparam name="T">Numeric type (float supported end-to-end; other types fall back to eager).</typeparam>
internal static class GpuResidentFusedStep<T>
{
    /// <summary>
    /// Asks the optimizer itself how it maps onto the fused kernel (<see cref="Optimizers.Fused.IFusedOptimizerSpec"/>),
    /// the same mapping <c>NeuralNetworkBase</c>'s fused step uses. Returns false - eager fallback - for an
    /// optimizer that has no fused equivalent or declines in its current configuration.
    /// </summary>
    /// <remarks>
    /// This used to match the optimizer's CLASS NAME ("Adam" / "AdamW" / "SGD" substrings) and read
    /// <c>Options.InitialLearningRate</c> by reflection. AdaMax, Nadam and RAdam all contain "adam", so they -
    /// and Adam with UseAMSGrad, momentum SGD and every scheduled or masked configuration - silently trained
    /// as plain Adam/SGD at the initial learning rate.
    /// </remarks>
    public static bool TryResolveOptimizerConfig(object? optimizer, out Optimizers.Fused.FusedOptimizerConfig config)
    {
        config = default;
        return optimizer is Optimizers.Fused.IFusedOptimizerSpec spec && spec.TryGetFusedOptimizerConfig(out config);
    }

    /// <summary>
    /// One-shot: runs a fused-resident training step if all preconditions hold
    /// (float, DirectGpu engine, compilation enabled, supported optimizer, at least
    /// one trainable layer) and returns the loss via <paramref name="lossValue"/>.
    /// Returns false when the fused path can't engage (caller must run its eager fallback).
    /// </summary>
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
        if (!IsGpuResidentAvailable) return false;
        // A callsite with only extras (no layers) is a valid config — e.g. a model
        // whose whole training surface is raw trainable tensors (learned scalars).
        if ((layers is null || layers.Count == 0) && (extraTensors is null || extraTensors.Count == 0))
            return false;
        if (!TryResolveOptimizerConfig(optimizer, out var cfg))
            return false;
        return CompiledTapeTrainingStep<T>.TryStepWithFusedOptimizer(
            layers ?? System.Array.Empty<ITrainableLayer<T>>(),
            input, target, forward, computeLoss,
            cfg.Type, cfg.LearningRate, cfg.Beta1, cfg.Beta2, cfg.Epsilon, cfg.WeightDecay, out lossValue,
            maxGradNorm: maxGradNorm,
            lrSchedule: cfg.Schedule,
            // The optimizer INSTANCE, not just its hyperparameters: its moments live in the compiled plan, and the
            // plan links itself to this instance so a checkpoint of the optimizer carries them.
            eagerOptimizer: optimizer as IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>,
            useBf16Moments: cfg.UseBf16Moments,
            extraTensors: extraTensors,
            fusedExtras: cfg.Extras,
            onGradients: onGradients,
            owner: owner);
    }

    /// <summary>
    /// True when the fused-resident training path is reachable on this thread's
    /// current engine (mirrors <c>TimeSeriesModelBase.CanTrainOnGpu</c> /
    /// <c>NeuralNetworkBase.CanTrainOnGpu</c>). Three conditions must hold:
    /// T == float, the current engine is a <see cref="DirectGpuTensorEngine"/>
    /// with GPU available, and graph compilation is enabled. When false, callers
    /// should stay on their eager tape+optimizer path.
    /// </summary>
    public static bool IsGpuResidentAvailable =>
        typeof(T) == typeof(float)
        && AiDotNetEngine.Current is DirectGpuTensorEngine gpu && gpu.SupportsGpu
        && AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;
}
