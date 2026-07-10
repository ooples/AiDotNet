using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
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
///     &amp;&amp; GpuResidentFusedStep&lt;T&gt;.TryResolveOptimizerConfig(_optimizer, out var t, out var lr, out var b1, out var b2, out var eps, out var wd)
///     &amp;&amp; CompiledTapeTrainingStep&lt;T&gt;.TryStepWithFusedOptimizer(
///         trainableLayers, input, target, Forward, loss.ComputeTapeLoss,
///         t, lr, b1, b2, eps, wd, out T fusedLoss))
/// {
///     LastLoss = fusedLoss;
///     return;
/// }
/// // eager fallback here
/// </code>
/// </summary>
/// <typeparam name="T">Numeric type (float supported end-to-end; other types fall back to eager).</typeparam>
public static class GpuResidentFusedStep<T>
{
    /// <summary>
    /// Maps the runtime optimizer instance to the fused-plan optimizer enum + hyperparameters.
    /// Recognises Adam / AdamW / SGD by class name (case-insensitive). Reads the learning rate
    /// from the optimizer's <c>Options.InitialLearningRate</c> via reflection; falls back to
    /// standard defaults if the property isn't available. Returns false for optimizers the fused
    /// path can't handle so callers cleanly fall through to the eager tape+optimizer route.
    /// </summary>
    public static bool TryResolveOptimizerConfig(
        object? optimizer,
        out OptimizerType type,
        out float learningRate,
        out float beta1, out float beta2,
        out float epsilon, out float weightDecay)
    {
        // Sensible defaults matching AiDotNet's AdamOptimizerOptions.
        type = OptimizerType.Adam;
        learningRate = 1e-3f;
        beta1 = 0.9f; beta2 = 0.999f; epsilon = 1e-8f; weightDecay = 0f;

        if (optimizer is null) return false;
        var name = optimizer.GetType().Name;
        if (name.IndexOf("AdamW", StringComparison.OrdinalIgnoreCase) >= 0)
            type = OptimizerType.AdamW;
        else if (name.IndexOf("Adam", StringComparison.OrdinalIgnoreCase) >= 0)
            type = OptimizerType.Adam;
        else if (name.IndexOf("SGD", StringComparison.OrdinalIgnoreCase) >= 0)
            type = OptimizerType.SGD;
        else
            return false; // Unsupported optimizer — let the caller's eager path handle it.

        // Best-effort hyperparameter extraction from Options.InitialLearningRate /
        // Options.Beta1 / .Beta2 / .Epsilon / .WeightDecay. Any missing property keeps the default.
        try
        {
            var flags = System.Reflection.BindingFlags.Instance
                | System.Reflection.BindingFlags.Public
                | System.Reflection.BindingFlags.NonPublic;
            var optsProp = optimizer.GetType().GetProperty("Options", flags);
            var opts = optsProp?.GetValue(optimizer);
            if (opts is null) return true;
            var optsType = opts.GetType();
            var lrVal = optsType.GetProperty("InitialLearningRate")?.GetValue(opts);
            if (lrVal is not null) learningRate = Convert.ToSingle(lrVal);
            var b1Val = optsType.GetProperty("Beta1")?.GetValue(opts);
            if (b1Val is not null) beta1 = Convert.ToSingle(b1Val);
            var b2Val = optsType.GetProperty("Beta2")?.GetValue(opts);
            if (b2Val is not null) beta2 = Convert.ToSingle(b2Val);
            var epsVal = optsType.GetProperty("Epsilon")?.GetValue(opts);
            if (epsVal is not null) epsilon = Convert.ToSingle(epsVal);
            var wdVal = optsType.GetProperty("WeightDecay")?.GetValue(opts);
            if (wdVal is not null) weightDecay = Convert.ToSingle(wdVal);
        }
        catch
        {
            // Keep defaults on any reflection failure.
        }
        return true;
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
        double maxGradNorm = 1.0)
    {
        lossValue = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Zero;
        if (layers is null || layers.Count == 0) return false;
        if (!TryResolveOptimizerConfig(optimizer, out var type, out var lr, out var b1, out var b2, out var eps, out var wd))
            return false;
        return CompiledTapeTrainingStep<T>.TryStepWithFusedOptimizer(
            layers, input, target, forward, computeLoss,
            type, lr, b1, b2, eps, wd, out lossValue,
            maxGradNorm: maxGradNorm);
    }
}
