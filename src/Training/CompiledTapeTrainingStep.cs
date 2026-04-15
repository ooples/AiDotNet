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
    [ThreadStatic]
    private static object? _lastConfiguredPlan;

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

            var parameters = _cachedParameters ??= CollectParameterArray(layers);

            // Zero gradients before forward pass
            foreach (var layer in layers)
                layer.ZeroGrad();

            // Get or compile training plan (cached by shape internally)
            var plan = cache.GetOrCompileTraining(
                (int[])input._shape.Clone(),
                () =>
                {
                    var predicted = forward(input);
                    computeLoss(predicted, target);
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
        _lastConfiguredPlan = null;
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
    public static bool TryStepWithFusedOptimizer(
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

            var parameters = _cachedParameters ??= CollectParameterArray(layers);

            foreach (var layer in layers)
                layer.ZeroGrad();

            var plan = cache.GetOrCompileTraining(
                (int[])input._shape.Clone(),
                () =>
                {
                    var predicted = forward(input);
                    computeLoss(predicted, target);
                },
                parameters);

            // Configure the fused optimizer exactly once per compiled plan.
            // Re-calling ConfigureOptimizer resets the Adam m/v buffers — we must
            // not do that per step. Track plan identity via ThreadStatic.
            if (!ReferenceEquals(_lastConfiguredPlan, plan))
            {
                plan.ConfigureOptimizer(
                    optimizerType,
                    learningRate,
                    beta1,
                    beta2,
                    epsilon,
                    weightDecay);
                _lastConfiguredPlan = plan;
            }

            // Execute forward + backward + fused parameter update in one replay.
            var lossOutput = plan.Step();
            lossValue = lossOutput.Length > 0 ? lossOutput[0] : MathHelper.GetNumericOperations<T>().Zero;
            return true;
        }
        catch
        {
            // Clear the configured-plan cache so next attempt reconfigures fresh.
            _lastConfiguredPlan = null;
            return false;
        }
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
