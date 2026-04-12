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
