using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides centralized helper methods for applying activation functions with optimal performance.
/// Uses Engine methods (GPU/SIMD) for known activation types, falls back to standard activation otherwise.
/// </summary>
/// <remarks>
/// This class consolidates activation type-checking logic in one place, following DRY (Don't Repeat Yourself)
/// and SOLID principles. All layers should use these methods instead of duplicating if-else chains.
/// </remarks>
public static class ActivationHelper
{
    /// <summary>
    /// Applies a vector activation function to a tensor using Engine methods when possible.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="activation">The vector activation function to apply.</param>
    /// <param name="input">The input tensor to activate.</param>
    /// <param name="engine">The engine to use for optimized operations.</param>
    /// <returns>The activated tensor, or input unchanged if activation is null.</returns>
    public static Tensor<T> ApplyActivation<T>(IVectorActivationFunction<T>? activation, Tensor<T> input, IEngine engine)
    {
        if (activation == null)
            return input;

        // Use Engine methods for optimized activations (CPU SIMD / GPU kernels)
        if (activation is TanhActivation<T>)
            return engine.Tanh(input);
        else if (activation is SigmoidActivation<T>)
            return engine.Sigmoid(input);
        else if (activation is ReLUActivation<T>)
            return engine.ReLU(input);
        else if (activation is GELUActivation<T>)
            return engine.GELU(input);
        else if (activation is MishActivation<T>)
            return engine.Mish(input);
        else if (activation is SwishActivation<T> || activation is SiLUActivation<T>)
            return engine.Swish(input);
        else if (activation is ELUActivation<T>)
            return engine.ELU(input);
        else
            // Fall back to activation function for custom types
            return activation.Activate(input);
    }

    /// <summary>
    /// Applies a vector activation function to a vector using Engine methods when possible.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.).</typeparam>
    /// <param name="activation">The vector activation function to apply.</param>
    /// <param name="input">The input vector to activate.</param>
    /// <param name="engine">The engine to use for optimized operations.</param>
    /// <returns>The activated vector, or input unchanged if activation is null.</returns>
    public static Vector<T> ApplyActivation<T>(IVectorActivationFunction<T>? activation, Vector<T> input, IEngine engine)
    {
        if (activation == null)
            return input;

        // Use Engine methods for optimized activations (CPU SIMD / GPU kernels)
        if (activation is TanhActivation<T>)
            return engine.Tanh(input);
        else if (activation is SigmoidActivation<T>)
            return engine.Sigmoid(input);
        else if (activation is ReLUActivation<T>)
            return engine.ReLU(input);
        else if (activation is GELUActivation<T>)
            return engine.GELU(input);
        else if (activation is MishActivation<T>)
            return engine.Mish(input);
        else if (activation is SwishActivation<T> || activation is SiLUActivation<T>)
            return engine.Swish(input);
        else if (activation is ELUActivation<T>)
            return engine.ELU(input);
        else
            // Fall back to activation function for custom types
            return activation.Activate(input);
    }
}
