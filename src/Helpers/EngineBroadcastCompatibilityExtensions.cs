namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Preserves the explicit broadcast call surface used by AiDotNet while consuming
/// AiDotNet.Tensors 0.121.0, whose element-wise engine operations now broadcast implicitly.
/// </summary>
/// <remarks>
/// Tensors 0.121.0 removed these four members from <see cref="IEngine"/> after making the
/// corresponding plain operations follow NumPy/PyTorch broadcasting rules. Keeping the
/// compatibility mapping in one place avoids a large mechanical migration in the model-fix PR
/// and is behaviorally equivalent to the removed interface methods.
/// </remarks>
internal static class EngineBroadcastCompatibilityExtensions
{
    /// <summary>Adds two tensors using the engine's implicit broadcasting rules.</summary>
    internal static Tensor<T> TensorBroadcastAdd<T>(this IEngine engine, Tensor<T> left, Tensor<T> right)
        => engine.TensorAdd(left, right);

    /// <summary>Subtracts two tensors using the engine's implicit broadcasting rules.</summary>
    internal static Tensor<T> TensorBroadcastSubtract<T>(this IEngine engine, Tensor<T> left, Tensor<T> right)
        => engine.TensorSubtract(left, right);

    /// <summary>Multiplies two tensors using the engine's implicit broadcasting rules.</summary>
    internal static Tensor<T> TensorBroadcastMultiply<T>(this IEngine engine, Tensor<T> left, Tensor<T> right)
        => engine.TensorMultiply(left, right);

    /// <summary>Divides two tensors using the engine's implicit broadcasting rules.</summary>
    internal static Tensor<T> TensorBroadcastDivide<T>(this IEngine engine, Tensor<T> left, Tensor<T> right)
        => engine.TensorDivide(left, right);
}
