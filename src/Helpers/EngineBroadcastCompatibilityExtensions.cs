// Declared in AiDotNet's OWN namespace, not in AiDotNet.Tensors.Engines where it started. Injecting a
// type into a dependency's namespace means that if a later AiDotNet.Tensors re-adds these members to
// IEngine, the instance methods win over these extensions everywhere at once, with no build break and
// no way to tell from a call site which one ran. Keeping the shim here makes it ours: it is reached
// through the project's `<Using Include="AiDotNet.Helpers" />` (src/AiDotNet.csproj), so no call site
// changes, and a future re-add on IEngine surfaces as an ordinary ambiguity to resolve deliberately.
namespace AiDotNet.Helpers;

/// <summary>
/// Preserves the explicit broadcast call surface used by AiDotNet while consuming
/// AiDotNet.Tensors 0.122.0, whose element-wise engine operations now broadcast implicitly.
/// </summary>
/// <remarks>
/// <para>
/// Tensors 0.121.0 removed these four members from <see cref="IEngine"/> after making the
/// corresponding plain operations follow NumPy/PyTorch broadcasting rules. Keeping the
/// compatibility mapping in one place avoids a large mechanical migration in the model-fix PR
/// and is behaviorally equivalent to the removed interface methods.
/// </para>
/// <para>
/// VERIFIED AGAINST THE PINNED PACKAGE (0.122.0), because the XML docs shipped with it are stale on
/// this point: <c>TensorAdd</c> and <c>TensorDivide</c> still document "thrown when tensor shapes
/// don't match" and only <c>TensorMultiply</c> mentions broadcasting. The shipped behavior is
/// broadcasting for all four. Measured on <c>DirectGpuTensorEngine</c> for
/// <c>[2,3] op [3]</c> (trailing), <c>[2,3] op [1,3]</c> (leading unit axis) and
/// <c>[2,3] op [2,1]</c> (trailing unit axis): every combination returns <c>[2,3]</c> with the
/// broadcast result rather than throwing. If a future bump reverts that, these four call sites are
/// the only ones that need to grow an explicit broadcast again.
/// </para>
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
