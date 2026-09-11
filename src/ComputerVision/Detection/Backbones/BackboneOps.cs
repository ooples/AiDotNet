using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Shared tensor primitives reused by the detection backbones. Every op here must go through the
/// engine so the gradient tape records it; the ResNet stem's max pool, which used to live here as
/// an element loop, is now <see cref="AiDotNet.ComputerVision.CvTensorOps{T}.MaxPoolPadded"/>.
/// </summary>
internal static class BackboneOps<T>
{
    /// <summary>
    /// Element-wise residual addition (a + b in-place into a fresh tensor of a's shape).
    /// Validates BOTH length and rank-by-rank shape so a same-element-count but
    /// different-rank mismatch (e.g. [1,16,8,8] vs [16,8,1,8]) is caught instead
    /// of silently producing semantically-wrong activations.
    /// </summary>
    public static Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length || a._shape.Length != b._shape.Length)
            throw new InvalidOperationException(
                $"BackboneOps.AddResidual shape mismatch: [{string.Join(",", a._shape)}] vs [{string.Join(",", b._shape)}].");
        for (int axis = 0; axis < a._shape.Length; axis++)
        {
            if (a._shape[axis] != b._shape[axis])
                throw new InvalidOperationException(
                    $"BackboneOps.AddResidual shape mismatch at axis {axis}: " +
                    $"[{string.Join(",", a._shape)}] vs [{string.Join(",", b._shape)}].");
        }
        // Engine add, not an element loop: a residual skip that drops to scalars severs the gradient
        // for every layer before it, which in a deep backbone is nearly all of them.
        return AiDotNetEngine.Current.TensorAdd(a, b);
    }

    // ApplyReLU / ApplySiLU / ApplySwish removed — backbones (ResNet, CSPDarknet,
    // EfficientNet, SwinTransformer) now accept a configurable IActivationFunction<T>?
    // ctor parameter that resolves to the paper-correct default when null.
}
