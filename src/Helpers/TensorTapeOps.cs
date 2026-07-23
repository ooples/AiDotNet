using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Tape-tracked element-wise tensor operations that wrap engine ops which
/// otherwise bypass the autodiff graph.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="IEngine.TensorMultiplyScalar{T}(Tensor{T}, T)"/> and
/// <see cref="IEngine.TensorAddScalar{T}(Tensor{T}, T)"/> are *not* recorded on the
/// autodiff tape — using them inside a layer's <c>Forward</c> path leaves the
/// downstream trainable parameters with zero gradients under tape-based training
/// (the LayerTestBase TapeGradient assertion specifically names this as a common
/// cause). The wrappers here construct a constant tensor of the operand's shape
/// and route the operation through <see cref="IEngine.TensorMultiply{T}"/> /
/// <see cref="IEngine.TensorAdd{T}"/>, both of which are tape-connected; the
/// constant tensor itself is not a trainable leaf, so backward only propagates
/// gradients through the original tensor as the math demands.
/// </para>
/// <para>
/// Cost: one fresh constant tensor allocation per call. For hot inner loops with
/// fixed scalar+shape pairs, lift the tensor with <see cref="LinearAlgebra.Tensor{T}.CreateDefault"/>
/// outside the loop and use <see cref="IEngine.TensorMultiply{T}"/> /
/// <see cref="IEngine.TensorAdd{T}"/> directly. For one-off element-wise scalar
/// ops in <c>Forward</c>, these wrappers are the simplest correct replacement.
/// </para>
/// </remarks>
public static class TensorTapeOps
{
    /// <summary>
    /// Tape-tracked replacement for <see cref="IEngine.TensorMultiplyScalar{T}"/>:
    /// returns <c>tensor * scalar</c> with the autodiff tape recording the op.
    /// </summary>
    public static Tensor<T> TapeMultiplyScalar<T>(IEngine engine, Tensor<T> tensor, T scalar)
    {
        var scalarTensor = Tensor<T>.CreateDefault(tensor._shape, scalar);
        return engine.TensorMultiply(tensor, scalarTensor);
    }

    /// <summary>
    /// Tape-tracked replacement for <see cref="IEngine.TensorAddScalar{T}"/>:
    /// returns <c>tensor + scalar</c> with the autodiff tape recording the op.
    /// </summary>
    public static Tensor<T> TapeAddScalar<T>(IEngine engine, Tensor<T> tensor, T scalar)
    {
        var scalarTensor = Tensor<T>.CreateDefault(tensor._shape, scalar);
        return engine.TensorAdd(tensor, scalarTensor);
    }
}
