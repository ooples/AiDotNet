using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Dtype-safe wrapper around <see cref="IEngine.TensorNanToNum{T}"/> for non-<see cref="IConvertible"/>
/// element types (such as <see cref="Half"/>) on published <c>AiDotNet.Tensors 0.151.2</c>.
/// </summary>
internal static class NanToNumHelper
{
    public static Tensor<T> Apply<T>(
        IEngine engine,
        Tensor<T> tensor,
        double? nan = null,
        double? posinf = null,
        double? neginf = null)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        if (typeof(IConvertible).IsAssignableFrom(typeof(T)))
        {
            return engine.TensorNanToNum(tensor, nan, posinf, neginf);
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T nanValue = ops.FromDouble(nan ?? 0.0);
        T posInfValue = posinf.HasValue ? ops.FromDouble(posinf.Value) : ops.MaxValue;
        T negInfValue = neginf.HasValue ? ops.FromDouble(neginf.Value) : ops.MinValue;

        Tensor<T> posFill;
        Tensor<T> negFill;
        Tensor<T> nanFill;
        using (new NoGradScope<T>())
        using (GraphMode.SuspendRecording())
        {
            posFill = Tensor<T>.CreateDefault(tensor._shape, posInfValue);
            negFill = Tensor<T>.CreateDefault(tensor._shape, negInfValue);
            nanFill = Tensor<T>.CreateDefault(tensor._shape, nanValue);
        }

        var nonNanMask = engine.TensorEquals(tensor, tensor);
        var zeros = engine.TensorZerosLike(tensor);
        var finiteProbe = engine.TensorWhere(nonNanMask, tensor, zeros);
        var withoutPosInf = engine.TensorWhere(engine.TensorGreaterThan(finiteProbe, ops.MaxValue), posFill, tensor);
        var withoutInf = engine.TensorWhere(engine.TensorLessThan(finiteProbe, ops.MinValue), negFill, withoutPosInf);
        return engine.TensorWhere(nonNanMask, withoutInf, nanFill);
    }

    public static void ApplyInPlace<T>(
        IEngine engine,
        Tensor<T> tensor,
        double? nan = null,
        double? posinf = null,
        double? neginf = null)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        if (typeof(IConvertible).IsAssignableFrom(typeof(T)))
        {
            engine.TensorNanToNumInPlace(tensor, nan, posinf, neginf);
            return;
        }

        var sanitized = Apply(engine, tensor, nan, posinf, neginf);
        engine.TensorCopyInPlace(tensor, sanitized);
    }
}
