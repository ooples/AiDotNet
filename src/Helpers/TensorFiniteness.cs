using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Finite / non-finite element masks that work for every numeric type the library supports.
/// </summary>
/// <remarks>
/// The engine's <c>TensorIsFinite</c> converts each element through <c>IConvertible</c>, which <c>Half</c> does
/// not implement, so it throws <c>InvalidCastException</c> on a <c>Tensor&lt;Half&gt;</c>. These masks use only NaN and magnitude comparisons, which every type supports.
/// </remarks>
internal static class TensorFiniteness
{
    /// <summary>
    /// 1 where the element is NaN or an infinity, 0 where it is finite.
    /// </summary>
    public static Tensor<T> NonFiniteMask<T>(Tensor<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();

        // An infinity is the only value whose magnitude exceeds the largest finite one; NaN compares false.
        var infinite = engine.TensorGreaterThan(engine.TensorAbs(values), ops.MaxValue);
        return engine.TensorWhere(engine.TensorIsNan(values), engine.TensorOnesLike(values), infinite);
    }
}
