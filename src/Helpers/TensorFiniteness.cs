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
    /// 1 where the element is finite (neither NaN nor an infinity), 0 otherwise.
    /// Uses only GraphMode-recorded comparison and selection ops so it works for every numeric type (including Half)
    /// and inside compiled training traces.
    /// </summary>
    public static Tensor<T> FiniteMask<T>(IEngine engine, Tensor<T> values)
    {
        if (engine is null)
        {
            throw new ArgumentNullException(nameof(engine));
        }

        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var zeros = engine.TensorZerosLike(values);
        var nonNan = engine.TensorEquals(values, values);
        var withinRange = engine.TensorWhere(
            engine.TensorGreaterThan(engine.TensorAbs(values), ops.MaxValue),
            zeros,
            engine.TensorOnesLike(values));
        return engine.TensorWhere(nonNan, withinRange, zeros);
    }

    /// <summary>
    /// 1 where the element is finite, 0 otherwise, using <see cref="AiDotNetEngine.Current"/>.
    /// </summary>
    public static Tensor<T> FiniteMask<T>(Tensor<T> values) => FiniteMask(AiDotNetEngine.Current, values);

    /// <summary>
    /// 1 where the element is NaN or an infinity, 0 where it is finite.
    /// </summary>
    public static Tensor<T> NonFiniteMask<T>(IEngine engine, Tensor<T> values)
    {
        if (engine is null)
        {
            throw new ArgumentNullException(nameof(engine));
        }

        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var nonNan = engine.TensorEquals(values, values);
        var infinite = engine.TensorGreaterThan(engine.TensorAbs(values), ops.MaxValue);
        return engine.TensorWhere(nonNan, infinite, engine.TensorOnesLike(values));
    }

    /// <summary>
    /// 1 where the element is NaN or an infinity, 0 where it is finite.
    /// </summary>
    public static Tensor<T> NonFiniteMask<T>(Tensor<T> values) => NonFiniteMask(AiDotNetEngine.Current, values);
}
