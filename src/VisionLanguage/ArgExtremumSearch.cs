using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.VisionLanguage;

/// <summary>
/// First-position argmax/argmin over a tensor with integer positions, so the selected index is exact
/// for any <typeparamref name="T"/> (a low-precision type such as Half cannot hold every position) and
/// no infinity has to be representable in <typeparamref name="T"/>.
/// </summary>
/// <typeparam name="T">The numeric type of the searched values.</typeparam>
internal static class ArgExtremumSearch<T>
{
    /// <summary>
    /// Returns the first position holding the largest (or smallest) non-NaN value, or -1 when every value
    /// is NaN (or there are none). NaN values never win; on ties the lowest position wins.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="values">Values to search (read flattened).</param>
    /// <param name="largest">True for the maximum, false for the minimum.</param>
    /// <param name="best">The winning value (zero when -1 is returned).</param>
    /// <returns>The first winning position, or -1.</returns>
    internal static int FirstIndex(IEngine engine, Tensor<T> values, bool largest, out T best)
    {
        best = MathHelper.GetNumericOperations<T>().Zero;
        int n = values.Length;
        if (n == 0)
            return -1;

        var flat = values.Reshape([n]);
        var isNan = engine.TensorIsNan(flat);
        var positions = engine.TensorArange<int>(0, n, 1);
        var sentinel = Tensor<int>.CreateDefault([n], n);
        int firstValid = engine.TensorMinValue(engine.TensorWhere(isNan, sentinel, positions));
        if (firstValid >= n)
            return -1;

        // NaN positions take a real value so they cannot change the extremum, and are excluded below.
        var comparable = engine.TensorWhere(isNan, Tensor<T>.CreateDefault([n], flat[firstValid]), flat);
        best = largest ? engine.TensorMaxValue(comparable) : engine.TensorMinValue(comparable);
        var isBest = engine.TensorEquals(comparable, best).Cast<int>();
        return engine.TensorMinValue(engine.TensorWhere(isNan, sentinel, engine.TensorWhere(isBest, positions, sentinel)));
    }
}
