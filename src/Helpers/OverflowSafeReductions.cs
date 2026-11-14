using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Whole-tensor reductions whose intermediate sums stay finite in narrow numeric types such as Half.
/// </summary>
/// <remarks>
/// In Half every value above 65504 is infinite, so a plain sum overflows long before the mean it feeds does:
/// 400 pixels of value 200, or 131072 values of 0.5, already sum past the limit. These reductions scale the
/// values into [-1, 1] and divide them across blocks before summing, then undo the scaling on the result.
/// </remarks>
internal static class OverflowSafeReductions
{
    /// <summary>
    /// The most scaled values that are summed per block; their sum stays at or below this in magnitude.
    /// </summary>
    private const int BlockSize = 16384;

    /// <summary>
    /// The mean of all entries of <paramref name="values"/>, computed so the intermediate sum cannot overflow T.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="values">A non-empty tensor.</param>
    /// <returns>The mean in T; zero when every entry is zero.</returns>
    public static T Mean<T>(Tensor<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length == 0)
        {
            throw new ArgumentException("Cannot compute the mean of an empty tensor.", nameof(values));
        }

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T largest = engine.TensorMaxValue(engine.TensorAbs(values));
        if (ops.IsNaN(largest) || !ops.GreaterThan(largest, ops.Zero))
        {
            return ops.IsNaN(largest) ? largest : ops.Zero;
        }

        // Each scaled entry lies in [-1 / blocks, 1 / blocks], so the running sum stays within BlockSize.
        T blocks = ops.FromDouble((values.Length + BlockSize - 1) / BlockSize);
        var scaled = engine.TensorDivideScalar(engine.TensorDivideScalar(values, largest), blocks);
        return ops.Multiply(ops.Multiply(engine.TensorMean(scaled), blocks), largest);
    }
}
