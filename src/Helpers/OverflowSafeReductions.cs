using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Whole-tensor reductions whose intermediate sums stay finite in narrow numeric types such as Half.
/// </summary>
/// <remarks>
/// In Half every value above 65504 is infinite, so a plain sum overflows long before the mean it feeds does:
/// 400 pixels of value 200, or 131072 values of 0.5, already sum past the limit. These reductions accumulate
/// in double and convert only the final result to T.
/// </remarks>
internal static class OverflowSafeReductions
{
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

        // The mean is accumulated and divided in double. Scaling by the largest magnitude and
        // dividing across blocks in T kept the sum finite, but the normalized entries and their mean were held
        // in T: in Half an entry below 2^-25 of the largest magnitude (and a mean below 2^-24) rounds to zero,
        // so many small entries next to one large one were lost from the mean.
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.FromDouble(engine.TensorMean(values.Contiguous().Cast<double>()));
    }
}
