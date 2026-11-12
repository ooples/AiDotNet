using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Pixel counts and masked means of segmentation masks, safe for a narrow numeric type.
/// </summary>
/// <remarks>
/// Counts are taken in <c>int</c>: summing a 0/1 mask in Half rounds every count above 2048 and overflows
/// to infinity above 65504 (a 512 x 512 image holds 262144 pixels). Means select the masked values
/// instead of multiplying by the mask, so a NaN or infinite value outside the mask cannot reach them.
/// </remarks>
internal static class MaskStatistics
{
    /// <summary>The number of non-zero entries of <paramref name="mask"/>, counted in <c>int</c>.</summary>
    internal static int CountNonZero<T>(Tensor<T> mask)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var nonZero = engine.ScalarMinusTensor(numOps.One, engine.TensorEquals(mask, numOps.Zero));
        return engine.TensorSum(nonZero.Cast<int>());
    }

    /// <summary>
    /// The mean of <paramref name="values"/> over the <paramref name="count"/> positions where
    /// <paramref name="mask"/> is non-zero, or zero when the mask is empty.
    /// </summary>
    /// <remarks>
    /// Each selected value is scaled by <c>1 / sqrt(count)</c> before the full engine sum (accumulated in
    /// float for Half) and once more after it, so the running total stays near <c>sqrt(count)</c> rather
    /// than <c>count</c> and cannot overflow a narrow <typeparamref name="T"/>.
    /// </remarks>
    internal static T MaskedMean<T>(Tensor<T> values, Tensor<T> mask, int count)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (count <= 0) return numOps.Zero;

        var engine = AiDotNetEngine.Current;
        T scale = numOps.FromDouble(1.0 / Math.Sqrt(count));
        var selected = engine.TensorWhere(mask,
            engine.TensorMultiplyScalar(values, scale), engine.TensorFullLike(values, numOps.Zero));
        return numOps.Multiply(engine.TensorSum(selected), scale);
    }

    /// <summary>
    /// The mean of <paramref name="values"/> over the positions whose label is <paramref name="label"/>,
    /// given that label's pixel <paramref name="count"/>.
    /// </summary>
    internal static T LabelMean<T>(Tensor<T> values, Tensor<int> labels, int label, int count)
        => MaskedMean(values, AiDotNetEngine.Current.TensorEquals(labels, label).Cast<T>(), count);
}
