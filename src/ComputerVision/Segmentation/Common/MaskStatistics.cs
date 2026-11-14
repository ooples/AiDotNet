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

    /// <summary>
    /// A 0/1 mask of the entries of <paramref name="values"/> strictly above the real number
    /// <paramref name="threshold"/>, decided exactly.
    /// </summary>
    /// <remarks>
    /// Comparing with <c>FromDouble(threshold)</c> is wrong whenever <typeparamref name="T"/> rounds the threshold
    /// up: a value equal to the rounded threshold is above the real one but fails <c>&gt;</c>. The threshold is
    /// therefore compared as its rounded value, with equality counting as above when the rounding went up.
    /// NaN is never above.
    /// </remarks>
    internal static Tensor<T> GreaterThan<T>(Tensor<T> values, double threshold)
    {
        if (double.IsNaN(threshold)) throw new ArgumentException("The threshold must be a number.", nameof(threshold));
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        T rounded = numOps.FromDouble(threshold);
        var above = engine.TensorGreaterThan(values, rounded);
        return numOps.ToDouble(rounded) > threshold
            ? engine.TensorAdd(above, engine.TensorEquals(values, rounded))
            : above;
    }

    /// <summary>
    /// A 0/1 mask of the logits whose sigmoid probability is strictly above <paramref name="probability"/>.
    /// </summary>
    /// <remarks>
    /// The sigmoid is strictly increasing, so <c>sigmoid(x) &gt; p</c> is exactly <c>x &gt; ln(p / (1 - p))</c>
    /// (LISA's <c>logit &gt; 0</c> for p = 0.5). Deciding on the logit keeps the decision exact: a sigmoid
    /// rounded to <typeparamref name="T"/> lands on <c>p</c> for every logit within half an ulp of 0.5 of the
    /// boundary (0.5 + 2.5e-9 is 0.5 in float, and Half's spacing near 0.5 is 4.9e-4).
    /// </remarks>
    internal static Tensor<T> SigmoidGreaterThan<T>(Tensor<T> logits, double probability)
    {
        if (double.IsNaN(probability)) throw new ArgumentException("The probability must be a number.", nameof(probability));
        double boundary = probability <= 0.0 ? double.NegativeInfinity
            : probability >= 1.0 ? double.PositiveInfinity
            : Math.Log(probability / (1.0 - probability));
        return GreaterThan(logits, boundary);
    }
}
