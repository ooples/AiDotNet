using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability;

/// <summary>
/// Helpers for aggregating per-instance feature attributions into a single global attribution vector.
/// </summary>
internal static class AttributionAggregation
{
    /// <summary>
    /// Global attribution = the mean absolute per-feature attribution across a set of local explanations.
    /// </summary>
    public static Vector<T> MeanAbsolute<T>(System.Collections.Generic.IEnumerable<IFeatureAttribution<T>> explanations)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double[]? sums = null;
        int count = 0;

        foreach (var explanation in explanations)
        {
            var attributions = explanation.GetFeatureAttributions();
            sums ??= new double[attributions.Length];
            int d = Math.Min(sums.Length, attributions.Length);
            for (int j = 0; j < d; j++) sums[j] += Math.Abs(numOps.ToDouble(attributions[j]));
            count++;
        }

        if (sums is null || count == 0) return new Vector<T>(0);

        var result = new Vector<T>(sums.Length);
        for (int j = 0; j < sums.Length; j++) result[j] = numOps.FromDouble(sums[j] / count);
        return result;
    }
}
